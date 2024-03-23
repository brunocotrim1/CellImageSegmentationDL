import os
import numpy as np
import torch
from PIL import Image
import cv2
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import PIL
import torchvision.transforms.functional as TF
import torch
import torchvision
from evaluation import evaluate
from Unet import UNet
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from skimage import io
from torch.optim import Adam
import torch.nn.functional as F
import time
import torch.optim as optim
import random
def dice_coefficient(predicted, target):
    intersection = torch.sum(predicted * target)
    union = torch.sum(predicted) + torch.sum(target)
    dice = (2. * intersection + 1e-7) / (union + 1e-7)
    return dice

def dice_loss(predicted, target):
    return 1 - dice_coefficient(predicted, target)

class CustomDiceCELoss(torch.nn.Module):
    def __init__(self):
        super(CustomDiceCELoss, self).__init__()

    def forward(self, pred, target):  
        # Calculate probabilities from logits using softmax
        pred_probs = F.softmax(pred, dim=1)
        # Calculate Dice loss for each class
        intersection = torch.sum(pred_probs * target, dim=(2, 3))
        union = torch.sum(pred_probs, dim=(2, 3)) + torch.sum(target, dim=(2, 3))
        dice_score = torch.mean((2. * intersection + 1) / (union + 1))  # Smoothing added to avoid division by zero

        # Calculate Cross-Entropy loss
        ce_loss = F.cross_entropy(pred, target.argmax(dim=1))

        # Combine Dice loss and Cross-Entropy loss
        combined_loss = 1 - dice_score + ce_loss
        return combined_loss
    
class CustomDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_files = sorted(os.listdir(image_dir))
        self.label_files = sorted(os.listdir(label_dir))
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        label_name = self.label_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        label_path = os.path.join(self.label_dir, label_name)
        image = np.load(img_path)
        label = np.load(label_path)
        #image, label = random_patch(image, label)
        label = torch.from_numpy(label)
        image = torch.from_numpy(image)
        return image, label
    
def train_batch(images, labels, model, optimizer,device):
    model.train()
    images = images.to(device)
    labels = labels.to(device)
    for param in model.parameters():
        param.grad = None
    outputs = model(images)
    loss = CustomDiceCELoss()(outputs, labels)
    loss.backward()
    optimizer.step()
    return loss.item()

def train_config(model, optimizer, device, data_loader_train, data_loader_test, num_epochs,model_name):
    patience = 3  # Number of epochs to wait for improvement
    counter = 0  # Counter to track epochs without improvement
    best_loss = np.inf  # Initialize with a large value
    losses = []  # List to store epoch losses
    val_iou = []
    for epoch in range(num_epochs):
        start = time.time()
        print(f"Epoch {epoch+1} of {num_epochs}")
        epoch_loss = 0.0
        for images, labels in data_loader_train:
            loss = train_batch(images, labels, model, optimizer, device)
            epoch_loss += loss * images.size(0)
        epoch_loss /= len(data_loader_train.dataset)
        losses.append(epoch_loss)  # Store epoch loss
        eval = evaluate(model, data_loader_test, device,simple=True,plot=False,save_dir=None)
        val_iou.append(eval['iou'])
        print(f"Epoch [{epoch+1}/{num_epochs}], Dice Loss: {epoch_loss:.4f}, took: {time.time()-start:.4f} seconds,with iou score: {eval['iou']}")
        # Implement early stopping
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            counter = 0  # Reset counter since there's improvement
        else:
            counter += 1  # Increment counter if no improvement
            if counter >= patience:
                print(f'Early stopping at epoch {epoch+1} as no improvement in validation loss.')
                break  # Stop training loop

    plot_epoch_loss(losses)
    plot_epoch_iou(val_iou)
    torch.save(model.state_dict(), f'./models/{model_name}.pth')

def plot_epoch_loss(losses,model_name):
    plt.plot(range(0, len(losses)), losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"plots/training_loss_{model_name}.png")  # Save the plot to a file
    plt.close()  # Close the plot to free memory

def plot_epoch_iou(val_iou,model_name):
    plt.plot(range(0, len(val_iou)), val_iou, label='Validation IoU')
    plt.xlabel('Epochs')
    plt.ylabel('IoU')
    plt.title('Validation IoU Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"plots/testing_iou_{model_name}.png")  # Save the plot to a file
    plt.close()  # Close the plot to free memory

def main():
    # Define your transformations
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Define paths to your preprocessed data
    image_dir = './Train_Pre_3class/images'
    label_dir = './Train_Pre_3class/labels'

    image_dir_test = './Test_Pre_3class/images'
    label_dir_test = './Test_Pre_3class/labels'
    # Create an instance of your custom dataset
    dataset_train = CustomDataset(image_dir, label_dir, transform)
    dataset_test = CustomDataset(image_dir_test, label_dir_test, transform)
    # Create a data loader
    batch_size = 32
    data_loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True,pin_memory=True,num_workers=0)
    data_loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False,pin_memory=True,num_workers=0)
    device = torch.device("mps" if torch.backends.mps.is_available else "cpu")
    model = UNet(3).to(device)

    #model = UnetArb(n_class=3).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    optimizer =  optim.AdamW(model.parameters(), lr=0.001)
    # Iterate through the data loader
    num_epochs = 40  # Adjust as needed
    patience = 3  # Number of epochs to wait for improvement
    counter = 0  # Counter to track epochs without improvement
    best_loss = np.inf  # Initialize with a large value
    #losses = []  # List to store epoch losses

    train_config(model, optimizer, device, data_loader_train=data_loader_train
                 ,data_loader_test=data_loader_test, num_epochs=num_epochs,model_name='model_40epoch_2')




    # for epoch in range(num_epochs):
    #     model.train()  # Set model to training mode
    #     start = time.time()
    #     print(f"Epoch {epoch+1} of {num_epochs}")
    #     epoch_loss = 0.0
    #     # Iterate through the data loader
    #     for images, labels in data_loader:
    #         # Move images and labels to the GPU
    #         images = images.to(device)
    #         labels = labels.to(device)
    #         # Zero the parameter gradients in a more efficient way
    #         for param in model.parameters():
    #             param.grad = None
    #         # Forward pass
    #         outputs = model(images)
    #         # Compute the Dice Loss
    #         loss = CustomDiceCELoss()(outputs, labels)
    #         # Backward pass and optimization
    #         loss.backward()
    #         optimizer.step()
    #         epoch_loss += loss.item() * images.size(0)
    #         #print(f"Batch took: {time.time()-start2:.4f} seconds")

    #     # Calculate average epoch loss
    #     epoch_loss /= len(data_loader.dataset)
    #     losses.append(epoch_loss)  # Store epoch loss

        # print(f"Epoch [{epoch+1}/{num_epochs}], Dice Loss: {epoch_loss:.4f}, took: {time.time()-start:.4f} seconds")
        # # Implement early stopping
        # if epoch_loss < best_loss:
        #     best_loss = epoch_loss
        #     counter = 0  # Reset counter since there's improvement
        # else:
        #     counter += 1  # Increment counter if no improvement
        #     if counter >= patience:
        #         print(f'Early stopping at epoch {epoch+1} as no improvement in validation loss.')
        #         break  # Stop training loop

    # Plotting loss over epochs
    # plt.plot(range(0, len(losses)), losses, label='Training Loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.title('Training Loss Over Epochs')
    # plt.legend()
    # plt.grid(True)
    # plt.savefig('training_loss_plot_35.png')  # Save the plot to a file
    # plt.close()  # Close the plot to free memory
    # torch.save(model.state_dict(), './model35.pth')
if __name__ == "__main__":
    main()