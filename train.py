import os
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
from evaluation import evaluate
from Unet import UNet
import os
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F
import time
import torch.optim as optim
import argparse
from FCN import FastFCN
import torchvision.models as models
# class CustomDiceCELoss(torch.nn.Module):
#     def __init__(self):
#         super(CustomDiceCELoss, self).__init__()

#     def forward(self, pred, target):  
#         # Calculate Dice loss
#         numerator = 2 * torch.sum(pred * target, dim=(2, 3))
#         denominator = torch.sum(pred + target, dim=(2, 3))
#         dice_score = torch.mean(numerator / denominator)

#         # Calculate Cross-Entropy loss
#         ce_loss = F.cross_entropy(pred, target.argmax(dim=1))

#         # Combine Dice loss and Cross-Entropy loss
#         combined_loss = 1 - dice_score + ce_loss

#         return combined_losss
class CustomDiceCELoss(torch.nn.Module):
    def __init__(self):
        super(CustomDiceCELoss, self).__init__()

    def forward(self, pred, target):  
        # Calculate probabilities from logits using softmax
        pred_probs = F.softmax(pred, dim=1)
        
        # Calculate Dice loss for each class
        intersection = torch.sum(pred_probs * target, dim=(2, 3))
        union = torch.sum(pred_probs, dim=(2, 3)) + torch.sum(target, dim=(2, 3))
        dice_score = torch.mean((2. * intersection + 1e-5) / (union + 1e-5))  # Smoothing added to avoid division by zero

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
    
def train_batch(images, labels, model, optimizer,device,pretrained=False):
    model.train()
    images = images.to(device)
    labels = labels.to(device)
    for param in model.parameters():
        param.grad = None
    outputs = model(images)
    if pretrained:
        outputs = outputs['out']
    loss = CustomDiceCELoss()(outputs, labels)
    loss.backward()
    optimizer.step()
    return loss.item()

def train_config(model, optimizer, device, data_loader_train, data_loader_test, num_epochs,model_name,pretrained = False):
    patience = 7  # Number of epochs to wait for improvement
    counter = 0  # Counter to track epochs without improvement
    best_loss = np.inf  # Initialize with a large value
    losses = []  # List to store epoch losses
    val_f1 = []
    for epoch in range(num_epochs):
        start = time.time()
        print(f"Epoch {epoch+1} of {num_epochs}")
        epoch_loss = 0.0
        for images, labels in data_loader_train:
            loss = train_batch(images, labels, model, optimizer, device,pretrained=pretrained)
            epoch_loss += loss * images.size(0)
            print(f"Batch took: {time.time()-start:.4f} seconds with loss: {loss}")
        epoch_loss /= len(data_loader_train.dataset)
        losses.append(epoch_loss)  # Store epoch loss
        eval = evaluate(model, data_loader_test, device,simple=True,plot=False,save_dir=None,pretrained=pretrained)
        val_f1.append(eval['f1_score'])
        print(f"Epoch [{epoch+1}/{num_epochs}], DiceCE Loss: {epoch_loss:.4f}, took: {time.time()-start:.4f} seconds,with F1 score: {eval['f1_score']}")
        # Implement early stopping
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            counter = 0  # Reset counter since there's improvement
        else:
            counter += 1  # Increment counter if no improvement
            if counter >= patience:
                print(f'Early stopping at epoch {epoch+1} as no improvement in validation loss.')
                break  # Stop training loop

    plot_epoch_loss(losses,model_name=model_name)
    plot_epoch_f1(val_f1,model_name=model_name)
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

def plot_epoch_f1(val_iou,model_name):
    plt.plot(range(0, len(val_iou)), val_iou, label='Validation F1')
    plt.xlabel('Epochs')
    plt.ylabel('F1')
    plt.title('Validation F1 Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"plots/testing_F1_{model_name}.png")  # Save the plot to a file
    plt.close()  # Close the plot to free memory

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-epochs', default=40, type=int,
                        help="""Number of epochs to train for.""")
    parser.add_argument('-optimizer',
                        choices=['sgd', 'adam'], default='adam')
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Learning rate for parameter updates""")
    opt = parser.parse_args()
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
    batch_size = 8
    data_loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True,pin_memory=True,num_workers=0)
    data_loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False,pin_memory=True,num_workers=0)
    device = torch.device("mps" if torch.backends.mps.is_available else "cpu")

    # Iterate through the data loader
    config = "{}-{}-{}".format(opt.learning_rate, opt.optimizer, opt.epochs).replace('.','')
    name='UNET-validation-{}'.format(config)
    train = True
    #model = FastFCN(3,256).to(device)
    #model = UNet(3).to(device)
    from torchvision.models.segmentation import deeplabv3_resnet50
    model = deeplabv3_resnet50(pretrained=False, num_classes=3,weights = None).to(device)
    from networks.vit_seg_modeling import VisionTransformer as ViT_seg
    from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
    #model = SwinUNet(3).to(device)
    # Example usage:
    config_vit = CONFIGS_ViT_seg["R50-ViT-B_16"]
    config_vit.n_classes = 3
    config_vit.n_skip = 3
    if "R50-ViT-B_16".find('R50') != -1:
        config_vit.patches.grid = (int(256 / 16), int(256 / 16))
    model = ViT_seg(config_vit, img_size=256, num_classes=config_vit.n_classes).to(device)
    if train:
        num_epochs = opt.epochs
        
        if opt.optimizer == 'adam':
            optimizer = optim.AdamW(model.parameters(), lr=opt.learning_rate)
        else:
            learning_rate = 0.01
            decay_rate = learning_rate / num_epochs
            momentum = 0.8
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, 
                                  momentum=momentum, weight_decay=decay_rate, nesterov=False)
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total number of parameters: {total_params}")
        train_config(model, optimizer, device, data_loader_train=data_loader_train
                 ,data_loader_test=data_loader_test, num_epochs=num_epochs,model_name=name,pretrained = False)
    else:
        model.load_state_dict(torch.load(f'./{name}.pth'))
        
        print(evaluate(model, data_loader_test, device,simple=False,plot=True,save_dir='plots/'))



    # for epoch in range(num_epochs):
    #     model.train()  # Set model to training mode
    #     start = time.time()
    #     print(f"Epoch {epoch+1} of {num_epochs}")
    #     epoch_loss = 0.0
    #     # Iterate through the data loader
    #     for images, labels in data_loader_train:
    #         # Move images and labels to the GPU
    #         images = images.to(device)
    #         labels = labels.to(device)
    #         # Zero the parameter gradients in a more efficient way
    #         for param in model.parameters():
    #             param.grad = None
    #         # Forward pass
    #         outputs = model(images)
    #         # Compute the Dice Loss
    #         print("Time to forward was: ",time.time()-start)
    #         loss = CustomDiceCELoss()(outputs, labels)
    #         print("Time to loss batch was2: ",time.time()-start)

    #         # Backward pass and optimization
    #         loss.backward()
    #         optimizer.step()
    #         epoch_loss += loss.item() * images.size(0)
    #         #print(f"Batch took: {time.time()-start2:.4f} seconds")

    #     # Calculate average epoch loss
    #     epoch_loss /= len(data_loader_train.dataset)
    #     losses.append(epoch_loss)  # Store epoch loss

    #     print(f"Epoch [{epoch+1}/{num_epochs}], Dice Loss: {epoch_loss:.4f}, took: {time.time()-start:.4f} seconds")
    #     # Implement early stopping
    #     if epoch_loss < best_loss:
    #         best_loss = epoch_loss
    #         counter = 0  # Reset counter since there's improvement
    #     else:
    #         counter += 1  # Increment counter if no improvement
    #         if counter >= patience:
    #             print(f'Early stopping at epoch {epoch+1} as no improvement in validation loss.')
    #             break  # Stop training loop

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