import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from evaluation import evaluate
from Unet import UNet
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
import time
import torch.optim as optim
import argparse
from FCN import FastFCN
import torchvision.models as models
from torchvision.models.segmentation import deeplabv3_resnet50
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from UnetV2 import UnetV2
import os
import cv2
import random
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from data_augmentation import *
#https://github.com/Beckschen/TransUNet/tree/main
torch.set_printoptions(threshold=float('inf'))
import torch
import torch.nn.functional as F

class MyFocalLoss(torch.nn.Module):
    def __init__(self, alpha=[1,1,1], gamma=2, reduction='mean',device='cuda'):
        super(MyFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.alpha = torch.tensor(alpha).to(device)
    def forward(self, preds, targets):
        # Compute cross entropy
        ce_loss = F.cross_entropy(preds, targets.float(), reduction='none',weight=self.alpha)

        # Compute focal loss
        pt = torch.exp(-ce_loss)
        # if(self.alpha is not None):
        #     focal_loss = (self.alpha * (1 - pt) ** self.gamma * ce_loss)
        # else:
        focal_loss = ((1 - pt) ** self.gamma * ce_loss)

        # Apply reduction
        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss
class CustomDiceCELoss(torch.nn.Module):
    def __init__(self):
        super(CustomDiceCELoss, self).__init__()

    def forward(self, pred, target):  
        # Calculate probabilities from logits using softmax
        # pred_probs = F.softmax(pred, dim=1)
        
        # # Exclude background class (assuming it's the first channel)
        # pred_probs_exc = pred_probs[:, 1:, :, :]
        # target_exc = target[:, 1:, :, :]
        
        # # Calculate Dice loss for each class
        # intersection = torch.sum(pred_probs_exc * target_exc, dim=(1, 2, 3))
        # union = torch.sum(pred_probs_exc, dim=(1, 2, 3)) + torch.sum(target_exc, dim=(1, 2, 3))
        # dice_score = torch.mean((2. * intersection + 1e-5) / (union + 1e-5))  # Smoothing added to avoid division by zero

        # Calculate Cross-Entropy loss
        #ce_loss = F.cross_entropy(pred, target.argmax(dim=1))
        
        # Combine Dice loss and Cross-Entropy loss
        #combined_loss = 1 - dice_score + ce_loss
        #combined_loss = 0.5 * (1 - dice_score) + 0.5 * ce_loss
        #combined_loss = 1 - dice_score
        #combined_loss = ce_loss
        
        ce_loss = F.cross_entropy(pred, target.argmax(dim=1))

        pred = F.softmax(pred[:,1:,:,:], dim=1)
        target = target[:,1:,:,:]
        intersection = torch.sum(pred * target, dim=(2, 3))
        union = torch.sum(pred, dim=(2, 3)) + torch.sum(target, dim=(2, 3))
        dice = torch.mean((2. * intersection + 1e-5) / (union + 1e-5))
        return 1-dice+ce_loss




# Example usage:
# Assuming pred and target are tensors of shape (batch_size, num_classes, width, height)
# loss = focal_loss(pred, target)
# print(loss)


class CustomDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None,test = False):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_files = sorted(os.listdir(image_dir))
        self.label_files = sorted(os.listdir(label_dir))
        self.test = test
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
        # label = torch.from_numpy(label)
        # image = torch.from_numpy(image)
        if not self.test:
            # image = np.transpose(image, (2, 0, 1))
            # if image.shape[0] ==256:
            #     image = np.transpose(image, (1, 0, 2))
            # if label.shape[0] ==256:
            #     label = np.transpose(label, (1, 0, 2))
            #image = scale_intensity(image)
            # image,label = spatial_pad(image,label, (3, 256, 256))
            # image, label = random_spatial_crop(image, label, (256, 256))
            image, label = random_axis_flip(image, label)
            image, label = random_rotate_90(image, label)
            image = random_gaussian_noise(image)
            image = random_adjust_contrast(image)
            image = random_gaussian_smooth(image)
            image = random_histogram_shift(image)
            #image, label = random_zoom(image, label, (256, 256), 0.8, 1.0)
            #plot_image_and_label(image, label)
        # else:
        #     image = scale_intensity(image)
            
        image = image.copy()
        label = label.copy()

        imageDev = torch.from_numpy(image)
        del image
        labelDev = torch.from_numpy(label)
        del label
        return imageDev, labelDev

def plot_image_and_label(image, label):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    # Plot image
    axes[0].imshow(image.transpose(1, 2, 0))
    axes[0].set_title('Image')
    axes[0].axis('off')

    # Plot label
    axes[1].imshow(label.argmax(axis=0), cmap='viridis', vmin=0, vmax=2)  # Assuming 3 classes
    axes[1].set_title('Label')
    axes[1].axis('off')

    plt.show()
def train_batch(images, labels, model, optimizer,device,pretrained=False):
    model.train()
    images = images.float()
    labels = labels.long()
    imagesDevice = images.to(device)
    labelsDevice = labels.to(device)
    del images,labels #Free up ram space
    for param in model.parameters():
        param.grad = None
    outputs = model(imagesDevice)
    if pretrained:
        outputs = outputs['out']
    #loss = MyFocalLoss(device=device)(outputs, labelsDevice)
    loss = CustomDiceCELoss()(outputs, labelsDevice)
    loss.backward()
    optimizer.step()
    del imagesDevice,labelsDevice,outputs
    return loss.item()

def train_config(model, optimizer, device, data_loader_train, data_loader_test, num_epochs,model_name,pretrained = False,scheduler=None):
    patience = 7  # Number of epochs to wait for improvement
    counter = 0  # Counter to track epochs without improvement
    best_loss = np.inf  # Initialize with a large value
    losses = []  # List to store epoch losses
    val_loss= []
    for epoch in range(num_epochs):
        start = time.time()
        print(f"Epoch {epoch+1} of {num_epochs}")
        if scheduler is not None:
            scheduler.step(epoch)
        epoch_loss = 0.0
        for images, labels in data_loader_train:
            size = images.size(0)
            loss = train_batch(images, labels, model, optimizer, device,pretrained=pretrained)
            epoch_loss += loss * size
            #print(f"Batch took: {time.time()-start:.4f} seconds with loss: {loss}")
        epoch_loss /= len(data_loader_train.dataset)
        losses.append(epoch_loss)  # Store epoch loss
        eval = evaluate(model, data_loader_test, device,simple=True,plot=False,save_dir=None,pretrained=pretrained)
        val_loss.append(eval['loss'])
        print(f"Epoch [{epoch+1}/{num_epochs}], DiceCE Loss: {epoch_loss:.4f}, took: {time.time()-start:.4f} seconds,with eval Dice Eval: {eval['loss']}")
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
    plot_epoch_f1(val_loss,model_name=model_name)
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
    parser.add_argument('-learning_rate', type=float, default=0.0001,
                        help="""Learning rate for parameter updates""")
    parser.add_argument('-batch', type=int, default=16,
                    help="""Image Batch size for training""")
    parser.add_argument('-size', type=int, default=256,
                    help="""Image Batch size for training""")
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
    if not os.path.exists("./plots"):
        os.makedirs("./plots")
    if not os.path.exists("./models"):
        os.makedirs("./models")
    if not os.path.exists("./visualizations"):
        os.makedirs("./visualizations")
    # Create an instance of your custom dataset
    dataset_train = CustomDataset(image_dir, label_dir, transform)
    dataset_test = CustomDataset(image_dir_test, label_dir_test, transform,test=True)
    # Create a data loader
    batch_size = opt.batch
    data_loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True,pin_memory=True,num_workers=0)
    data_loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False,pin_memory=True,num_workers=0)
    device = torch.device("mps" if torch.backends.mps.is_available else "cpu")

    # Iterate through the data loader
    
    config = "{}-{}-{}-{}-{}".format(opt.learning_rate, opt.optimizer, opt.epochs,batch_size,opt.size).replace('.','')
    train = False
    size = opt.size
    model = "MyUNETV2"
    if model == "FCN":
        pretrained = False
        model = FastFCN(3,size).to(device)
        name = "FCN-validation-{}".format(config)
    elif model == "UNET":
        pretrained = False
        model = UNet(3).to(device)
        name = "UNET-validation-{}".format(config)
    elif model == "MyUNETV2":
        pretrained = False
        model = UnetV2(3,3).to(device)             
        name = "UNETV2-validation-{}".format(config)
    elif model == "DEEPLAB":
        pretrained = True
        model = models.segmentation.deeplabv3_resnet50(pretrained=False, num_classes=3,weights = None).to(device)
        name = "DEEPLAB-validation-{}".format(config)
    elif model == "VIT":
        pretrained = False
        vit = "R50-ViT-B_16"
        config_vit = CONFIGS_ViT_seg[vit]
        config_vit.n_classes = 3
        config_vit.n_skip = 3
        if vit.find('R50') != -1:
            config_vit.patches.grid = (int(size / 16), int(size / 16))
        model = ViT_seg(config_vit, img_size=size, num_classes=config_vit.n_classes).to(device)
        name = "VIT-validation-{}".format(config)
    elif model == "UNETPLUSPLUS":
        pretrained = False
        
        from transformers import AutoModelForSemanticSegmentation
        from hf_config import UnetConfig  # Import the provided config class
        from hf_model import HFUnetPlusPlus  # Import the provided model class
        # Define your configuration
        my_config = UnetConfig(
            encoder_name="resnet18",
            num_classes=3,
            input_channels=3,
            decoder_channels=(1024, 512, 256, 128, 64)
        )

        # Load the model using AutoModelForSemanticSegmentation
        model = HFUnetPlusPlus(config=my_config).to(device)
        name = "HFUNET-validation-{}".format(config)

    print("Training model: ",name)

    if train:
        num_epochs = opt.epochs
        
        if opt.optimizer == 'adam':
            optimizer = optim.AdamW(model.parameters(), lr=opt.learning_rate)
            scheduler = None
        else:
            learning_rate = 0.001
            #decay_rate = learning_rate / num_epochs
            momentum = 0.9
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, 
                                  momentum=momentum, nesterov=True)
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=3, T_mult=1)
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total number of parameters: {total_params}")
        train_config(model, optimizer, device, data_loader_train=data_loader_train
                 ,data_loader_test=data_loader_test, num_epochs=num_epochs,model_name=name,pretrained = pretrained,scheduler=scheduler)
    else:
        model.load_state_dict(torch.load(f'./models/{name}.pth'))
        
        print(evaluate(model, data_loader_test, device,simple=False,plot=True,save_dir='visualizations/'))



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