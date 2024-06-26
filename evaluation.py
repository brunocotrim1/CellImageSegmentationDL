from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import f1_score
import numpy as np
import torch.nn.functional as F
import os
from matplotlib.colors import ListedColormap
from matplotlib.colors import ListedColormap
from sklearn.metrics import precision_recall_fscore_support, jaccard_score
import argparse
import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import OrderedDict
import pandas as pd
from skimage import segmentation
import tifffile as tif
import os
join = os.path.join
from tqdm import tqdm
# Define custom colormap
custom_colors = [(0, 0, 0),  # Background (black)
                 (0, 0, 1),  # Inside of cells (blue)
                 (1, 0, 0)]  # Boundary of cells (red)
custom_cmap = ListedColormap(custom_colors)
np.set_printoptions(threshold=np.inf)
# Create a rainbow colormap


encoding_map = {
    0: [1, 0, 0],  # background
    1: [0, 1, 0],  # interior
    2: [0, 0, 1]   # boundary
}

def one_hot_encode(mask, encoding_map):
    num_classes = len(encoding_map)
    batch_size = mask.shape[0]
    encoded_masks = torch.zeros((batch_size, num_classes, mask.shape[1], mask.shape[2]), dtype=torch.uint8)
    for class_id, encoding in encoding_map.items():
        encoded_masks[:, class_id] = (mask == class_id).to(torch.uint8)
    return encoded_masks


def dice_coefficient(y_true, y_pred):
    jaccard = jaccard_score(y_true, y_pred, average='weighted',zero_division=0)
    return 2*jaccard / (1 + jaccard)


def calculate_metrics(predicted, labels):
    # Flatten the predicted and true labels
    predicted_flat = predicted.flatten()
    labels_flat = labels.flatten()

    # Calculate precision, recall, F1-score, and support
    precision, recall, f1_score, _ = precision_recall_fscore_support(labels_flat, predicted_flat, average='weighted',zero_division=0)

    # Calculate intersection over union (IoU)
    iou = jaccard_score(labels_flat, predicted_flat, average='weighted',zero_division=0)
    dice = dice_coefficient(predicted_flat.numpy(), labels_flat.numpy())

    return precision, recall, f1_score, iou, dice
def calculate_accuracy(mask_pred, mask_true):
    # Flatten masks to 1D arrays
    pred_flat = mask_pred.flatten()
    true_flat = mask_true.flatten()
    
    # Count total pixels
    total_pixels = len(pred_flat)
    
    # Count correctly classified pixels
    correct_pixels = np.sum(pred_flat == true_flat)
    
    # Calculate accuracy
    accuracy = correct_pixels / total_pixels
    
    return accuracy


def predict(model, truth,pretrained=False,simple=False):
    model.eval()
    with torch.no_grad():
        predicted = model(truth)
        if pretrained:
            predicted = predicted['out']
        if simple:
            return predicted
    probabilities = F.softmax(predicted, dim=1)
    _, predicted = torch.max(probabilities, 1)
    return predicted


import cv2
import numpy as np
import torch

def watershed_postprocessing(mask):
    batch_size = mask.size(0)
    segmented_masks = []

    for i in range(batch_size):
        # Convert tensor to numpy array
        mask_np = mask[i].cpu().numpy() if mask.is_cuda else mask[i].numpy()

        # Convert mask to uint8
        mask_uint8 = (mask_np * 255).astype(np.uint8)

        # Invert the binary mask to obtain background region
        background = cv2.bitwise_not(mask_uint8)

        # Apply thresholding to obtain binary image
        _, thresh = cv2.threshold(mask_uint8, 1, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create markers for watershed segmentation
        markers = np.zeros_like(mask_uint8, dtype=np.int32)

        # Mark background region
        cv2.drawContours(markers, contours, -1, 1, -1)

        # Mark boundary region
        markers[mask_uint8 == 2] = 2

        # Apply watershed segmentation
        markers = cv2.watershed(cv2.cvtColor(mask_uint8, cv2.COLOR_GRAY2BGR), markers)

        # Convert markers back to segmented mask
        segmented_mask = np.zeros_like(mask_np, dtype=np.uint8)
        segmented_mask[markers == -1] = 2  # Boundary
        segmented_mask[markers == 1] = 1   # Inside
        segmented_mask[background == 255] = 0   # Outside

        # Convert back to tensor and append to list
        segmented_masks.append(torch.from_numpy(segmented_mask))
    return torch.stack(segmented_masks)

def evaluate(model, loader, device,simple=True,plot=False,save_dir="visualization",pretrained=False, loss = None):
    model.eval()
    f1_score_acc = []
    recall_acc  = []
    precision_acc  = []
    iou_acc  = []
    pixel_acc_acc  = []
    challenge_evaluation_acc = []
    dice_acc = []
    trainLossAcc = []
    for batch_idx,(images, labels) in enumerate(loader):
        #print(f"Predicting Batch ")
        images = images.to(device)  # Move images to GPU
        predicted = predict(model, images,pretrained,simple=simple) # Model prediction and conversion to NumPy on GPU
        if simple:
            labelsOneHot = one_hot_encode(labels, encoding_map).to(torch.float32).to(device)  # Move labels to GPU
            trainLossAcc.append(loss(predicted, labelsOneHot).cpu().numpy())
            del labelsOneHot
            probabilities = F.softmax(predicted, dim=1)
            _, predicted = torch.max(probabilities, 1)
        predicted = predicted.cpu()
        predicted = watershed_postprocessing(predicted)
        if plot:
            plot_results(images, labels, predicted, save_dir, batch_idx)
        for i in range(len(predicted)):
            if not simple: 
                precision, recall, f1_score, iou, dice = calculate_metrics(predicted[i], labels[i])
                f1_score_acc.append(f1_score)
                recall_acc.append(recall)
                precision_acc.append(precision)
                dice_acc.append(dice)
                iou_acc.append(iou)
                pixel_acc_acc.append(calculate_accuracy(predicted[i].numpy(), labels[i].numpy()))
            else:
                predicted_flat = predicted.flatten()
                labels_flat = labels.flatten()
                dice_acc.append(dice_coefficient(predicted_flat.numpy(), labels_flat.numpy()))
        del images, labels, predicted
    if simple:
        return {'loss': np.mean(trainLossAcc), 'accuracy': np.mean(dice_acc)}
    else:
        return {'iou': "{:.4f}".format(np.mean(iou_acc)),
                       'dice': "{:.4f}".format(np.mean(dice_acc)),
                       'pixel_accuracy': "{:.4f}".format(np.mean(pixel_acc_acc)),
                       'precision': "{:.4f}".format(np.mean(precision_acc)),
                       'recall': "{:.4f}".format(np.mean(recall_acc)),
                       'f1_score': "{:.4f}".format(np.mean(f1_score_acc))}



def plot_results(images, labels, predicted, save_dir, batch_idx):
    for i in range(len(predicted)):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot original image
        axes[0].imshow(np.clip(np.transpose(images[i].cpu().numpy(), (1, 2, 0)), 0, 1))
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Define custom colormap
        custom_cmap = plt.cm.get_cmap('viridis', 3)
        custom_cmap.set_under('black')  # Set color for values less than vmin
        
        # Plot ground truth mask
        axes[1].imshow(np.clip(labels[i], 0, 2), cmap=custom_cmap)
        axes[1].contour(labels[i] == 1, colors='white', levels=[0.5])  # Overlay white lines on boundary
        axes[1].set_title('Ground Truth Mask')
        axes[1].axis('off')
        
        # Plot predicted mask with custom colormap
        axes[2].imshow(np.clip(np.transpose(images[i].cpu().numpy(), (1, 2, 0)), 0, 1))
        axes[2].imshow(np.clip(predicted[i], 0, 1), cmap=custom_cmap, alpha=0.7, vmin=0.5)
        axes[2].contour(predicted[i] == 1, colors='white', levels=[0.5])  # Overlay white lines on boundary
        axes[2].set_title('Predicted Mask on Original Image')
        axes[2].axis('off')
        
        save_path = os.path.join(save_dir, f'batch_{batch_idx}_image_{i}_prediction.png')
        plt.savefig(save_path)
        plt.close()

# def plot_results(images, labels, predicted, save_dir, batch_idx):
#         for i in range(len(predicted)):
#             fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
#             # Plot original image
#             axes[0].imshow(np.clip(np.transpose(images[i].cpu().numpy(), (1, 2, 0)),0,1))
#             axes[0].set_title('Original Image')
#             axes[0].axis('off')
            
#             # Plot ground truth mask
#             axes[1].imshow(np.clip(labels[i],0,1), cmap='gray')
#             axes[1].set_title('Ground Truth Mask')
#             axes[1].axis('off')
            
#             # Plot predicted mask with custom colormap
#             axes[2].imshow(np.clip(np.transpose(images[i].cpu().numpy(), (1, 2, 0)),0,1))
#             axes[2].imshow(np.clip(predicted[i],0,1), cmap=custom_cmap, alpha=0.7)
#             axes[2].set_title('Predicted Mask on Original Image')
#             axes[2].axis('off')
            
#             save_path = os.path.join(save_dir, f'batch_{batch_idx}_image_{i}_prediction.png')
#             plt.savefig(save_path)
#             plt.close()

# def main():
#     # Define the path to your .pth file
#     file_path = "./model35.pth"

#     # Instantiate your model
#     model = UNet(3)

#     image_dir = './Test_Pre_3class/images'
#     label_dir = './Test_Pre_3class/labels'
#     save_dir = "visualization"
#     os.makedirs(save_dir, exist_ok=True)
#     transform = transforms.Compose([
#         transforms.ToTensor()
#     ])

#     dataset = CustomDataset(image_dir, label_dir, transform)
#     # Load the parameters
#     model.load_state_dict(torch.load(file_path))
#     test_loader = DataLoader(dataset, batch_size=30, shuffle=False, num_workers=0, pin_memory=True)
#     device = torch.device("mps" if torch.backends.mps.is_available else "cpu")
#     model = model.to(device)
#     print("Model loaded into device")

#     print("Model Evaluation:", evaluate(model, test_loader, device,save_dir=save_dir,plot=True))









    # f1_score_acc = []
    # recall_acc  = []
    # precision_acc  = []
    # iou_acc  = []
    # pixel_acc_acc  = []

    # for batch_idx,(images, labels) in enumerate(test_loader):
    #     print(f"Predicting Batch {batch_idx}")
    #     images = images.to(device)
    #     outputs = model(images)
    #     # Forward pass
    #     with torch.no_grad():
    #         outputs = model(images)
        
    #     # Apply softmax to get probabilities
    #     probabilities = F.softmax(outputs, dim=1)
 
    #     # Get predicted labels
    #     _, predicted = torch.max(probabilities, 1)
    #     labels = labels.numpy()
    #     predicted = predicted.cpu().numpy()
    #     plot_results(images, labels, predicted, save_dir, batch_idx)
    #     for i in range(len(predicted)):

    #         precision, recall, f1_score, iou = calculate_metrics(predicted[i], labels[i])
    #         f1_score_acc.append(f1_score)
    #         recall_acc.append(recall)
    #         precision_acc.append(precision)
    #         iou_acc.append(iou)
