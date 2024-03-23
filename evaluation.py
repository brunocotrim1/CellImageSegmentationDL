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
# Define custom colormap
custom_colors = [(0, 0, 0),  # Background (black)
                 (0, 0, 1),  # Inside of cells (blue)
                 (1, 0, 0)]  # Boundary of cells (red)
custom_cmap = ListedColormap(custom_colors)
np.set_printoptions(threshold=np.inf)
# Create a rainbow colormap


def dice_coefficient(predicted, labels):
    # Calculate the intersection and union
    intersection = np.sum(predicted * labels)
    total = np.sum(predicted) + np.sum(labels)
    
    # Calculate the Dice coefficient
    dice = (2.0 * intersection) / (total + 1e-8)  # Add a small epsilon to avoid division by zero
    return dice
def calculate_metrics(predicted, labels):
    # Flatten the predicted and true labels
    predicted_flat = predicted.flatten()
    labels_flat = labels.flatten()

    # Calculate precision, recall, F1-score, and support
    precision, recall, f1_score, _ = precision_recall_fscore_support(labels_flat, predicted_flat, average='weighted')

    # Calculate intersection over union (IoU)
    iou = jaccard_score(labels_flat, predicted_flat, average='weighted')
    dice = dice_coefficient(predicted_flat, labels_flat)

    return precision, recall, f1_score, iou, dice

def predict(model, truth):
    model.eval()
    with torch.no_grad():
        predicted = model(truth)
    probabilities = F.softmax(predicted, dim=1)
    _, predicted = torch.max(probabilities, 1)
    return predicted

def evaluate(model, loader, device,simple=True,plot=False,save_dir="visualization"):
    model.eval()
    f1_score_acc = []
    recall_acc  = []
    precision_acc  = []
    iou_acc  = []
    pixel_acc_acc  = []

    for batch_idx,(images, labels) in enumerate(loader):
        #print(f"Predicting Batch ")
        images = images.to(device)  # Move images to GPU
        predicted = predict(model, images).cpu()  # Model prediction and conversion to NumPy on GPU
        if plot:
            plot_results(images, labels, predicted, save_dir, batch_idx)
        for i in range(len(predicted)):
            if simple: 
                iou = jaccard_score(labels[i].flatten(), predicted[i].flatten(), average='weighted')
                iou_acc.append(iou)
            else:
                precision, recall, f1_score, iou = calculate_metrics(predicted[i], labels[i])
                f1_score_acc.append(f1_score)
                recall_acc.append(recall)
                precision_acc.append(precision)
                iou_acc.append(iou)
    if simple:
        return {'iou': np.mean(iou_acc)}
    else:
        return {'iou': np.mean(iou_acc), 'pixel_accuracy': np.mean(pixel_acc_acc), 
                'precision': np.mean(precision_acc), 'recall': np.mean(recall_acc), 
                'f1_score': np.mean(f1_score_acc)}

def plot_results(images, labels, predicted, save_dir, batch_idx):
        for i in range(len(predicted)):
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Plot original image
            axes[0].imshow(np.transpose(images[i].cpu().numpy(), (1, 2, 0)))
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            # Plot ground truth mask
            axes[1].imshow(labels[i], cmap='gray')
            axes[1].set_title('Ground Truth Mask')
            axes[1].axis('off')
            
            # Plot predicted mask with custom colormap
            axes[2].imshow(np.transpose(images[i].cpu().numpy(), (1, 2, 0)))
            axes[2].imshow(predicted[i], cmap=custom_cmap, alpha=0.7)
            axes[2].set_title('Predicted Mask on Original Image')
            axes[2].axis('off')
            
            save_path = os.path.join(save_dir, f'batch_{batch_idx}_image_{i}_prediction.png')
            plt.savefig(save_path)
            plt.close()
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

if __name__ == '__main__':
    main()