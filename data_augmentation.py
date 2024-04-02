import numpy as np
import random
import cv2
def scale_intensity(img, min_val=0, max_val=255):
    # Scale image intensity between min_val and max_val
    img = np.clip(img, np.min(img), np.max(img))
    img = (img - np.min(img)) / (np.max(img) - np.min(img)) * (max_val - min_val) + min_val
    return img

def spatial_pad(image, mask, expected_shape):
    """
    Pad image and mask if width or height do not match the expected dimensions.

    Parameters:
    image (numpy.ndarray): Input image with shape (channels, height, width).
    mask (numpy.ndarray): Input mask with shape (channels, height, width).
    expected_shape (tuple): Tuple containing (channels, height, width) of the expected shape.

    Returns:
    tuple: Padded image and mask.
    """
    channels, height, width = expected_shape
    
    if image.shape[1:] != (height, width):
        pad_height = max(0, height - image.shape[1])
        pad_width = max(0, width - image.shape[2])
        padding = ((0, 0), (0, pad_height), (0, pad_width))
        image = np.pad(image, padding, mode='constant')
    
    if mask.shape[1:] != (height, width):
        pad_height = max(0, height - mask.shape[1])
        pad_width = max(0, width - mask.shape[2])
        padding = ((0, 0), (0, pad_height), (0, pad_width))
        mask = np.pad(mask, padding, mode='constant')
    
    return image, mask


def random_spatial_crop(img, mask, roi_size):
    # Randomly crop the image and mask
    h, w = roi_size
    y = random.randint(0, img.shape[1] - h)
    x = random.randint(0, img.shape[2] - w)
    img = img[:, y:y+h, x:x+w]
    mask = mask[:, y:y+h, x:x+w]
    return img, mask


def random_axis_flip(img, mask):
    if random.random() > 0.5:
        return img, mask
    # Randomly flip the image and mask along spatial axes
    if random.random() > 0.5:
        img = np.flip(img, axis=1)
        mask = np.flip(mask, axis=1)
    if random.random() > 0.5:
        img = np.flip(img, axis=2)
        mask = np.flip(mask, axis=2)
    return img, mask

def random_rotate_90(img, mask):
    if random.random() > 0.5:
        return img,mask
    # Randomly rotate the image and mask by 90 degrees
    k = random.randint(0, 3)
    img = np.rot90(img, k, axes=(1, 2))
    mask = np.rot90(mask, k, axes=(1, 2))
    return img, mask

def random_gaussian_noise(img, mean=0, std=0.1):
    if random.random() > 0.25:
        return img
    # Add random Gaussian noise to the image
    noise = np.random.normal(mean, std, img.shape)
    img = img + noise
    return img
def random_adjust_contrast(image, gamma_range=(1, 2), prob=0.25):
    if random.random() < prob:
        # Check for invalid values in the image
        if np.any(np.isnan(image)) or np.any(np.isinf(image)):
            return image  # If invalid values are present, return the original image
        
        # Adjust contrast of the image using gamma correction
        gamma = random.uniform(*gamma_range)
        image = np.clip((image - np.min(image)) / (np.max(image) - np.min(image)), 0, 1)  # Normalize image to [0, 1]
        image = np.clip(image ** gamma, 0, 1)  # Apply gamma correction

    return image
def random_gaussian_smooth(img, sigma_range=(1, 2)):
    # Apply random Gaussian smoothing to the image
    if random.random() > 0.25:
        return img
    sigma = random.uniform(*sigma_range)
    img = cv2.GaussianBlur(img.transpose(1, 2, 0), (0, 0), sigma)
    return img.transpose(2, 0, 1)

def random_histogram_shift(img, num_control_points=3):
    if random.random() > 0.25:
        return img
    # Apply random histogram shift to the image
    flat_img = img.reshape(-1)
    quantiles = np.linspace(0, 1, num_control_points)
    control_points = np.quantile(flat_img, quantiles)
    interp_points = np.linspace(0, 1, num_control_points)
    img = np.interp(img, control_points, interp_points).reshape(img.shape)
    return img

def random_zoom(img, mask, input_size=(256, 256), min_zoom=0.8, max_zoom=1.0):
    if random.random() > 0.15:
        return img, mask
    # Calculate the maximum zoom factor that maintains the input size
    max_zoom_factor = min(input_size[0] / img.shape[1], input_size[1] / img.shape[2], max_zoom)

    # Randomly select zoom factor within the allowed range
    zoom_factor = random.uniform(min_zoom, max_zoom_factor)

    # Determine the new size after zooming
    new_size = (int(img.shape[1] * zoom_factor), int(img.shape[2] * zoom_factor))

    # Apply zoom to image
    img_zoomed = cv2.resize(img.transpose(1, 2, 0), new_size[::-1], interpolation=cv2.INTER_NEAREST).transpose(2, 0, 1)

    # Apply zoom to mask
    mask_zoomed = cv2.resize(mask.transpose(1, 2, 0), new_size[::-1], interpolation=cv2.INTER_NEAREST).transpose(2, 0, 1)
    # Pad to maintain input size
    pad_width = [(0, max(0, input_size[i] - img_zoomed.shape[i + 1])) for i in range(len(input_size))]
    pad_width = [(0, 0)] + pad_width  # Pad along channel dimension as well
    img_zoomed = np.pad(img_zoomed, pad_width, mode='constant')
    mask_zoomed = np.pad(mask_zoomed, pad_width, mode='constant')
    return img_zoomed, mask_zoomed