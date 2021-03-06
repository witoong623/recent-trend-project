import os
import pickle
from albumentations.augmentations.transforms import Rotate
import torch
import numpy as np
import matplotlib.pyplot as plt
import albumentations as album
from datetime import datetime

from torch._C import Value


# helper function for data visualization
def visualize(**images):
    """
    Plot images in one row
    """
    n_images = len(images)
    plt.figure(figsize=(20,8))
    for idx, (name, image) in enumerate(images.items()):
        plt.subplot(1, n_images, idx + 1)
        plt.xticks([]); 
        plt.yticks([])
        # get title from the parameter names
        plt.title(name.replace('_',' ').title(), fontsize=20)
        plt.imshow(image)
    plt.savefig('sample_gt_pred_2_max.jpeg')
    plt.show()

# Perform one hot encoding on label
def one_hot_encode(label, label_values):
    """
    Convert a segmentation image label array to one-hot format
    by replacing each pixel value with a vector of length num_classes
    # Arguments
        label: The 2D array segmentation image label
        label_values
        
    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of num_classes
    """
    semantic_map = []
    for colour in label_values:
        equality = np.equal(label, colour)
        class_map = np.all(equality, axis = -1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)

    return semantic_map
    
# Perform reverse one-hot-encoding on labels / preds
def reverse_one_hot(image):
    """
    Transform a 2D array in one-hot format (depth is num_classes),
    to a 2D array with only 1 channel, where each pixel value is
    the classified class key.
    # Arguments
        image: The one-hot format image 
        
    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of 1, where each pixel value is the classified 
        class key.
    """
    x = np.argmax(image, axis = -1)
    return x

# Perform colour coding on the reverse-one-hot outputs
def colour_code_segmentation(image, label_values):
    """
    Given a 1-channel array of class keys, colour code the segmentation results.
    # Arguments
        image: single channel array where each value represents the class key.
        label_values

    # Returns
        Colour coded image for segmentation visualization
    """
    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]

    return x


def get_training_augmentation(type='basic'):
    if type == 'basic':
        train_transform = [
            album.RandomCrop(height=1024, width=1024, always_apply=True),
            album.HorizontalFlip(p=0.5),
            album.VerticalFlip(p=0.5),
        ]
    elif type == 'advance':
        train_transform = [
            album.RandomCrop(height=1024, width=1024, always_apply=True),
            album.HorizontalFlip(p=0.5),
            album.VerticalFlip(p=0.5),
            album.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25),
            album.HueSaturationValue(p=0.5),
            album.Blur(blur_limit=7, p=0.5),
            album.ToGray(p=0.5),
            album.IAASharpen(p=0.5),
        ]
    elif type == 'intermediate':
        train_transform = [
            album.RandomCrop(height=1024, width=1024, always_apply=True),
            album.HorizontalFlip(p=0.5),
            album.VerticalFlip(p=0.5),
            album.Rotate(limit=90, p=0.5),
            album.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25),
            album.HueSaturationValue(p=0.5),
        ]
    else:
        raise ValueError(f'type {type} is not supported')

    return album.Compose(train_transform)


def get_validation_augmentation():
    train_transform = [
        album.CenterCrop(height=1024, width=1024, always_apply=True),
    ]
    return album.Compose(train_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


dataset_mean = np.array([147.2528, 160.7285,  75.4926]) / 255
dataset_std = np.array([0.3661, 0.3492, 0.3359])

def no_pretrain_precessing(x, **kwargs):
    if x.max() > 1:
        x = x / 255
    else:
        raise ValueError('image has invalid data')

    x = x - dataset_mean
    x = x / dataset_std

    return x


def get_preprocessing(preprocessing_fn=None):
    """Construct preprocessing transform    
    Args:
        preprocessing_fn (callable): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    """
    _transform = []
    if preprocessing_fn:
        _transform.append(album.Lambda(name='Model preprocessing', image=preprocessing_fn))
    _transform.append(album.Lambda(name='To tensor', image=to_tensor, mask=to_tensor))
        
    return album.Compose(_transform)


def save_history(data, name, dir):
    ''' save history in pickle, name is full path to the file '''

    if not os.path.exists(dir):
        os.mkdir(dir)

    name = f'{name}.pickle'
    pickle.dump(data, open(os.path.join(dir, name), 'wb+'))


def save_model(model, name, dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

    name = f'{name}.pth'
    torch.save(model.state_dict(), os.path.join(dir, name))
