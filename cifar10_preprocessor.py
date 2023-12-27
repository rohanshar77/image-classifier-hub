import numpy as np
import cv2
from tensorflow.keras.datasets import cifar10
import torch
from torch.utils.data import DataLoader, TensorDataset


def load_cifar10():
    """Load the CIFAR-10 dataset using TensorFlow and return it."""
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    return (train_images, train_labels), (test_images, test_labels)


# This is more of a placeholder since CIFAR-10 images are already 32x32
def preprocess_data(images):
    """Preprocess images: normalize and resize."""
    # Normalize pixel values
    images = images.astype('float32') / 255.0
    # Resize images using OpenCV
    resized_images = np.array([cv2.resize(img, (32, 32)) for img in images])
    return resized_images


def split_data(images, labels, train_ratio=0.8):
    """Split the dataset into training and testing sets."""
    num_train = int(len(images) * train_ratio)
    return (images[:num_train], labels[:num_train]), (images[num_train:], labels[num_train:])


def create_pytorch_dataloader(images, labels, batch_size=32):
    """Create a DataLoader for PyTorch."""
    tensor_x = torch.Tensor(images) # transform to torch tensor
    tensor_y = torch.Tensor(labels).long().squeeze()
    dataset = TensorDataset(tensor_x, tensor_y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


# Load and preprocess the CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = load_cifar10()
train_images = preprocess_data(train_images)
test_images = preprocess_data(test_images)

# Split the dataset
(train_images, train_labels), (test_images, test_labels) = split_data(train_images, train_labels)

# Create DataLoaders for PyTorch
train_loader = create_pytorch_dataloader(train_images, train_labels)
test_loader = create_pytorch_dataloader(test_images, test_labels)

# The data is now ready to be used in TensorFlow, PyTorch, and Scikit-Learn models.