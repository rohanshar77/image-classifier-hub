import argparse
import numpy as np
import tensorflow as tf
import torch
from torchvision import transforms
import joblib
import cv2
import os
import sys

from cifar10_preprocessor import preprocess_data
from pytorch_cnn_classifier import CNN


def load_image(image_path):
    """Load and preprocess an image from the given path."""
    # Load the image using OpenCV
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # Check if the image was successfully loaded
    if image is None:
        raise ValueError(f"Unable to load image at {image_path}")

    # Resize the image if it's not in the expected dimensions (32x32 for CIFAR-10)
    if image.shape[:2] != (32, 32):
        image = cv2.resize(image, (32, 32))

    # Ensure the image has 3 channels (RGB)
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError(f"Image at {image_path} does not have 3 channels")

    # Preprocess the image (normalize pixel values)
    image = preprocess_data(np.array([image]))  # preprocess_data expects a batch of images

    return image


def load_tensorflow_model():
    """Load the trained TensorFlow model."""
    return tf.keras.models.load_model('models/tensorflow_model.h5')


def load_pytorch_model():
    """Load the trained PyTorch model."""
    model = CNN()
    model.load_state_dict(torch.load('models/pytorch_model.pth'))
    model.eval()  # Set the model to evaluation mode
    return model


def load_sklearn_model():
    """Load the trained Scikit-Learn model."""
    return joblib.load('models/sklearn_model.pkl')


def classify_image(model_framework, image):
    """Classify the image using the specified model framework."""
    if model_framework == 'tensorflow':
        model = load_tensorflow_model()
        predictions = model.predict(image)
        result = np.argmax(predictions, axis=1)

    elif model_framework == 'pytorch':
        model = load_pytorch_model()
        # Convert the normalized float32 image back to uint8
        image = (image.squeeze(0) * 255).astype(np.uint8)
        # Convert the image to a PyTorch tensor and normalize
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        image = transform(image)  # Apply transform
        image = image.unsqueeze(0)  # Add batch dimension back after transform
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            result = predicted.numpy()

    elif model_framework == 'sklearn':
        model = load_sklearn_model()
        image_flat = image.reshape(1, -1)
        result = model.predict(image_flat)

    else:
        raise ValueError("Invalid model framework specified.")

    return result[0]  # Assuming result contains class index


def get_model_choice():
    """Get the user's choice of models to use for classification."""
    print("Select the model(s) to use for classification:")
    print("1: TensorFlow")
    print("2: PyTorch")
    print("3: Scikit-Learn")
    print("Enter your choice(s) separated by space (e.g., '1 2' for TensorFlow and PyTorch):")

    while True:
        choices = input().strip().split()
        valid_choices = {'1', '2', '3'}
        if all(choice in valid_choices for choice in choices):
            return choices
        else:
            print("Invalid choice. Please enter 1, 2, or 3 separated by space.")


def get_image_path():
    """Prompt the user for an image path and validate it."""
    while True:
        print("Enter the path to the image file (or type 'exit' to quit):")
        image_path = input().strip()
        if image_path.lower() == 'exit':
            sys.exit("Exiting the program.")
        elif os.path.isfile(image_path):
            return image_path
        else:
            print("Invalid path or file does not exist. Please try again.")


def get_class_name(class_index):
    """Map the numerical class index to the actual class name."""
    class_names = ["Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]
    return class_names[class_index]


def main():
    print("Welcome to the Cross-Framework Image Classifier Hub!")

    model_choices = get_model_choice()

    while True:
        try:
            image_path = get_image_path()
            image = load_image(image_path)
            break
        except ValueError as e:
            print(e)
            print("Please enter a valid image file.")

    image_path = get_image_path()
    image = load_image(image_path)

    if '1' in model_choices:
        result = classify_image('tensorflow', image)
        class_name = get_class_name(result)
        print("TensorFlow Classification Result: ", class_name)

    if '2' in model_choices:
        result = classify_image('pytorch', image)
        class_name = get_class_name(result)
        print("PyTorch Classification Result: ", class_name)

    if '3' in model_choices:
        result = classify_image('sklearn', image)
        class_name = get_class_name(result)
        print("Scikit-Learn Classification Result: ", class_name)


if __name__ == "__main__":
    main()
