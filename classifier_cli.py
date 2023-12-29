import argparse
import numpy as np
import cv2
import tensorflow as tf
import torch
from torchvision import transforms
import joblib
import cv2

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

    # Preprocess the image (normalize pixel values)
    image = preprocess_data(image)

    # Expand dimensions to match the model's input shape (batch size, height, width, channels)
    image = np.expand_dims(image, axis=0)

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
        # TensorFlow expects the image in a certain format. The preprocessing should already handle this.
        predictions = model.predict(image)
        result = np.argmax(predictions, axis=1)

    elif model_framework == 'pytorch':
        model = load_pytorch_model()
        # Convert the image to a PyTorch tensor and normalize
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        image = transform(image).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            result = predicted.numpy()

    elif model_framework == 'sklearn':
        model = load_sklearn_model()
        # Flatten the image for Scikit-Learn and predict
        image_flat = image.reshape(1, -1)
        result = model.predict(image_flat)

    else:
        raise ValueError("Invalid model framework specified.")

    # Return the classification result
    return result[0]  # Assuming result contains class index


def main():
    parser = argparse.ArgumentParser(description='Image Classifier CLI')
    parser.add_argument('image_path', type=str, help='Path to the image file')
    parser.add_argument('--model', choices=['tensorflow', 'pytorch', 'sklearn'],
                        default='tensorflow', help='Model framework to use for classification')

    args = parser.parse_args()

    # Load and preprocess the image
    image = load_image(args.image_path)

    # Classify the image
    result = classify_image(args.model, image)

    # Display the classification result
    print(f"Classification Result: {result}")


if __name__ == "__main__":
    main()
