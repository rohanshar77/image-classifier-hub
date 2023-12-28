import unittest
import numpy as np
import torch
from cifar10_preprocessor import load_cifar10, preprocess_data, split_data, create_pytorch_dataloader
from tensorflow_cnn_classifier import build_cnn_model
from pytorch_cnn_classifier import CNN, load_dataset, train_model
from sklearn_baseline_classifier import train_and_evaluate_model, flatten_images


class TestCifar10Preprocessor(unittest.TestCase):
    def test_preprocess_data(self):
        """Ensures that the preprocess_data function correctly normalizes the image data"""
        (train_images, _), _ = load_cifar10()
        processed_images = preprocess_data(train_images)
        self.assertTrue((processed_images >= 0).all() and (processed_images <= 1).all())

    def test_split_data(self):
        """Checks whether the split_data function accurately splits the dataset into training and testing sets"""
        (train_images, train_labels), _ = load_cifar10()
        (train_images, _), (test_images, _) = split_data(train_images, train_labels, train_ratio=0.8)
        self.assertEqual(len(train_images), 40000)
        self.assertEqual(len(test_images), 10000)


class TestTensorflowCNNClassifier(unittest.TestCase):
    def test_model_construction(self):
        """Confirms that the CNN model is built with the correct number of layers"""
        model = build_cnn_model((32, 32, 3))
        self.assertEqual(len(model.layers), 8)

    def test_model_compilation(self):
        """Verifies that the model compiles without errors"""
        model = build_cnn_model((32, 32, 3))
        try:
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
            compiled = True
        except Exception:
            compiled = False
        self.assertTrue(compiled)

    def test_training_initialization(self):
        """Checks if the model starts training without issues"""
        (train_images, train_labels), _ = load_cifar10()
        train_images = preprocess_data(train_images)
        model = build_cnn_model((32, 32, 3))
        try:
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            model.fit(train_images, train_labels, epochs=1, batch_size=32, verbose=0)
            training_started = True
        except Exception as e:
            print(f"Training failed due to: {e}")
            training_started = False
        self.assertTrue(training_started)


class TestPytorchCNNClassifier(unittest.TestCase):
    def test_model_construction(self):
        """Confirms that the PyTorch CNN model is instantiated without errors"""
        try:
            model = CNN()
            constructed = True
        except Exception as e:
            print(f"Model construction failed due to: {e}")
            constructed = False
        self.assertTrue(constructed)

    def test_model_forward_pass(self):
        """Verifies that the model's forward pass operates correctly"""
        model = CNN()
        try:
            # Create a dummy input tensor
            dummy_input = torch.randn(1, 3, 32, 32)
            output = model(dummy_input)
            forward_pass_successful = True
        except Exception as e:
            print(f"Forward pass failed due to: {e}")
            forward_pass_successful = False
        self.assertTrue(forward_pass_successful)

    def test_training_initialization(self):
        """Checks if the PyTorch model starts training without issues"""
        model = CNN()
        train_loader, valid_loader, _ = load_dataset(batch_size=32)
        try:
            train_model(model, train_loader, valid_loader, learning_rate=0.001, num_epochs=1)
            training_started = True
        except Exception as e:
            print(f"Training failed due to: {e}")
            training_started = False
        self.assertTrue(training_started)


class TestSklearnBaselineClassifier(unittest.TestCase):
    def test_flatten_images(self):
        """Ensures that the flatten_images function correctly reshapes the image data"""
        (train_images, _), _ = load_cifar10()
        processed_images = preprocess_data(train_images)
        flattened_images = flatten_images(processed_images)
        self.assertEqual(flattened_images.shape[1], 32*32*3)

    def test_model_training_and_evaluation(self):
        """Checks if the Scikit-Learn model trains and evaluates without errors"""
        (train_images, train_labels), (test_images, test_labels) = load_cifar10()
        train_images = preprocess_data(train_images)
        test_images = preprocess_data(test_images)
        train_images_flat = flatten_images(train_images)
        test_images_flat = flatten_images(test_images)
        (train_images_flat, train_labels), (test_images_flat, test_labels) = split_data(train_images_flat, train_labels)

        try:
            model = train_and_evaluate_model(train_images_flat, train_labels, test_images_flat, test_labels)
            training_and_evaluation_successful = True
        except Exception as e:
            print(f"Training and evaluation failed due to: {e}")
            training_and_evaluation_successful = False
        self.assertTrue(training_and_evaluation_successful)


if __name__ == '__main__':
    unittest.main()
