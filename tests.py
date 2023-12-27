import unittest
import numpy as np
from cifar10_preprocessor import load_cifar10, preprocess_data, split_data
from tensorflow_cnn_classifier import build_cnn_model


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


if __name__ == '__main__':
    unittest.main()
