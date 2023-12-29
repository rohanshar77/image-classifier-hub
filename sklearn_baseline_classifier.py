from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib
import json

from cifar10_preprocessor import load_cifar10, preprocess_data, split_data


def flatten_images(images):
    """Flatten the image array for use in Scikit-Learn models."""
    return images.reshape(images.shape[0], -1)


def train_and_evaluate_model(train_images, train_labels, test_images, test_labels):
    """Train an SVM model and evaluate its performance."""
    print("Training the Scikit-Learn model...")

    # Flatten the images for the Scikit-Learn model
    test_images_flat = flatten_images(test_images)

    # Reducing dimensionality for SVM using PCA, as it struggles with high-dimensional data
    pca = PCA(n_components=150, whiten=True, random_state=42)
    svc = SVC(kernel='rbf', class_weight='balanced')

    # Create a pipeline that first reduces dimensionality then trains an SVM
    model = make_pipeline(pca, svc)

    # Training the model
    model.fit(train_images, train_labels.ravel())

    # Predicting and evaluating on the test set
    predictions = model.predict(test_images)
    accuracy = accuracy_score(test_labels, predictions)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # Save the model
    joblib.dump(model, 'models/sklearn_model.pkl')

    # Calculate metrics
    predictions = model.predict(test_images_flat)
    accuracy = accuracy_score(test_labels, predictions)
    precision = precision_score(test_labels, predictions, average='macro')
    recall = recall_score(test_labels, predictions, average='macro')

    # Save metrics
    metrics = {'accuracy': accuracy, 'precision': precision, 'recall': recall}
    save_metrics(metrics, 'metrics/sklearn_metrics.json')


def save_metrics(metrics, filename):
    with open(filename, 'w') as f:
        json.dump(metrics, f)


def main():
    # Load and preprocess the CIFAR-10 dataset
    (train_images, train_labels), (test_images, test_labels) = load_cifar10()
    train_images = preprocess_data(train_images)
    test_images = preprocess_data(test_images)

    # Flatten the images for the Scikit-Learn model
    train_images_flat = flatten_images(train_images)
    test_images_flat = flatten_images(test_images)

    # Split the dataset
    (train_images_flat, train_labels), (test_images_flat, test_labels) = split_data(train_images_flat, train_labels)

    # Train and evaluate the model
    train_and_evaluate_model(train_images_flat, train_labels, test_images_flat, test_labels)


if __name__ == "__main__":
    main()