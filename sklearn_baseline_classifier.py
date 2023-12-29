from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from cifar10_preprocessor import load_cifar10, preprocess_data, split_data
import joblib


def flatten_images(images):
    """Flatten the image array for use in Scikit-Learn models."""
    return images.reshape(images.shape[0], -1)


def train_and_evaluate_model(train_images, train_labels, test_images, test_labels):
    """Train an SVM model and evaluate its performance."""
    print("Training the Scikit-Learn model...")

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

    return model


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
    sklearn_model = train_and_evaluate_model(train_images_flat, train_labels, test_images_flat, test_labels)


if __name__ == "__main__":
    main()