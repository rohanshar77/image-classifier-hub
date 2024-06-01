# Cross-Framework Image Classifier Hub

## Introduction
The Cross-Framework Image Classifier Hub is a Python-based application designed for the task of image classification. It showcases the use of multiple machine learning frameworks to build, train, and evaluate models for classifying images into predefined categories. This project is tailored for users interested in comparing the performance and characteristics of different machine learning models and frameworks in a unified platform.

## Project Components

### cifar10_preprocessor.py
- **Purpose**: Handles the loading, preprocessing, and splitting of the CIFAR-10 dataset.
- **Functionality**:
  - Loads the CIFAR-10 dataset using TensorFlow.
  - Preprocesses images by normalizing and resizing them to the format suitable for model input.
  - Splits the dataset into training and testing sets.
  - Provides a DataLoader for PyTorch.

### tensorflow_cnn_classifier.py
- **Purpose**: Constructs, compiles, trains, and evaluates a CNN model using TensorFlow and Keras.
- **Functionality**:
  - Builds a convolutional neural network (CNN) suited for CIFAR-10 images.
  - Compiles and trains the model, then evaluates it using metrics like accuracy, precision, and recall.
  - Saves the trained model and evaluation metrics.

### pytorch_cnn_classifier.py
- **Purpose**: Similar to the TensorFlow script but utilizes PyTorch for the CNN.
- **Functionality**:
  - Defines the CNN architecture using PyTorch.
  - Handles the training and evaluation of the model.
  - Saves the trained model and its performance metrics.

### sklearn_baseline_classifier.py
- **Purpose**: Provides a baseline comparison by training a simpler machine learning model (SVM with PCA) using Scikit-Learn.
- **Functionality**:
  - Prepares the dataset for Scikit-Learn models.
  - Trains the SVM model and evaluates its performance.
  - Saves the model and its evaluation metrics.

### classifier_cli.py
- **Purpose**: Offers a Command-Line Interface for user interaction.
- **Functionality**:
  - Allows users to load images and select a model/framework for classification.
  - Supports TensorFlow, PyTorch, and Scikit-Learn models for image classification.
  - Provides a user-friendly interface for model selection, image classification, and viewing results.

## Installation and Usage

To use the Cross-Framework Image Classifier Hub, follow these steps:

1. **Clone the Repository**:
    ```
    git clone https://github.com/rohanshar77/image-classifier-hub.git
    cd image-classifier-hub
    ```

2. **Set Up the Conda Environment**:
- Create and activate a new Conda environment:
  ```
  conda create --name image_classifier_hub python=3.8
  conda activate image_classifier_hub
  ```
- Install the required packages:
  ```
  conda install tensorflow
  conda install keras
  conda install pytorch torchvision -c pytorch
  conda install scikit-learn
  conda install pillow
  conda install numpy
  conda install pandas
  conda install opencv
  ```

3. **Run the Scripts**:
- To train and evaluate the models, run the respective scripts:
  ```
  python tensorflow_cnn_classifier.py
  python pytorch_cnn_classifier.py
  python sklearn_baseline_classifier.py
  ```
- To classify images using the CLI:
  ```
  python classifier_cli.py
  ```

4. **Using the CLI**:
- Follow the on-screen prompts to select a model and classify your images.

## Project Findings and Insights

In this project, we trained and evaluated three different models using the CIFAR-10 dataset. The models included a Convolutional Neural Network (CNN) implemented in both TensorFlow and PyTorch, and a Support Vector Machine (SVM) model with Principal Component Analysis (PCA) implemented in Scikit-Learn. Here are the evaluation metrics for each model:

### Model Performance Metrics

- **PyTorch CNN Model Metrics**:
  - Accuracy: 71.40%
  - Precision: 71.45%
  - Recall: 71.40%

- **TensorFlow CNN Model Metrics**:
  - Accuracy: 68.19%
  - Precision: 69.58%
  - Recall: 68.21%

- **Scikit-Learn SVM Model Metrics**:
  - Accuracy: 55.34%
  - Precision: 55.32%
  - Recall: 55.29%

### Analysis

The metrics reveal some interesting insights about the performance of different machine learning models on the CIFAR-10 dataset:

1. **CNN Models Outperform SVM**: The CNN models (both PyTorch and TensorFlow) significantly outperformed the SVM model. This is expected as CNNs are more suited for image classification tasks due to their ability to capture spatial hierarchies in image data.

2. **PyTorch vs. TensorFlow**: The PyTorch implementation of the CNN showed slightly higher accuracy, precision, and recall compared to TensorFlow. This difference could be attributed to various factors such as differences in default hyperparameters, weight initialization, and optimization algorithms between the two frameworks.

3. **SVM Performance**: The lower performance of the SVM model can be attributed to its linear nature and the high-dimensional nature of image data, even though PCA was used for dimensionality reduction. SVMs generally struggle with the complexity and variability present in image datasets like CIFAR-10.

4. **Dataset Complexity**: CIFAR-10 is a relatively complex dataset with high intra-class variation, making it a challenging task for models, especially simpler ones like SVM.

These results demonstrate the effectiveness of deep learning models, particularly CNNs, in image classification tasks compared to traditional machine learning approaches like SVMs.

## Conclusion

This project provides a comparative study of different machine learning models and frameworks, showcasing their strengths and weaknesses in the context of image classification. It serves as a valuable educational tool for understanding and exploring machine learning techniques.

## Technologies Used
- Python
- TensorFlow and Keras
- PyTorch
- Scikit-Learn
- OpenCV
- NumPy
- Conda

## Contributions

Contributions to this project are welcome. Feel free to fork the repository and submit pull requests.

## License

This project is open-sourced under the MIT License. See the LICENSE file for more details.
