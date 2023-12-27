import tensorflow as tf
from cifar10_preprocessor import load_cifar10, preprocess_data, split_data


def build_cnn_model(input_shape):
    """Builds a CNN model using TensorFlow and Keras."""
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model


def compile_and_train_model(model, train_images, train_labels, test_images, test_labels):
    """Compiles and trains the CNN model."""
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    print(train_labels.shape, train_labels.dtype)
    print(test_labels.shape, test_labels.dtype)
    model.fit(train_images, train_labels, epochs=10,
              validation_data=(test_images, test_labels))


# Load and preprocess the CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = load_cifar10()
train_images = preprocess_data(train_images)
test_images = preprocess_data(test_images)

# Split the dataset
(train_images, train_labels), (test_images, test_labels) = split_data(train_images, train_labels)

# Build the model
input_shape = train_images.shape[1:]  # Shape of CIFAR-10 images
cnn_model = build_cnn_model(input_shape)

# Compile and train the model
compile_and_train_model(cnn_model, train_images, train_labels, test_images, test_labels)
