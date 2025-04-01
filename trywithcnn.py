import os #For file path operations
import numpy as np #For data manipulation
import pandas as pd #For data manipulation
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Load AHCD dataset from CSV files
def load_ahcd_csv(image_csv_path, label_csv_path, image_size=(32, 32)):
    # Load images
    images_df = pd.read_csv(image_csv_path, header=None)
    images = images_df.values.reshape(-1, image_size[0], image_size[1], 1)  # Reshape to (num_images, 32, 32, 1)
    images = images / 255.0  # Normalize pixel values to [0, 1]

    # Load labels
    labels_df = pd.read_csv(label_csv_path, header=None)
    labels = labels_df.values.flatten()  # Flatten to 1D array

    return images, labels

# Paths to AHCD CSV files
base_dir = r'C:\Users\demiana\Desktop\BachelorYear\ahcd'  # Update this path
train_images_csv = os.path.join(base_dir, 'csvTrainImages 13440x1024.csv')
train_labels_csv = os.path.join(base_dir, 'csvTrainLabel 13440x1.csv')
test_images_csv = os.path.join(base_dir, 'csvTestImages 3360x1024.csv')
test_labels_csv = os.path.join(base_dir, 'csvTestLabel 3360x1.csv')

# Load training data
X_train, y_train = load_ahcd_csv(train_images_csv, train_labels_csv)
print("Training data shape:", X_train.shape)
print("Training labels shape:", y_train.shape)

# Load testing data
X_test, y_test = load_ahcd_csv(test_images_csv, test_labels_csv)
print("Testing data shape:", X_test.shape)
print("Testing labels shape:", y_test.shape)

# Encode labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Convert labels to one-hot encoded vectors
num_classes = len(label_encoder.classes_)
y_train_onehot = to_categorical(y_train_encoded, num_classes)
y_test_onehot = to_categorical(y_test_encoded, num_classes)

# Build the CNN model
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train_onehot, epochs=10, batch_size=32, validation_data=(X_test, y_test_onehot))

# Plot training history
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test_onehot)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Save the model
model.save('arabic_handwriting_model.h5')

# Load the model
model = load_model('arabic_handwriting_model.h5')

# Predict on a new image
def predict_image(image_path):
    from PIL import Image
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = img.resize((32, 32))  # Resize to 32x32
    img = np.array(img) / 255.0  # Normalize
    img = img.reshape(1, 32, 32, 1)  # Reshape for model input
    prediction = model.predict(img)
    predicted_class = label_encoder.inverse_transform([np.argmax(prediction)])
    return predicted_class[0]

# Example usage
print(predict_image('path/to/new_image.png'))
2