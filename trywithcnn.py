import os #For file path operations
import numpy as np #For data manipulation
import pandas as pd #For data manipulation
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report  #For model evaluation
import seaborn as sns #For data visualization
import tensorflow as tf #For deep learning
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Load AHCD dataset from CSV files
def load_ahcd_csv(image_csv_path, label_csv_path, image_size=(32, 32)):
   ## This function loads Arabic handwritten character images and their labels from CSV files
##image_csv_path: Path to file containing image data (each row is a flattened 32x32 image)
##label_csv_path: Path to file containing corresponding labels (what character each image represents)
##What happens inside:
##Images are loaded and reshaped from 1024 pixels (32x32) to 32x32x1 (width x height x channels)
##Pixel values are normalized from 0-255 to 0-1 (helps the model learn better
##Labels are loaded as simple numbers/letters
    # Load images
    images_df = pd.read_csv(image_csv_path, header=None)
    images = images_df.values.reshape(-1, image_size[0], image_size[1], 1)  # Reshape to (num_images, 32, 32, 1)
    images = images / 255.0  # Normalize pixel values to [0, 1]

    # Load labels
    labels_df = pd.read_csv(label_csv_path, header=None)
    labels = labels_df.values.flatten()  # Flatten to 1D array

    return images, labels

# Paths to AHCD CSV files
base_dir = r'C:\Users\demiana\Desktop\BachelorYear\ahcd'  
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
history = model.fit(X_train, y_train_onehot, epochs=17, batch_size=32, validation_data=(X_test, y_test_onehot))

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
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")

def comprehensive_evaluation(model, X_test, y_test_onehot, label_encoder):
    # Get predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test_onehot, axis=1)
    
    # 1. Confusion Matrix Visualization
    plt.figure(figsize=(15, 12))
    cm = confusion_matrix(y_true, y_pred_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    
    # 2. Classification Report
    print("\nDetailed Classification Report:")
    target_names = [str(cls) for cls in label_encoder.classes_]
    print(classification_report(y_true, y_pred_classes, target_names=target_names))

    
    # 3. Advanced Metrics Table
    metrics = {
        'Character': [],
        'Error Rate': [],
        'Precision': [],
        'Recall': [],
        'F1-Score': []
    }
    
    for i, class_name in enumerate(label_encoder.classes_):
        TP = np.sum((y_true == i) & (y_pred_classes == i))
        FP = np.sum((y_true != i) & (y_pred_classes == i))
        FN = np.sum((y_true == i) & (y_pred_classes != i))
        TN = np.sum((y_true != i) & (y_pred_classes != i))
        
        metrics['Character'].append(class_name)
        metrics['Error Rate'].append((FP + FN) / (TP + TN + FP + FN))
        metrics['Precision'].append(TP / (TP + FP) if (TP + FP) > 0 else 0)
        metrics['Recall'].append(TP / (TP + FN) if (TP + FN) > 0 else 0)
        metrics['F1-Score'].append(2 * (metrics['Precision'][-1] * metrics['Recall'][-1]) / 
                                  (metrics['Precision'][-1] + metrics['Recall'][-1]) 
                                  if (metrics['Precision'][-1] + metrics['Recall'][-1]) > 0 else 0)
    
    # Print metrics table
    print("\nPer-Class Performance Metrics:")
    metrics_df = pd.DataFrame(metrics)
    print(metrics_df.to_string(index=False))
        # Print overall macro-average metrics
    from sklearn.metrics import precision_score, recall_score, f1_score

    print("\nOverall Metrics:")
    print(f"Macro Precision: {precision_score(y_true, y_pred_classes, average='macro'):.3f}")
    print(f"Macro Recall:    {recall_score(y_true, y_pred_classes, average='macro'):.3f}")
    print(f"Macro F1-Score:  {f1_score(y_true, y_pred_classes, average='macro'):.3f}")


# Run comprehensive evaluation
comprehensive_evaluation(model, X_test, y_test_onehot, label_encoder)




# Save the model
model.save('arabic_handwriting_model.h5')

# Load the model
model = load_model('arabic_handwriting_model.h5')

# Predict on a new image
#def predict_image(image_path):
#    from PIL import Image
 #  img = img.resize((32, 32))  # Resize to 32x32
  #  img = np.array(img) / 255.0  # Normalize
   # img = img.reshape(1, 32, 32, 1)  # Reshape for model input
    #prediction = model.predict(img)
    #predicted_class = label_encoder.inverse_transform([np.argmax(prediction)])
    #return predicted_class[0]


def predict_image(image_path):
    try:
        from PIL import Image
        img = Image.open(image_path).convert('L')  # Convert to grayscale
        img = img.resize((32, 32))  # Resize to 32x32
        img = np.array(img) / 255.0  # Normalize
        img = img.reshape(1, 32, 32, 1)  # Reshape for model input
        
        prediction = model.predict(img, verbose=0)
        predicted_class = label_encoder.inverse_transform([np.argmax(prediction)])
        confidence = np.max(prediction)
        
        print(f"\nPredicted Character: {predicted_class[0]}")
        print(f"Confidence: {confidence:.2%}")
        
        # Show top 3 predictions
        top3 = np.argsort(prediction[0])[-3:][::-1]
        print("\nTop 3 Predictions:")
        for i, idx in enumerate(top3):
            print(f"{i+1}. {label_encoder.classes_[idx]} ({prediction[0][idx]:.2%})")
            
        return predicted_class[0]
    except Exception as e:
        print(f"\nError processing image: {e}")
        return None
# Example usage
print(predict_image('C:/Users/demiana/Desktop/BachelorYear/sample_char_image.png'))

