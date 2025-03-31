import numpy as np
import random
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import cv2
import os
import gc

path = "/content/drive/MyDrive/finalMp3_dataset/Dataset"
labelFile = "/content/drive/MyDrive/finalMp3_dataset/Dataset/labels.csv"

# Constants
imageDimensions = (32, 32, 1)  # Grayscale images
testRatio = 0.2
validationRatio = 0.2
batch_size = 32
epochs = 15
noOfClasses = 43  # Classes are from 0 to 42

# Load dataset
print("Loading Dataset...")
images, classNo = [], []
for class_id in range(noOfClasses):  # Iterate from 0 to 42
    class_path = os.path.join(path, str(class_id))
    if not os.path.exists(class_path):
        print(f"Warning: Class {class_id} directory not found!")
        continue
    print(f"Processing class {class_id}...")
    for img_name in os.listdir(class_path):
        img = cv2.imread(os.path.join(class_path, img_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.equalizeHist(img) / 255.0
        images.append(img)
        classNo.append(class_id)
    print(f"Class {class_id} loaded with {len(os.listdir(class_path))} images.")

    # Convert to NumPy arrays
print("Converting dataset to NumPy arrays...")
images = np.array(images).reshape(-1, imageDimensions[0], imageDimensions[1], 1)
classNo = np.array(classNo)

# Split dataset
print("Splitting dataset into train, validation, and test sets...")
X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=testRatio)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validationRatio)
print("Dataset split completed.")

# Convert labels to categorical
y_train = to_categorical(y_train, noOfClasses)
y_val = to_categorical(y_val, noOfClasses)
y_test = to_categorical(y_test, noOfClasses)

# Data Augmentation
print("Setting up data augmentation...")
dataGen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    shear_range=0.1,
    rotation_range=10
)
dataGen.fit(X_train)
print("Data augmentation setup completed.")


# Best Model Parameters
best_params = {
    "conv1_filters": 32,
    "conv2_filters": 128,
    "kernel_size": 5,
    "dense_units": 1024,
    "dropout_rate": 0.3,
    "learning_rate": 0.0005
}

# Define CNN Model
def create_best_model(params):
    model = Sequential([
        Conv2D(params["conv1_filters"], (params["kernel_size"], params["kernel_size"]), activation='relu',
               input_shape=(imageDimensions[0], imageDimensions[1], 1)),
        Conv2D(params["conv2_filters"], (params["kernel_size"], params["kernel_size"]), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(params["dropout_rate"]),

        Flatten(),
        Dense(params["dense_units"], activation='relu'),
        Dropout(params["dropout_rate"]),
        Dense(noOfClasses, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=params["learning_rate"]),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Train the best model
best_model = create_best_model(best_params)
history = best_model.fit(
    dataGen.flow(X_train, y_train, batch_size=batch_size),
    validation_data=(X_val, y_val),
    epochs=15,
    callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
               ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)],
    verbose=1
)

# Evaluate on test set
score = best_model.evaluate(X_test, y_test, verbose=1)
print(f'\nFinal Test Score: {score[0]}')
print(f'Final Test Accuracy: {score[1]}')

# Save the best model
best_model.save("/content/drive/MyDrive/best_model.h5")
print("\nBest Model saved successfully!")
