import os
import numpy as np
import cv2
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from config import *

def load_data():
    """Load and preprocess the dataset"""
    data, labels = [], []
    
    # Load with_mask images
    for img_name in os.listdir(WITH_MASK_DIR):
        img_path = os.path.join(WITH_MASK_DIR, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (224, 224))  # MobileNetV2 input size
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            data.append(img)
            labels.append(1)
    
    # Load without_mask images
    for img_name in os.listdir(WITHOUT_MASK_DIR):
        img_path = os.path.join(WITHOUT_MASK_DIR, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (224, 224))  # MobileNetV2 input size
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            data.append(img)
            labels.append(0)
    
    return np.array(data, dtype="float32") / 255.0, np.array(labels)

def create_transfer_model():
    """Create transfer learning model with MobileNetV2"""
    # Load pre-trained MobileNetV2
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Freeze base model
    base_model.trainable = False
    
    # Add custom head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main():
    print("Loading dataset...")
    data, labels = load_data()
    print(f"Dataset loaded: {len(data)} images")
    print(f"With mask: {sum(labels)}, Without mask: {len(labels) - sum(labels)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        validation_split=0.2
    )
    
    # Create transfer learning model
    print("Creating transfer learning model...")
    model = create_transfer_model()
    model.summary()
    
    # Train model
    print("Training model...")
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=32, subset='training'),
        validation_data=datagen.flow(X_train, y_train, batch_size=32, subset='validation'),
        epochs=10,
        verbose=1
    )
    
    # Evaluate model
    print("Evaluating model...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Predictions
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Without Mask', 'With Mask']))
    
    # Save model
    model.save("transfer_face_mask_detector.h5")
    print("Transfer learning model saved!")

if __name__ == "__main__":
    main()