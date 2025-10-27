import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import os

class FaceMaskDetector:
    def __init__(self, input_shape=(128, 128, 3)):
        self.input_shape = input_shape
        self.model = None
        
    def build_model(self):
        """Build enhanced CNN model for production-level accuracy"""
        self.model = Sequential([
            # First Conv Block
            Conv2D(32, (3,3), activation='relu', input_shape=self.input_shape),
            BatchNormalization(),
            MaxPooling2D(2,2),
            Dropout(0.25),
            
            # Second Conv Block
            Conv2D(64, (3,3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2,2),
            Dropout(0.25),
            
            # Third Conv Block
            Conv2D(128, (3,3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2,2),
            Dropout(0.25),
            
            # Fourth Conv Block
            Conv2D(256, (3,3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2,2),
            Dropout(0.25),
            
            # Dense Layers
            Flatten(),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def get_data_generators(self, train_dir, val_dir, batch_size=32):
        """Create data generators with augmentation"""
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='binary'
        )
        
        val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='binary'
        )
        
        return train_generator, val_generator
    
    def train(self, train_generator, val_generator, epochs=50):
        """Train the model with callbacks"""
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.2, patience=5, min_lr=1e-7),
            ModelCheckpoint('best_mask_model.h5', save_best_only=True)
        ]
        
        history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=callbacks
        )
        
        return history
    
    def predict_image(self, image_path):
        """Predict mask detection for single image"""
        img = cv2.imread(image_path)
        img = cv2.resize(img, self.input_shape[:2])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        
        prediction = self.model.predict(img)[0][0]
        confidence = prediction if prediction > 0.5 else 1 - prediction
        
        return {
            'class': 'Without Mask' if prediction > 0.5 else 'With Mask',
            'confidence': f"{confidence * 100:.1f}%"
        }
    
    def save_model(self, filepath):
        """Save trained model"""
        self.model.save(filepath)
    
    def load_model(self, filepath):
        """Load pre-trained model"""
        self.model = tf.keras.models.load_model(filepath)

if __name__ == "__main__":
    # Initialize detector
    detector = FaceMaskDetector()
    model = detector.build_model()
    
    print("Model Architecture:")
    model.summary()
    
    # Example usage (uncomment when you have data)
    # train_gen, val_gen = detector.get_data_generators('data/train', 'data/val')
    # history = detector.train(train_gen, val_gen)
    # detector.save_model('face_mask_detector.h5')