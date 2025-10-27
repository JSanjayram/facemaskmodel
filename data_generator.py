import os
import requests
import zipfile
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2

class MaskDatasetGenerator:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        
    def download_dataset(self):
        """Download and extract mask dataset"""
        print("Creating sample dataset structure...")
        
        # Create directory structure
        os.makedirs(f"{self.data_dir}/train/with_mask", exist_ok=True)
        os.makedirs(f"{self.data_dir}/train/without_mask", exist_ok=True)
        os.makedirs(f"{self.data_dir}/val/with_mask", exist_ok=True)
        os.makedirs(f"{self.data_dir}/val/without_mask", exist_ok=True)
        
        print("Dataset structure created!")
        print("Please add your images to:")
        print(f"- {self.data_dir}/train/with_mask/")
        print(f"- {self.data_dir}/train/without_mask/")
        print(f"- {self.data_dir}/val/with_mask/")
        print(f"- {self.data_dir}/val/without_mask/")
        
    def generate_synthetic_data(self, num_samples=1000):
        """Generate synthetic training data for testing"""
        print("Generating synthetic data for testing...")
        
        for split in ['train', 'val']:
            for class_name in ['with_mask', 'without_mask']:
                dir_path = f"{self.data_dir}/{split}/{class_name}"
                os.makedirs(dir_path, exist_ok=True)
                
                samples = num_samples if split == 'train' else num_samples // 5
                
                for i in range(samples):
                    # Generate random image
                    img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
                    
                    # Add some pattern to differentiate classes
                    if class_name == 'with_mask':
                        # Add mask-like pattern (rectangle in lower face area)
                        cv2.rectangle(img, (30, 80), (98, 110), (255, 255, 255), -1)
                    else:
                        # Add face-like features
                        cv2.circle(img, (45, 70), 3, (0, 0, 0), -1)  # Eye
                        cv2.circle(img, (83, 70), 3, (0, 0, 0), -1)  # Eye
                        cv2.ellipse(img, (64, 95), (10, 5), 0, 0, 180, (255, 0, 0), 2)  # Mouth
                    
                    # Save image
                    cv2.imwrite(f"{dir_path}/sample_{i:04d}.jpg", img)
                
                print(f"Generated {samples} samples for {split}/{class_name}")
    
    def get_data_info(self):
        """Get information about the dataset"""
        info = {}
        
        for split in ['train', 'val']:
            info[split] = {}
            for class_name in ['with_mask', 'without_mask']:
                dir_path = f"{self.data_dir}/{split}/{class_name}"
                if os.path.exists(dir_path):
                    count = len([f for f in os.listdir(dir_path) if f.endswith(('.jpg', '.png', '.jpeg'))])
                    info[split][class_name] = count
                else:
                    info[split][class_name] = 0
        
        return info
    
    def create_data_generators(self, batch_size=32):
        """Create data generators for training"""
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
            f"{self.data_dir}/train",
            target_size=(128, 128),
            batch_size=batch_size,
            class_mode='binary'
        )
        
        val_generator = val_datagen.flow_from_directory(
            f"{self.data_dir}/val",
            target_size=(128, 128),
            batch_size=batch_size,
            class_mode='binary'
        )
        
        return train_generator, val_generator

if __name__ == "__main__":
    generator = MaskDatasetGenerator()
    
    # Create dataset structure
    generator.download_dataset()
    
    # Generate synthetic data for testing
    generator.generate_synthetic_data(500)
    
    # Show dataset info
    info = generator.get_data_info()
    print("\nDataset Information:")
    for split, classes in info.items():
        print(f"{split.upper()}:")
        for class_name, count in classes.items():
            print(f"  {class_name}: {count} images")
    
    print("\nDataset ready for training!")