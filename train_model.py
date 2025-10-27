import matplotlib.pyplot as plt
import numpy as np
from mask_detection_model import FaceMaskDetector
from data_generator import MaskDatasetGenerator

def plot_training_history(history):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    print("🎭 Face Mask Detection Model Training")
    print("=" * 50)
    
    # Initialize data generator
    data_gen = MaskDatasetGenerator()
    
    # Check if data exists, if not create synthetic data
    info = data_gen.get_data_info()
    total_samples = sum([sum(classes.values()) for classes in info.values()])
    
    if total_samples == 0:
        print("No data found. Generating synthetic dataset...")
        data_gen.download_dataset()
        data_gen.generate_synthetic_data(1000)
        info = data_gen.get_data_info()
    
    # Display dataset info
    print("\n📊 Dataset Information:")
    for split, classes in info.items():
        print(f"{split.upper()}:")
        for class_name, count in classes.items():
            print(f"  {class_name}: {count} images")
    
    # Create data generators
    print("\n🔄 Creating data generators...")
    train_gen, val_gen = data_gen.create_data_generators(batch_size=32)
    
    # Initialize and build model
    print("\n🏗️ Building model...")
    detector = FaceMaskDetector()
    model = detector.build_model()
    
    print("\n📋 Model Architecture:")
    model.summary()
    
    # Train model
    print("\n🚀 Starting training...")
    history = detector.train(train_gen, val_gen, epochs=30)
    
    # Save model
    print("\n💾 Saving model...")
    detector.save_model('face_mask_detector.h5')
    
    # Plot training history
    print("\n📈 Plotting training history...")
    plot_training_history(history)
    
    # Evaluate model
    print("\n📊 Final Model Performance:")
    final_acc = max(history.history['val_accuracy'])
    final_loss = min(history.history['val_loss'])
    
    print(f"Best Validation Accuracy: {final_acc:.4f} ({final_acc*100:.2f}%)")
    print(f"Best Validation Loss: {final_loss:.4f}")
    
    if final_acc >= 0.90:
        print("🎉 SUCCESS: Model achieved 90%+ accuracy!")
    else:
        print("⚠️  Model accuracy below 90%. Consider:")
        print("   - Adding more training data")
        print("   - Increasing training epochs")
        print("   - Adjusting model architecture")
    
    print("\n✅ Training completed!")
    print("Model saved as 'face_mask_detector.h5'")
    print("Run 'python real_time_detection.py' for webcam detection")

if __name__ == "__main__":
    main()