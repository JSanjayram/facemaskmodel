import tensorflow as tf
import tensorflowjs as tfjs
import os

def convert_model_to_tfjs():
    """Convert the trained Keras model to TensorFlow.js format"""
    
    # Load the trained model
    model_path = 'face_mask_model.h5'
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found!")
        return False
    
    try:
        # Load model
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully")
        
        # Create output directory
        output_dir = 'tfjs_model'
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert to TensorFlow.js format
        tfjs.converters.save_keras_model(model, output_dir)
        print(f"Model converted successfully to {output_dir}")
        
        # Print model info
        print("\nModel Summary:")
        model.summary()
        
        return True
        
    except Exception as e:
        print(f"Error converting model: {e}")
        return False

if __name__ == "__main__":
    convert_model_to_tfjs()