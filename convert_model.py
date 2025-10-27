import tensorflow as tf
import os

def convert_model_to_tfjs():
    """Convert the trained Keras model to TensorFlow.js format"""
    
    model_path = 'face_mask_model.h5'
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found!")
        return False
    
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully")
        
        output_dir = 'tfjs_model'
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as SavedModel first, then convert
        tf.saved_model.save(model, 'temp_saved_model')
        
        # Use tensorflowjs_converter command
        os.system(f'tensorflowjs_converter --input_format=tf_saved_model temp_saved_model {output_dir}')
        
        print(f"Model converted successfully to {output_dir}")
        return True
        
    except Exception as e:
        print(f"Error converting model: {e}")
        return False

if __name__ == "__main__":
    convert_model_to_tfjs()