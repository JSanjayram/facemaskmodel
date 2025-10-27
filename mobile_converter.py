import tensorflow as tf
import numpy as np

class MobileConverter:
    def __init__(self, model_path='face_mask_detector.h5'):
        self.model_path = model_path
        
    def convert_to_tflite(self, output_path='mask_detector_mobile.tflite'):
        """Convert Keras model to TensorFlow Lite for mobile deployment"""
        model = tf.keras.models.load_model(self.model_path)
        
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        tflite_model = converter.convert()
        
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
            
        print(f"Mobile model saved: {output_path}")
        return output_path

if __name__ == "__main__":
    converter = MobileConverter()
    converter.convert_to_tflite()