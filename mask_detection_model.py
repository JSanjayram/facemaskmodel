import tensorflow as tf
import numpy as np

class FaceMaskDetector:
    def __init__(self):
        self.model = None
        
    def load_model(self, model_path):
        """Load the trained model"""
        try:
            self.model = tf.keras.models.load_model(model_path)
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def predict(self, image):
        """Predict mask presence in image"""
        if self.model is None:
            return None
        
        # Preprocess image
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        # Normalize
        image = image / 255.0
        
        # Predict
        prediction = self.model.predict(image, verbose=0)
        return prediction[0][0]