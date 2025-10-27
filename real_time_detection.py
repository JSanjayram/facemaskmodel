import cv2
import numpy as np
import tensorflow as tf
from mask_detection_model import FaceMaskDetector

class RealTimeMaskDetector:
    def __init__(self, model_path='face_mask_detector.h5'):
        self.detector = FaceMaskDetector()
        self.detector.load_model(model_path)
        
        # Load face cascade for face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
    def detect_faces_and_masks(self, frame):
        """Detect faces and predict mask status"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        results = []
        for (x, y, w, h) in faces:
            # Extract face region
            face_roi = frame[y:y+h, x:x+w]
            
            # Preprocess for model
            face_resized = cv2.resize(face_roi, (128, 128))
            face_normalized = face_resized / 255.0
            face_batch = np.expand_dims(face_normalized, axis=0)
            
            # Predict
            prediction = self.detector.model.predict(face_batch, verbose=0)[0][0]
            confidence = prediction if prediction > 0.5 else 1 - prediction
            
            mask_status = 'Without Mask' if prediction > 0.5 else 'With Mask'
            color = (0, 0, 255) if prediction > 0.5 else (0, 255, 0)
            
            results.append({
                'bbox': (x, y, w, h),
                'status': mask_status,
                'confidence': confidence,
                'color': color
            })
            
        return results
    
    def draw_predictions(self, frame, results):
        """Draw bounding boxes and predictions on frame"""
        for result in results:
            x, y, w, h = result['bbox']
            status = result['status']
            confidence = result['confidence']
            color = result['color']
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Draw label
            label = f"{status}: {confidence*100:.1f}%"
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
        return frame
    
    def run_webcam_detection(self):
        """Run real-time detection using webcam"""
        cap = cv2.VideoCapture(0)
        
        print("Starting real-time mask detection. Press 'q' to quit.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Detect faces and masks
            results = self.detect_faces_and_masks(frame)
            
            # Draw predictions
            frame = self.draw_predictions(frame, results)
            
            # Display frame
            cv2.imshow('Face Mask Detection', frame)
            
            # Break on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        detector = RealTimeMaskDetector()
        detector.run_webcam_detection()
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have a trained model file 'face_mask_detector.h5'")