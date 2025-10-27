import cv2
import numpy as np
from mask_detection_model import FaceMaskDetector
from alert_system import AlertSystem

class IntegratedMaskDetection:
    def __init__(self, model_path='face_mask_detector.h5'):
        self.detector = FaceMaskDetector()
        self.detector.load_model(model_path)
        self.alert_system = AlertSystem()
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Alert settings
        self.alert_threshold = 0.8  # Confidence threshold for alerts
        self.violation_count = 0
        self.max_violations = 3  # Trigger alert after 3 violations
        
    def process_frame_with_alerts(self, frame, location="Camera 1"):
        """Process frame and trigger alerts if needed"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        violations = []
        
        for (x, y, w, h) in faces:
            # Extract and preprocess face
            face_roi = frame[y:y+h, x:x+w]
            face_resized = cv2.resize(face_roi, (128, 128))
            face_normalized = face_resized / 255.0
            face_batch = np.expand_dims(face_normalized, axis=0)
            
            # Predict
            prediction = self.detector.model.predict(face_batch, verbose=0)[0][0]
            confidence = prediction if prediction > 0.5 else 1 - prediction
            
            mask_status = 'Without Mask' if prediction > 0.5 else 'With Mask'
            color = (0, 0, 255) if prediction > 0.5 else (0, 255, 0)
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Add label
            label = f"{mask_status}: {confidence*100:.1f}%"
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Check for violations
            if mask_status == 'Without Mask' and confidence > self.alert_threshold:
                violations.append({
                    'status': mask_status,
                    'confidence': confidence * 100,
                    'bbox': (x, y, w, h)
                })
        
        # Handle violations
        if violations:
            self.violation_count += len(violations)
            
            if self.violation_count >= self.max_violations:
                for violation in violations:
                    self.alert_system.trigger_alert(violation, location)
                self.violation_count = 0  # Reset counter
        
        return frame, violations
    
    def run_monitoring_system(self, camera_id=0, location="Main Entrance"):
        """Run continuous monitoring with alerts"""
        cap = cv2.VideoCapture(camera_id)
        
        print(f"Starting mask monitoring at {location}")
        print("Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame with alerts
            processed_frame, violations = self.process_frame_with_alerts(frame, location)
            
            # Display violation count
            cv2.putText(processed_frame, f"Violations: {self.violation_count}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Show frame
            cv2.imshow(f'Mask Detection - {location}', processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    monitor = IntegratedMaskDetection()
    monitor.run_monitoring_system(location="Main Entrance")