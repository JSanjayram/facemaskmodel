"""
Mobile App Template for Face Mask Detection
This is a Kivy-based mobile app template
"""

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.camera import Camera
from kivy.clock import Clock
import tensorflow as tf
import numpy as np
import cv2

class MaskDetectionApp(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        
        # Load TensorFlow Lite model
        self.interpreter = tf.lite.Interpreter(model_path='mask_detector_mobile.tflite')
        self.interpreter.allocate_tensors()
        
        # Camera
        self.camera = Camera(play=True, resolution=(640, 480))
        self.add_widget(self.camera)
        
        # Status label
        self.status_label = Label(text='Ready to detect masks', size_hint_y=0.1)
        self.add_widget(self.status_label)
        
        # Control buttons
        button_layout = BoxLayout(size_hint_y=0.1)
        
        self.detect_btn = Button(text='Start Detection')
        self.detect_btn.bind(on_press=self.toggle_detection)
        button_layout.add_widget(self.detect_btn)
        
        self.add_widget(button_layout)
        
        # Detection state
        self.detecting = False
        
    def toggle_detection(self, instance):
        if self.detecting:
            self.detecting = False
            self.detect_btn.text = 'Start Detection'
            Clock.unschedule(self.detect_mask)
        else:
            self.detecting = True
            self.detect_btn.text = 'Stop Detection'
            Clock.schedule_interval(self.detect_mask, 1.0/10.0)  # 10 FPS
    
    def detect_mask(self, dt):
        # Get camera frame
        texture = self.camera.texture
        if texture is None:
            return
            
        # Convert texture to numpy array
        frame = np.frombuffer(texture.pixels, np.uint8)
        frame = frame.reshape(texture.height, texture.width, 4)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        
        # Preprocess for model
        resized = cv2.resize(frame, (128, 128))
        normalized = resized / 255.0
        input_data = np.expand_dims(normalized, axis=0).astype(np.float32)
        
        # Run inference
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        
        self.interpreter.set_tensor(input_details[0]['index'], input_data)
        self.interpreter.invoke()
        
        prediction = self.interpreter.get_tensor(output_details[0]['index'])[0][0]
        
        # Update status
        if prediction > 0.5:
            self.status_label.text = f'Without Mask - {prediction*100:.1f}%'
            self.status_label.color = (1, 0, 0, 1)  # Red
        else:
            self.status_label.text = f'With Mask - {(1-prediction)*100:.1f}%'
            self.status_label.color = (0, 1, 0, 1)  # Green

class MaskDetectionMobileApp(App):
    def build(self):
        return MaskDetectionApp()

if __name__ == '__main__':
    MaskDetectionMobileApp().run()