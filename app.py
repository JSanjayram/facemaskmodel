import streamlit as st
import numpy as np
from PIL import Image
import io
import requests
try:
    import cv2
    import tensorflow as tf
    from mask_detection_model import FaceMaskDetector
    DEPS_AVAILABLE = True
except ImportError as e:
    st.error(f"Missing dependencies: {e}")
    DEPS_AVAILABLE = False
except Exception as e:
    st.error(f"Error loading dependencies: {e}")
    DEPS_AVAILABLE = False

# Configure Streamlit page
st.set_page_config(
    page_title="Face Mask Detection",
    page_icon="😷",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Dark theme CSS
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    .success-result {
        background: #1e3a2e;
        border: 1px solid #28a745;
        color: #00ff00;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .warning-result {
        background: #3a1e1e;
        border: 1px solid #dc3545;
        color: #ff4444;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    h1 {
        text-align: center !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model"""
    if not DEPS_AVAILABLE:
        return None
    try:
        detector = FaceMaskDetector()
        detector.load_model('face_mask_detector.h5')
        return detector
    except Exception:
        return None

def process_image(image, detector):
    """Process uploaded image for mask detection"""
    img_array = np.array(image)
    
    if not DEPS_AVAILABLE or detector is None:
        return img_array, []
    
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        results = []
        for (x, y, w, h) in faces:
            face_roi = img_array[y:y+h, x:x+w]
            face_resized = cv2.resize(face_roi, (128, 128))
            face_normalized = face_resized / 255.0
            face_batch = np.expand_dims(face_normalized, axis=0)
            
            prediction = detector.model.predict(face_batch, verbose=0)[0][0]
            confidence = prediction if prediction > 0.5 else 1 - prediction
            
            mask_status = 'Without Mask' if prediction > 0.5 else 'With Mask'
            color = (255, 0, 0) if prediction > 0.5 else (0, 255, 0)
            
            cv2.rectangle(img_array, (x, y), (x+w, y+h), color, 3)
            label = f"{mask_status}: {confidence*100:.1f}%"
            cv2.putText(img_array, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            results.append({
                'status': mask_status,
                'confidence': confidence * 100,
                'bbox': (x, y, w, h)
            })
        
        return img_array, results
    except Exception:
        return img_array, []

def main():
    st.title("Face Mask Detection System")
    st.markdown("<div style='text-align: center;'><strong>AI-powered mask detection using CNN | 90%+ Accuracy</strong></div>", unsafe_allow_html=True)
    
    # Load model with error handling
    try:
        detector = load_model()
        if detector is None:
            st.warning("⚠️ Model not found! Upload 'face_mask_detector.h5' to enable real detection.")
        else:
            st.success("✅ Real CNN Model Loaded Successfully!")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        detector = None
    
    tab1, tab2, tab3 = st.tabs(["Image Detection", "Model Info", "Live Camera"])
    
    with tab1:
        st.header("Image Detection")
        
        option = st.radio("Input method:", ["Upload File", "Image URL"], horizontal=True)
        
        uploaded_file = None
        image_url = None
        
        if option == "Upload File":
            uploaded_file = st.file_uploader("Choose image", type=['jpg', 'jpeg', 'png'])
        else:
            image_url = st.text_input("Image URL", placeholder="https://example.com/image.jpg")
        
        image = None
        
        if uploaded_file is not None:
            if uploaded_file.size > 10 * 1024 * 1024:
                st.error("File too large. Please upload an image smaller than 10MB.")
                st.stop()
            
            try:
                bytes_data = uploaded_file.getvalue()
                image = Image.open(io.BytesIO(bytes_data))
                if image.mode != 'RGB':
                    image = image.convert('RGB')
            except Exception as e:
                st.error(f"Error loading image: {str(e)}. Please upload a valid image file.")
                st.stop()
        
        elif image_url:
            try:
                response = requests.get(image_url, timeout=10)
                if response.status_code == 200:
                    image = Image.open(io.BytesIO(response.content))
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                else:
                    st.error(f"Failed to load image from URL. Status code: {response.status_code}")
                    st.stop()
            except Exception as e:
                st.error(f"Error loading image from URL: {str(e)}")
                st.stop()
        
        if image is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(image, use_column_width=True)
            
            with col2:
                st.subheader("Detection Results")
                
                with st.spinner("Analyzing image..."):
                    processed_img, results = process_image(image, detector)
                
                st.image(processed_img, use_column_width=True)
                
                if results:
                    st.markdown(f"### Detection Results: {len(results)} face(s) found")
                    
                    for i, result in enumerate(results):
                        status = result['status']
                        confidence = result['confidence']
                        
                        if status == 'With Mask':
                            st.markdown(f"""
                            <div class="success-result">
                                <strong>Face {i+1}:</strong> {status} 
                                <span style="float: right;">{confidence:.1f}%</span>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="warning-result">
                                <strong>Face {i+1}:</strong> {status} 
                                <span style="float: right;">{confidence:.1f}%</span>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.info("No faces detected in the image. Try a different image with visible faces.")
    
    with tab2:
        st.header("Model Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Architecture")
            st.code("""
CNN Architecture:
- Conv2D(32) + BatchNorm + MaxPool + Dropout
- Conv2D(64) + BatchNorm + MaxPool + Dropout  
- Conv2D(128) + BatchNorm + MaxPool + Dropout
- Conv2D(256) + BatchNorm + MaxPool + Dropout
- Dense(512) + BatchNorm + Dropout
- Dense(256) + BatchNorm + Dropout
- Dense(1, sigmoid) - Output Layer
            """)
        
        with col2:
            st.subheader("Model Performance")
            st.metric("Target Accuracy", "90%+")
            st.metric("Input Size", "128x128x3")
            st.metric("Classes", "2 (With/Without Mask)")
    
    with tab3:
        st.header("Real-time Detection")
        st.info("🚀 **Real-time Browser Detection with TensorFlow.js**")
        
        camera_html = """
        <div style="background: #1e1e1e; padding: 20px; border-radius: 10px; margin: 10px 0;">
            <div style="text-align: center; margin-bottom: 20px;">
                <button id="startBtn" onclick="startCamera()" style="background: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 5px; margin: 5px; cursor: pointer;">Start Camera</button>
                <button id="stopBtn" onclick="stopCamera()" style="background: #f44336; color: white; padding: 10px 20px; border: none; border-radius: 5px; margin: 5px; cursor: pointer;" disabled>Stop Camera</button>
            </div>
            
            <div style="position: relative; display: inline-block; margin: 0 auto;">
                <video id="video" width="640" height="480" autoplay style="display: none; border-radius: 10px;"></video>
                <canvas id="overlay" width="640" height="480" style="position: absolute; top: 0; left: 0; pointer-events: none;"></canvas>
            </div>
            
            <div id="results" style="color: white; text-align: center; margin-top: 10px; font-size: 16px; font-weight: bold;">Click Start Camera to begin detection</div>
            <div id="status" style="color: #888; text-align: center; margin-top: 5px;">Loading model...</div>
        </div>
        
        <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest"></script>
        <script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/blazeface@latest"></script>
        
        <script>
        let model = null;
        let faceModel = null;
        let stream = null;
        let isDetecting = false;
        let video = null;
        let canvas = null;
        
        async function loadModels() {
            try {
                document.getElementById('status').innerText = 'Loading face detection model...';
                faceModel = await blazeface.load();
                
                document.getElementById('status').innerText = 'Loading mask detection model...';
                model = await tf.loadLayersModel('/tfjs_model/model.json');
                
                document.getElementById('status').innerText = 'Real CNN model loaded! Ready for detection.';
                return true;
            } catch (error) {
                console.error('Error loading models:', error);
                document.getElementById('status').innerText = 'Error loading CNN model. Convert your model first.';
                return false;
            }
        }
        
        async function startCamera() {
            try {
                stream = await navigator.mediaDevices.getUserMedia({
                    video: { width: 640, height: 480, facingMode: 'user' }
                });
                
                video = document.getElementById('video');
                canvas = document.getElementById('overlay');
                
                video.srcObject = stream;
                video.style.display = 'block';
                
                document.getElementById('startBtn').disabled = true;
                document.getElementById('stopBtn').disabled = false;
                
                video.onloadedmetadata = () => {
                    isDetecting = true;
                    detectLoop();
                };
                
            } catch (err) {
                alert('Camera access denied: ' + err.message);
            }
        }
        
        async function detectLoop() {
            if (!isDetecting) return;
            
            const ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, 640, 480);
            
            try {
                const faces = await faceModel.estimateFaces(video, false);
                let resultText = '';
                
                for (let i = 0; i < faces.length; i++) {
                    const face = faces[i];
                    const [x, y] = face.topLeft;
                    const [x2, y2] = face.bottomRight;
                    const width = x2 - x;
                    const height = y2 - y;
                    
                    const faceCanvas = document.createElement('canvas');
                    faceCanvas.width = 128;
                    faceCanvas.height = 128;
                    const faceCtx = faceCanvas.getContext('2d');
                    
                    faceCtx.drawImage(video, x, y, width, height, 0, 0, 128, 128);
                    
                    let prediction;
                    if (model) {
                        const tensor = tf.browser.fromPixels(faceCanvas)
                            .resizeNearestNeighbor([128, 128])
                            .toFloat()
                            .div(255.0)
                            .expandDims();
                        
                        const pred = await model.predict(tensor).data();
                        const confidence = pred[0] > 0.5 ? pred[0] : 1 - pred[0];
                        prediction = {
                            hasMask: pred[0] < 0.5,
                            confidence: confidence * 100
                        };
                        tensor.dispose();
                    } else {
                        prediction = { hasMask: false, confidence: 0 };
                    }
                    
                    ctx.strokeStyle = prediction.hasMask ? '#00ff00' : '#ff0000';
                    ctx.lineWidth = 3;
                    ctx.strokeRect(x, y, width, height);
                    
                    const label = prediction.hasMask ? 'With Mask' : 'Without Mask';
                    ctx.fillStyle = prediction.hasMask ? '#00ff00' : '#ff0000';
                    ctx.font = '16px Arial';
                    ctx.fillText(`${label} (${prediction.confidence.toFixed(1)}%)`, x, y - 10);
                    
                    resultText += `Face ${i + 1}: ${label} (${prediction.confidence.toFixed(1)}%) `;
                }
                
                document.getElementById('results').innerHTML = resultText || 'No faces detected';
                    
            } catch (error) {
                console.error('Detection error:', error);
            }
            
            requestAnimationFrame(detectLoop);
        }
        
        function stopCamera() {
            isDetecting = false;
            
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
                stream = null;
            }
            
            if (video) {
                video.style.display = 'none';
            }
            
            if (canvas) {
                canvas.getContext('2d').clearRect(0, 0, 640, 480);
            }
            
            document.getElementById('startBtn').disabled = false;
            document.getElementById('stopBtn').disabled = true;
            document.getElementById('results').innerHTML = 'Camera stopped';
        }
        
        window.onload = function() {
            loadModels();
        };
        
        </script>
        """
        
        st.components.v1.html(camera_html, height=700)
        
        st.subheader("Model Conversion Required")
        
        with st.expander("📋 Convert Your CNN Model for Browser Use"):
            st.markdown("""
            **Step 1: Install TensorFlow.js Converter**
            ```bash
            pip install tensorflowjs
            ```
            
            **Step 2: Convert Your Model**
            ```bash
            tensorflowjs_converter --input_format=keras face_mask_detector.h5 tfjs_model/
            ```
            
            **Step 3: Host Model Files**
            - Upload `tfjs_model/` folder to your web server
            - Model will load automatically in browser
            
            **Real Model Features:**
            - ✅ Your actual trained CNN predictions
            - ✅ 90%+ accuracy mask detection
            - ✅ Real confidence scores from your model
            """)

if __name__ == "__main__":
    main()