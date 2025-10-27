# 📱 Mobile Setup Guide

## Quick Steps to Use on Mobile

### 1. Convert Model
```bash
python mobile_converter.py
```

### 2. Install Mobile Dependencies
```bash
pip install kivy kivymd buildozer
```

### 3. Build Android APK
```bash
buildozer android debug
```

### 4. Install APK
- Transfer `bin/maskdetector-1.0-debug.apk` to phone
- Enable "Unknown Sources" in Android settings
- Install APK

### 5. Use App
- Open "Mask Detector" app
- Grant camera permission
- Tap "Start Detection"
- Point camera at faces

## Alternative: Web App on Mobile

### Use Streamlit App
```bash
streamlit run app.py --server.port 8501
```
- Open browser on phone
- Go to `http://your-computer-ip:8501`
- Upload photos or use camera

## Files Needed on Mobile
- `mask_detector_mobile.tflite` (converted model)
- `mobile_app_template.py` (main app)
- `buildozer.spec` (build config)

## Permissions Required
- Camera access
- Storage access
- Internet (for alerts)