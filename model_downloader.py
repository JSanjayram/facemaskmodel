import os
import requests
import streamlit as st
from pathlib import Path

def download_model():
    """Download the trained model from cloud storage"""
    model_path = 'face_mask_detector.h5'
    
    # Check if model already exists
    if os.path.exists(model_path):
        return model_path
    
    # Model download URLs with your uploaded model
    model_urls = [
        "https://github.com/JSanjayram/facemaskmodel/raw/main/face_mask_detector.h5",
        "https://github.com/JSanjayram/facemaskmodel/raw/main/best_mask_model.h5"
    ]
    
    for url in model_urls:
        try:
            st.info(f"Downloading model from cloud storage...")
            response = requests.get(url, stream=True, timeout=30)
            
            if response.status_code == 200:
                total_size = int(response.headers.get('content-length', 0))
                
                with open(model_path, 'wb') as f:
                    downloaded = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            
                            if total_size > 0:
                                progress = downloaded / total_size
                                st.progress(progress)
                
                st.success("✅ Model downloaded successfully!")
                return model_path
                
        except Exception as e:
            st.warning(f"Failed to download from {url}: {str(e)}")
            continue
    
    st.error("❌ Could not download model from any source. Using fallback detection.")
    return None