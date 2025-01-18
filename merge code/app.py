import streamlit as st
import os
import tempfile
import subprocess
import cv2
import librosa
import soundfile as sf
import shutil
import numpy as np
from pathlib import Path
import gdown
import torch

class Wav2LipInference:
    def __init__(self):
        self.checkpoint_dir = "checkpoints"
        self.model_urls = {
            "wav2lip": "https://github.com/justinjohn0306/Wav2Lip/releases/download/models/wav2lip.pth",
            "wav2lip_gan": "https://github.com/justinjohn0306/Wav2Lip/releases/download/models/wav2lip_gan.pth"
        }
        
    def download_models(self):
        """Download the pre-trained models if they don't exist"""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        for model_name, url in self.model_urls.items():
            model_path = os.path.join(self.checkpoint_dir, f"{model_name}.pth")
            if not os.path.exists(model_path):
                gdown.download(url, model_path, quiet=False)

    def process_video(self, video_path, audio_path, use_gan=False, nosmooth=True):
        """Process the video using Wav2Lip"""
        output_path = "results/result_voice.mp4"
        os.makedirs("results", exist_ok=True)
        
        model_path = os.path.join(self.checkpoint_dir, 
                                 "wav2lip_gan.pth" if use_gan else "wav2lip.pth")
        
        command = [
            "python", "inference.py",
            "--checkpoint_path", model_path,
            "--face", video_path,
            "--audio", audio_path,
            "--pads", "0", "0", "0", "0",
            "--resize_factor", "1"
        ]
        
        if nosmooth:
            command.append("--nosmooth")
            
        try:
            subprocess.run(command, check=True)
            return output_path if os.path.exists(output_path) else None
        except subprocess.CalledProcessError as e:
            st.error(f"Error processing video: {str(e)}")
            return None

def get_video_resolution(video_path):
    """Get the resolution of a video file"""
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width, height

def resize_video(video_path, max_height=720):
    """Resize video if it's larger than max_height"""
    width, height = get_video_resolution(video_path)
    if height > max_height:
        scale = max_height / height
        new_width = int(width * scale)
        new_height = max_height
        
        output_path = f"{video_path}_resized.mp4"
        command = [
            "ffmpeg", "-i", video_path,
            "-vf", f"scale={new_width}:{new_height}",
            "-c:a", "copy",
            output_path
        ]
        
        subprocess.run(command, check=True)
        return output_path
    return video_path

def main():
    st.title("Wav2Lip Video Lip Sync")
    st.write("Upload a video and audio file to synchronize lips with speech")
    
    # Initialize Wav2Lip
    wav2lip = Wav2LipInference()
    
    # Download models
    with st.spinner("Downloading pre-trained models..."):
        wav2lip.download_models()
    
    # File uploaders
    video_file = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'])
    audio_file = st.file_uploader("Upload Audio", type=['wav', 'mp3'])
    
    # Model options
    use_gan = st.checkbox("Use GAN model (better quality but slower)", value=False)
    nosmooth = st.checkbox("No smooth (faster processing)", value=True)
    
    if video_file and audio_file:
        # Save uploaded files to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as video_tmp:
            video_tmp.write(video_file.read())
            video_path = video_tmp.name
            
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as audio_tmp:
            # Convert audio to wav if needed
            if audio_file.type == "audio/mp3":
                audio_data, sr = librosa.load(audio_file, sr=None)
                sf.write(audio_tmp.name, audio_data, sr)
            else:
                audio_tmp.write(audio_file.read())
            audio_path = audio_tmp.name
        
        # Process files
        if st.button("Generate Lip Sync Video"):
            with st.spinner("Processing... This may take a while."):
                # Resize video if needed
                video_path = resize_video(video_path)
                
                # Process with Wav2Lip
                result_path = wav2lip.process_video(
                    video_path, 
                    audio_path,
                    use_gan=use_gan,
                    nosmooth=nosmooth
                )
                
                if result_path and os.path.exists(result_path):
                    # Display result
                    st.video(result_path)
                    
                    # Download button
                    with open(result_path, 'rb') as file:
                        st.download_button(
                            label="Download Result",
                            data=file,
                            file_name="lip_sync_result.mp4",
                            mime="video/mp4"
                        )
                else:
                    st.error("Failed to process video")
        
        # Cleanup temporary files
        try:
            os.unlink(video_path)
            os.unlink(audio_path)
        except:
            pass

if __name__ == "__main__":
    main()