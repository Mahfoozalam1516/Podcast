import streamlit as st
import os
import subprocess
import tempfile
import cv2
import librosa
import soundfile as sf
import shlex
import sys
import importlib
import gdown
from PIL import Image
from pathlib import Path
import numpy as np

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

    def process_media(self, input_path, audio_path, is_image=False, use_gan=False, nosmooth=True):
        """Process either image or video using Wav2Lip"""
        output_dir = "results"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "result_voice.mp4")
        
        model_path = os.path.join(self.checkpoint_dir, 
                                 "wav2lip_gan.pth" if use_gan else "wav2lip.pth")
        
        command = [
            "python", "inference.py",
            "--checkpoint_path", model_path,
            "--face", input_path,
            "--audio", audio_path,
            "--outfile", output_path,
            "--pads", "0", "0", "0", "0",
            "--resize_factor", "1"
        ]
        
        if nosmooth:
            command.append("--nosmooth")
            
        try:
            script_dir = os.path.dirname(os.path.abspath("inference.py"))
            result = subprocess.run(command, check=True, capture_output=True, text=True, 
                                 env=dict(os.environ, PYTHONPATH=script_dir))
            return output_path if os.path.exists(output_path) else None, result.stdout
        except subprocess.CalledProcessError as e:
            st.error(f"Error processing {'image' if is_image else 'video'}: {str(e.stderr)}")
            return None, str(e.stderr)

def check_module(module_name):
    """Check if a Python module is installed"""
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False

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
    st.title("WAV2LIP-GAN Lip Sync App")
    st.write("Generate lip-synced videos from either images or videos")

    # Initialize Wav2Lip
    wav2lip = Wav2LipInference()
    
    # Download models
    with st.spinner("Downloading pre-trained models..."):
        wav2lip.download_models()

    # Input type selection
    input_type = st.radio("Select Input Type", ["Image", "Video"])

    # File uploaders
    if input_type == "Image":
        input_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    else:
        input_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

    audio_file = st.file_uploader("Upload Audio", type=["wav", "mp3"])

    # Model options
    use_gan = st.checkbox("Use GAN model (better quality but slower)", value=True)
    nosmooth = st.checkbox("No smooth (faster processing)", value=True)

    if input_file and audio_file:
        # Save uploaded files to temporary location
        if input_type == "Image":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as input_tmp:
                img = Image.open(input_file)
                img.save(input_tmp.name, format="PNG")
                input_path = input_tmp.name
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as input_tmp:
                input_tmp.write(input_file.read())
                input_path = input_tmp.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as audio_tmp:
            # Convert audio to wav if needed
            if audio_file.type == "audio/mp3":
                audio_data, sr = librosa.load(audio_file, sr=None)
                sf.write(audio_tmp.name, audio_data, sr)
            else:
                audio_tmp.write(audio_file.read())
            audio_path = audio_tmp.name

        if st.button("Generate Lip-Synced Video"):
            # Check required modules
            required_modules = ['scipy', 'cv2', 'librosa']
            missing_modules = [module for module in required_modules if not check_module(module)]
            
            if missing_modules:
                st.error(f"Missing required modules: {', '.join(missing_modules)}")
                st.info("Please install the missing modules and ensure they are in your Python path.")
            else:
                with st.spinner("Processing... This may take a while."):
                    try:
                        # Resize video if needed
                        if input_type == "Video":
                            input_path = resize_video(input_path)

                        # Process with Wav2Lip
                        result_path, process_output = wav2lip.process_media(
                            input_path,
                            audio_path,
                            is_image=(input_type == "Image"),
                            use_gan=use_gan,
                            nosmooth=nosmooth
                        )

                        if result_path and os.path.exists(result_path):
                            st.success("Lip-synced video generated successfully!")
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
                            st.error("Failed to generate output video")
                            st.text(process_output)

                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")

                    finally:
                        # Cleanup temporary files
                        try:
                            os.unlink(input_path)
                            os.unlink(audio_path)
                        except:
                            pass

if __name__ == "__main__":
    main()