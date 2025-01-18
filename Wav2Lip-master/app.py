import streamlit as st
import os
import tempfile
import subprocess
import cv2
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
import gdown
import torch
import mediapipe as mp
import logging
from PIL import Image
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceAligner:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5
        )

    def get_face_angle(self, image):
        """Calculate face rotation angle"""
        try:
            results = self.face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if not results.multi_face_landmarks:
                return None

            # Get specific landmarks for angle calculation
            landmarks = results.multi_face_landmarks[0].landmark
            left_eye = (landmarks[33].x, landmarks[33].y)
            right_eye = (landmarks[263].x, landmarks[263].y)
            
            # Calculate angle
            angle = np.degrees(np.arctan2(
                right_eye[1] - left_eye[1],
                right_eye[0] - left_eye[0]
            ))
            return angle
        except Exception as e:
            logger.error(f"Error calculating face angle: {str(e)}")
            return None

    def align_face(self, image, max_angle=60):
        """Align face if tilted"""
        try:
            angle = self.get_face_angle(image)
            if angle is None or abs(angle) > max_angle:
                return image
                
            if abs(angle) > 5:  # Only rotate if angle is significant
                height, width = image.shape[:2]
                center = (width // 2, height // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                aligned_image = cv2.warpAffine(
                    image, rotation_matrix, (width, height),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE
                )
                return aligned_image
            return image
        except Exception as e:
            logger.error(f"Error aligning face: {str(e)}")
            return image

class VideoProcessor:
    def __init__(self):
        self.face_aligner = FaceAligner()

    def process_video(self, input_path, output_path, target_height=720):
        """Process video: resize and align faces"""
        try:
            cap = cv2.VideoCapture(input_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            # Calculate new dimensions
            if height > target_height:
                scale = target_height / height
                new_width = int(width * scale)
                new_height = target_height
            else:
                new_width, new_height = width, height

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            progress_bar = st.progress(0)
            
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Resize frame
                if height > target_height:
                    frame = cv2.resize(frame, (new_width, new_height))
                
                # Align face in frame
                frame = self.face_aligner.align_face(frame)
                
                out.write(frame)
                
                # Update progress
                frame_idx += 1
                progress = int((frame_idx / frame_count) * 100)
                progress_bar.progress(progress / 100)

            cap.release()
            out.release()
            return True

        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            return False

class Wav2LipInference:
    def __init__(self):
        self.checkpoint_dir = "checkpoints"
        self.model_urls = {
            "wav2lip": "https://github.com/justinjohn0306/Wav2Lip/releases/download/models/wav2lip.pth",
            "wav2lip_gan": "https://github.com/justinjohn0306/Wav2Lip/releases/download/models/wav2lip_gan.pth"
        }
        self.video_processor = VideoProcessor()
        
    def download_models(self):
        """Download the pre-trained models if they don't exist"""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        for model_name, url in self.model_urls.items():
            model_path = os.path.join(self.checkpoint_dir, f"{model_name}.pth")
            if not os.path.exists(model_path):
                gdown.download(url, model_path, quiet=False)

    def process_video(self, video_path, audio_path, use_gan=False, nosmooth=True):
        """Process the video using Wav2Lip"""
        try:
            # Create temporary directory for processed files
            temp_dir = tempfile.mkdtemp()
            processed_video = os.path.join(temp_dir, "processed_input.mp4")
            
            # Process input video (resize and align faces)
            st.info("Pre-processing video (resizing and aligning faces)...")
            success = self.video_processor.process_video(video_path, processed_video)
            if not success:
                raise Exception("Failed to pre-process video")

            # Prepare output path
            output_path = "results/result_voice.mp4"
            os.makedirs("results", exist_ok=True)
            
            # Select model
            model_path = os.path.join(self.checkpoint_dir, 
                                    "wav2lip_gan.pth" if use_gan else "wav2lip.pth")
            
            # Prepare Wav2Lip command
            command = [
                "python", "inference.py",
                "--checkpoint_path", model_path,
                "--face", processed_video,
                "--audio", audio_path,
                "--pads", "0", "0", "0", "0",
                "--resize_factor", "1"
            ]
            
            if nosmooth:
                command.append("--nosmooth")
                
            st.info("Running Wav2Lip inference...")
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            # Monitor process output
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    st.text(output.strip())
            
            # Check result
            if process.returncode == 0 and os.path.exists(output_path):
                return output_path
            else:
                stderr = process.stderr.read()
                logger.error(f"Wav2Lip error: {stderr}")
                raise Exception(f"Wav2Lip processing failed: {stderr}")
                
        except Exception as e:
            logger.error(f"Error in process_video: {str(e)}")
            st.error(f"Processing error: {str(e)}")
            return None
            
        finally:
            # Cleanup temporary files
            try:
                if 'temp_dir' in locals():
                    shutil.rmtree(temp_dir)
            except Exception as e:
                logger.error(f"Cleanup error: {str(e)}")

def main():
    st.set_page_config(page_title="Advanced Wav2Lip Video Lip Sync", layout="wide")
    
    st.title("Advanced Wav2Lip Video Lip Sync")
    st.write("""
    Upload a video and audio file to synchronize lips with speech.
    This version supports:
    - Face alignment for tilted faces (up to 60 degrees)
    - Automatic video resolution adjustment
    - Enhanced preprocessing for better results
    """)
    
    # Initialize Wav2Lip
    wav2lip = Wav2LipInference()
    
    # Download models
    with st.spinner("Downloading pre-trained models..."):
        wav2lip.download_models()
    
    # Create columns for inputs
    col1, col2 = st.columns(2)
    
    with col1:
        video_file = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'])
        if video_file:
            st.video(video_file)
            
    with col2:
        audio_file = st.file_uploader("Upload Audio", type=['wav', 'mp3'])
        if audio_file:
            st.audio(audio_file)
    
    # Advanced options
    with st.expander("Advanced Options"):
        use_gan = st.checkbox("Use GAN model (better quality but slower)", value=False)
        nosmooth = st.checkbox("No smooth (faster processing)", value=True)
    
    if video_file and audio_file:
        try:
            # Save uploaded files to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as video_tmp:
                video_tmp.write(video_file.read())
                video_path = video_tmp.name
                
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as audio_tmp:
                if audio_file.type == "audio/mp3":
                    audio_data, sr = librosa.load(audio_file, sr=None)
                    sf.write(audio_tmp.name, audio_data, sr)
                else:
                    audio_tmp.write(audio_file.read())
                audio_path = audio_tmp.name
            
            if st.button("Generate Lip Sync Video"):
                with st.spinner("Processing... This may take a while."):
                    try:
                        result_path = wav2lip.process_video(
                            video_path, 
                            audio_path,
                            use_gan=use_gan,
                            nosmooth=nosmooth
                        )
                        
                        if result_path and os.path.exists(result_path):
                            st.success("Video processing completed!")
                            st.video(result_path)
                            
                            with open(result_path, 'rb') as file:
                                st.download_button(
                                    label="Download Result",
                                    data=file,
                                    file_name="lip_sync_result.mp4",
                                    mime="video/mp4"
                                )
                        else:
                            st.error("Failed to process video")
                            
                    except Exception as e:
                        st.error(f"Error during processing: {str(e)}")
                        logger.error(f"Processing error: {str(e)}")
                        
        except Exception as e:
            st.error(f"Error handling files: {str(e)}")
            logger.error(f"File handling error: {str(e)}")
            
        finally:
            # Cleanup temporary files
            try:
                if 'video_path' in locals():
                    os.unlink(video_path)
                if 'audio_path' in locals():
                    os.unlink(audio_path)
            except Exception as e:
                logger.error(f"Cleanup error: {str(e)}")

if __name__ == "__main__":
    main()