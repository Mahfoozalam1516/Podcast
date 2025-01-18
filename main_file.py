import streamlit as st
import os
import subprocess
import tempfile
from PIL import Image
import shlex
import sys
import importlib

def check_module(module_name):
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False

def run_wav2lip(audio_path, image_path, output_path, inference_script, checkpoint_path):
    script_dir = os.path.dirname(os.path.abspath(inference_script))
    sys.path.append(script_dir)
    
    command = f'python "{inference_script}" --checkpoint_path "{checkpoint_path}" --face "{image_path}" --audio "{audio_path}" --outfile "{output_path}"'
    try:
        result = subprocess.run(shlex.split(command), check=True, capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=script_dir))
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Command failed with error: {e.stderr}"

def main():
    st.title("WAV2LIP-GAN Lip Sync App")

    st.sidebar.header("Manual File Inputs")
    inference_script = st.sidebar.text_input("Path to inference.py", "inference.py")
    checkpoint_path = st.sidebar.text_input("Path to model checkpoint", "checkpoints/wav2lip_gan.pth")

    audio_file = st.file_uploader("Upload an audio file (WAV format)", type=["wav"])
    image_file = st.file_uploader("Upload an avatar image", type=["jpg", "jpeg", "png"])

    if audio_file and image_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(audio_file.read())
            audio_path = temp_audio.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_image:
            img = Image.open(image_file)
            img.save(temp_image.name, format="PNG")
            image_path = temp_image.name

        # Use a relative path for output
        output_dir = "Outputs"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, "output_video1.mp4")

        if st.button("Generate Lip-Synced Video"):
            if not os.path.exists(inference_script):
                st.error(f"Inference script not found at {inference_script}")
            elif not os.path.exists(checkpoint_path):
                st.error(f"Model checkpoint not found at {checkpoint_path}")
            else:
                required_modules = ['scipy', 'cv2', 'audio']
                missing_modules = [module for module in required_modules if not check_module(module)]
                
                if missing_modules:
                    st.error(f"Missing required modules: {', '.join(missing_modules)}")
                    st.info("Please install the missing modules and ensure they are in your Python path.")
                else:
                    with st.spinner("Processing... This may take a while."):
                        try:
                            result = run_wav2lip(audio_path, image_path, output_path, inference_script, checkpoint_path)
                            if "Command failed" in result:
                                st.error(result)
                            else:
                                st.success("Lip-synced video generated successfully!")
                                st.text(result)
                                
                                # Debug information
                                st.text(f"Current working directory: {os.getcwd()}")
                                st.text(f"Output path: {output_path}")
                                st.text(f"Output directory exists: {os.path.exists(os.path.dirname(output_path))}")
                                st.text(f"Output file exists: {os.path.exists(output_path)}")
                                
                                if os.path.exists(output_path):
                                    st.video(output_path)
                                else:
                                    st.error(f"Output video not found at {output_path}")
                                    # List files in the output directory
                                    st.text("Files in output directory:")
                                    for file in os.listdir(os.path.dirname(output_path)):
                                        st.text(file)
                        except Exception as e:
                            st.error(f"An error occurred: {str(e)}")

        os.unlink(audio_path)
        os.unlink(image_path)

if __name__ == "__main__":
    main()