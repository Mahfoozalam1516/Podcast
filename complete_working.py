import streamlit as st
import asyncio
import edge_tts
import io
import base64
from diffusers import StableDiffusionPipeline, ControlNetModel, StableDiffusionControlNetPipeline
from controlnet_aux import CannyDetector
import torch
from PIL import Image
import os
from diffusers.schedulers import DPMSolverMultistepScheduler
import insightface
import cv2
import numpy as np
import time
import subprocess
import tempfile
import librosa
import soundfile as sf
import shlex
import sys
import importlib
from pathlib import Path

# Text-to-Speech Constants
VOICES = {
    "English (US) - Male": "en-US-ChristopherNeural",
    "English (US) - Female": "en-US-JennyNeural",
    "English (UK) - Male": "en-GB-RyanNeural",
    "English (UK) - Female": "en-GB-SoniaNeural",
    "Spanish (Spain) - Male": "es-ES-AlvaroNeural",
    "Spanish (Mexico) - Female": "es-MX-DaliaNeural",
    "French (France) - Male": "fr-FR-HenriNeural",
    "French (France) - Female": "fr-FR-DeniseNeural",
    "German (Germany) - Male": "de-DE-ConradNeural",
    "German (Germany) - Female": "de-DE-KatjaNeural",
    "Italian (Italy) - Male": "it-IT-DiegoNeural",
    "Italian (Italy) - Female": "it-IT-ElsaNeural",
    "Japanese (Japan) - Male": "ja-JP-KeitaNeural",
    "Japanese (Japan) - Female": "ja-JP-NanamiNeural",
    "Chinese (Mandarin) - Male": "zh-CN-YunxiNeural",
    "Chinese (Mandarin) - Female": "zh-CN-XiaoxiaoNeural",
    "Hindi (India) - Male": "hi-IN-MadhurNeural",
    "Hindi (India) - Female": "hi-IN-SwaraNeural",
    "Arabic (Saudi Arabia) - Male": "ar-SA-HamedNeural",
    "Russian (Russia) - Female": "ru-RU-SvetlanaNeural",
    "Portuguese (Brazil) - Male": "pt-BR-AntonioNeural",
    "Korean (Korea) - Female": "ko-KR-SunHiNeural"
}

# Text-to-Speech Functions
async def text_to_speech(text, voice, rate):
    communicate = edge_tts.Communicate(text, voice, rate=rate)
    audio_data = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_data += chunk["data"]
    return audio_data

def get_binary_file_downloader_html(bin_file, file_label='File'):
    bin_str = base64.b64encode(bin_file).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{file_label}">Download {file_label}</a>'
    return href

# Avatar Generator Functions
@st.cache_resource
def load_model(model_path):
    pipe = StableDiffusionPipeline.from_single_file(
        model_path,
        torch_dtype=torch.float16,
        use_safetensors=True
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
        pipe.enable_attention_slicing()
        pipe.enable_sequential_cpu_offload()
    return pipe

@st.cache_resource
def load_controlnet_model(controlnet_path):
    controlnet = ControlNetModel.from_pretrained(
        controlnet_path,
        torch_dtype=torch.float16
    )
    return controlnet

@st.cache_resource
def load_canny_pipe(model_path, controlnet_path):
    controlnet = load_controlnet_model(controlnet_path)
    pipe = StableDiffusionControlNetPipeline.from_single_file(
        model_path,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        use_safetensors=True
    )
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
        pipe.enable_attention_slicing()
        pipe.enable_sequential_cpu_offload()
    return pipe

@st.cache_resource
def load_face_swap_model(face_swap_path):
    face_swapper = insightface.model_zoo.get_model(face_swap_path)
    face_analyser = insightface.app.FaceAnalysis()
    face_analyser.prepare(ctx_id=0, det_size=(640, 640))
    return face_swapper, face_analyser

# Wav2Lip Functions
class Wav2LipInference:
    def __init__(self, wav2lip_path, wav2lip_gan_path):
        self.wav2lip_path = wav2lip_path
        self.wav2lip_gan_path = wav2lip_gan_path

    def process_media(self, input_path, audio_path, is_image=False, use_gan=False, nosmooth=True):
        output_dir = "results"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "result_voice.mp4")
        
        model_path = self.wav2lip_gan_path if use_gan else self.wav2lip_path
        
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
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False

def get_video_resolution(video_path):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width, height

def resize_video(video_path, max_height=720):
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

# Common Helper Functions
def process_canny_reference(reference_image, low_threshold=100, high_threshold=200):
    canny = CannyDetector()
    reference_np = np.array(reference_image)
    canny_image = canny(reference_np, low_threshold, high_threshold)
    return Image.fromarray(canny_image)

def swap_face(source_img, target_img, face_swap_path):
    face_swapper, face_analyser = load_face_swap_model(face_swap_path)
    source_cv2 = cv2.cvtColor(np.array(source_img), cv2.COLOR_RGB2BGR)
    target_cv2 = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)
    source_face = face_analyser.get(source_cv2)
    target_face = face_analyser.get(target_cv2)
    if len(source_face) == 0 or len(target_face) == 0:
        raise Exception("No face detected in one or both images")
    result = face_swapper.get(target_cv2, target_face[0], source_face[0], paste_back=True)
    return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

def generate_avatar(prompt, negative_prompt, num_images, guidance_scale, steps, height, width, model_path):
    pipe = load_model(model_path)
    images = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_images_per_prompt=num_images,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width
    ).images
    return images

def generate_avatar_with_controlnet(prompt, negative_prompt, reference_image, num_images, guidance_scale, steps, height, width, canny_threshold, model_path, controlnet_path):
    pipe = load_canny_pipe(model_path, controlnet_path)
    canny_image = process_canny_reference(
        reference_image,
        low_threshold=canny_threshold[0],
        high_threshold=canny_threshold[1]
    )
    images = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=canny_image,
        num_images_per_prompt=num_images,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width
    ).images
    return images

def upscale_image(image, scale_factor):
    if scale_factor == 1:
        return image
    original_width, original_height = image.size
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

def main():
    st.set_page_config(
        page_title="AI-Driven Podcast Creation Suite",
        page_icon="🎙️",
        layout="wide"
    )

    # Add main title and description
    st.title("Designing and Developing a Lipsynced and Stable Diffusion Faces AI-Driven Podcast Creation Suite")
    st.markdown("""
    <style>
        .main-title {
            text-align: center;
            padding: 1rem 0;
            color: #1E88E5;
        }
        .sub-header {
            text-align: center;
            color: #666;
            padding-bottom: 2rem;
        }
    </style>
    <div class="sub-header">
        An all-in-one solution for creating AI-powered podcasts with lip-synced avatars and multilingual voice synthesis
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "Text-to-Speech Converter",
        "AI Avatar Generator",
        "Lip-Sync Video Generator",
        "Model Management"
    ])

    # Text-to-Speech Tab
    with tab1:
        st.title("Advanced Multilingual Text-to-Speech Converter")
        text_input = st.text_area("Enter the text you want to convert to speech:", height=150)
        col1, col2 = st.columns(2)
        
        with col1:
            voice_name = st.selectbox("Select a voice:", list(VOICES.keys()))
        with col2:
            rate_option = st.selectbox("Select speech rate:", ["Very Slow", "Slow", "Normal", "Fast", "Very Fast"])

        rate_map = {
            "Very Slow": "-50%",
            "Slow": "-25%",
            "Normal": "+0%",
            "Fast": "+25%",
            "Very Fast": "+50%"
        }

        if st.button("Convert to Speech"):
            if text_input:
                with st.spinner("Converting text to speech..."):
                    voice = VOICES[voice_name]
                    rate = rate_map[rate_option]
                    audio_data = asyncio.run(text_to_speech(text_input, voice, rate))
                    st.audio(audio_data, format="audio/wav")
                    st.success("Text-to-speech conversion completed!")
                    st.markdown(get_binary_file_downloader_html(audio_data, 'audio.wav'), unsafe_allow_html=True)
            else:
                st.warning("Please enter some text to convert.")

    # Avatar Generator Tab
    with tab2:
        st.title("AI Avatar Generator with Face Swap and Pose/Clothing Imitating Feature")
        
        with st.sidebar:
            st.header("Settings")
            
            # New Gender Selection
            gender = st.selectbox(
                "Gender",
                options=[
                    "Male",
                    "Female",
                    "Non-Binary"
                ]
            )
            
            # New Ethnicity Selection
            ethnicity = st.selectbox(
                "Ethnicity",
                options=[
                    "Caucasian",
                    "Asian",
                    "African",
                    "Hispanic/Latino",
                    "Middle Eastern",
                    "South Asian",
                    "East Asian",
                    "Southeast Asian",
                    "Pacific Islander",
                    "Mixed/Multi-ethnic"
                ]
            )
            
            age_range = st.selectbox(
                "Character Age Range",
                options=[
                    "Young Adult (20-30)",
                    "Adult (30-40)",
                    "Middle Age (40-50)",
                    "Mature (50-60)",
                    "Senior (60+)"
                ]
            )

            # Modified preset prompts with safety considerations and additional base prompt
            base_prompt = "A realistic human avatar standing at a distance from the camera, straight face, standing away from the camera, full body view, full body view from top to bottom, without beard and moustache, distance, center-aligned, straight face looking into the camera, facing the camera directly. The background is lively and realistic, focusing on the subject. Full-body view, ensuring the figure is proportional and balanced, with clear and sharp details. Safe for work, fully clothed, professional appearance."

            preset_prompts = {
                "Professional Businessman": f"Full body view, Clear facial features, {ethnicity} {gender}, {age_range} professional wearing conservative formal business suit and tie, confident yet approachable pose, modern corporate background, sharp jawline, well-groomed, studio lighting, professional demeanor. {base_prompt}",
                
                "Corporate Executive": f"Full body view, Clear facial features, {ethnicity} {gender}, {age_range} executive portrait, professional business attire, confident leadership stance, modern office background, commanding presence, well-lit environment, business formal dress code. {base_prompt}",
                
                "Tech Professional": f"Full body view, Clear facial features, {ethnicity} {gender}, {age_range} tech professional, smart casual business attire, modern workspace, friendly approachable expression, startup environment, natural lighting, business casual dress code. {base_prompt}",
                
                "Medical Professional": f"Full body view, Clear facial features, {ethnicity} {gender}, {age_range} medical professional, wearing white coat with professional attire underneath, clinical setting, trustworthy expression, clean medical background, professional lighting. {base_prompt}",
                
                "Academic Professor": f"Full body view, Clear facial features, {ethnicity} {gender}, {age_range} distinguished professor, intellectual appearance, professional academic attire, library or office background, scholarly look, wearing glasses optional, professional setting. {base_prompt}",
                
                "Business Consultant": f"Full body view, Clear facial features, {ethnicity} {gender}, {age_range} consultant, wearing professional business attire, modern office setting, engaging expression, professional pose, clean background. {base_prompt}",
                
                "Creative Professional": f"Full body view, Clear facial features, {ethnicity} {gender}, {age_range} creative professional portrait, smart casual attire, modern creative space, professional studio lighting, expressive yet professional face. {base_prompt}",
                
                "Casual Professional": f"Full body view, Clear facial features, {ethnicity} {gender}, {age_range} natural casual professional portrait, business casual attire, relaxed yet professional pose, friendly expression, outdoor or office lighting, authentic professional look. {base_prompt}"
            }

            source_image = st.file_uploader(
                "Upload Source Face Image",
                type=['jpg', 'jpeg', 'png'],
                key="avatar_source_image"
            )
            
            enable_face_swap = st.checkbox("Enable Face Swap")
            
            style_reference = st.file_uploader(
                "Upload Style Reference Image",
                type=['jpg', 'jpeg', 'png'],
                key="avatar_style_reference"
            )
            
            if style_reference:
                reference_img = Image.open(style_reference)
                st.image(reference_img, caption="Style Reference", use_column_width=True)
                canny_threshold = st.slider(
                    "Edge Detection Sensitivity",
                    min_value=0,
                    max_value=500,
                    value=(100, 200)
                )
            
            prompt_preset = st.selectbox(
                "Prompt Presets",
                options=["Custom Prompt"] + list(preset_prompts.keys())
            )

            if prompt_preset == "Custom Prompt":
                prompt = st.text_area(
                    "Custom Prompt",
                    value=f"Full body view, Clear facial features, {ethnicity} {gender}, {age_range} Professional looking person, looking in front"
                )
            else:
                prompt = st.text_area(
                    "Prompt (Pre-filled)",
                    value=preset_prompts[prompt_preset],
                    height=100
                )
            
            negative_prompt = st.text_area(
                "Negative Prompt",
                value="ugly, deformed, noisy, blurry, low quality, cartoon, anime, illustration, painting, drawing, art, disfigured, mutation, extra limbs, nsfw, nude, naked, suggestive poses, inappropriate content, revealing clothing, excessive skin exposure"
            )
            
            num_images = st.slider("Number of Images", min_value=1, max_value=30, value=1)
            guidance_scale = st.slider("CFG Scale", min_value=1.0, max_value=3.0, value=1.5, step=0.1)
            
            steps = st.selectbox(
                "Number of Steps",
                options=[
                    ("Hyper-SD (4 steps)", 4),
                    ("Lightning (8 steps)", 8),
                    ("Quick (15 steps)", 15),
                    ("Balanced (20 steps)", 20),
                    ("Quality (25 steps)", 25),
                    ("Maximum Quality (30 steps)", 30),
                    ("Ultra Quality (50 steps)", 50)
                ],
                format_func=lambda x: x[0],
                help="Select the number of inference steps. More steps generally means better quality but slower generation."
            )

            aspect_ratio = st.selectbox(
                "Aspect Ratio",
                options=[
                    ("Square (1:1)", (512, 512)),
                    ("Portrait (2:3)", (512, 768)),
                    ("Portrait (3:4)", (512, 683)),
                    ("Portrait (4:5)", (512, 640)),
                    ("Landscape (3:2)", (768, 512)),
                    ("Landscape (4:3)", (683, 512)),
                    ("Landscape (5:4)", (640, 512)),
                    ("Wide (16:9)", (912, 512)),
                    ("Ultrawide (21:9)", (1024, 512))
                ],
                format_func=lambda x: x[0]
            )
            
            upscale_factor = st.selectbox(
                "Image Upscaling",
                options=[
                    ("No Upscaling", 1),
                    ("1.25X Upscale", 1.25),
                    ("1.5X Upscale", 1.5),
                    ("1.75X Upscale", 1.75),
                    ("2X Upscale", 2),
                    ("2.5X Upscale", 2.5),
                    ("3X Upscale", 3),
                    ("4X Upscale", 4)
                ],
                format_func=lambda x: x[0]
            )

            generate_button = st.button("Generate Avatar")

        if generate_button:
            with st.spinner("Generating your avatar..."):
                try:
                    model_path = st.session_state.get("model_path", "models/Merged_JuggernautXL_Realistic_Vision.safetensors")
                    controlnet_path = st.session_state.get("controlnet_path", "lllyasviel/sd-controlnet-canny")
                    face_swap_path = st.session_state.get("face_swap_path", "models/inswapper_128.onnx")

                    if style_reference is not None:
                        images = generate_avatar_with_controlnet(
                            prompt,
                            negative_prompt,
                            Image.open(style_reference),
                            num_images,
                            guidance_scale,
                            steps[1],
                            height=aspect_ratio[1][1],
                            width=aspect_ratio[1][0],
                            canny_threshold=canny_threshold,
                            model_path=model_path,
                            controlnet_path=controlnet_path
                        )
                    else:
                        images = generate_avatar(
                            prompt,
                            negative_prompt,
                            num_images,
                            guidance_scale,
                            steps[1],
                            height=aspect_ratio[1][1],
                            width=aspect_ratio[1][0],
                            model_path=model_path
                        )
                    
                    cols = st.columns(2)
                    for idx, image in enumerate(images):
                        upscaled_image = upscale_image(image, upscale_factor[1])
                        
                        if enable_face_swap and source_image is not None:
                            try:
                                upscaled_image = swap_face(Image.open(source_image), upscaled_image, face_swap_path)
                                st.success(f"Face swap successful for image {idx + 1}")
                            except Exception as e:
                                st.warning(f"Face swap failed for image {idx + 1}: {str(e)}")
                        
                        with cols[idx % 2]:
                            st.image(upscaled_image, use_column_width=True)
                            
                            img_path = f"avatar_{idx}.png"
                            upscaled_image.save(img_path)
                            
                            with open(img_path, "rb") as file:
                                st.download_button(
                                    label="Download Image",
                                    data=file,
                                    file_name=img_path,
                                    mime="image/png"
                                )
                            
                            os.remove(img_path)
                            
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

    # Wav2Lip Tab
    with tab3:
        st.title("Lip-Synced Video generator")
        st.write("Generate lip-synced videos from either images or videos")

        # Initialize Wav2Lip
        wav2lip_path = st.session_state.get("wav2lip_path", "checkpoints/wav2lip.pth")
        wav2lip_gan_path = st.session_state.get("wav2lip_gan_path", "checkpoints/wav2lip_gan.pth")
        wav2lip = Wav2LipInference(wav2lip_path, wav2lip_gan_path)

        # Input type selection
        input_type = st.radio("Select Input Type", ["Image", "Video"])

        # File uploaders
        if input_type == "Image":
            input_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="wav2lip_image")
        else:
            input_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"], key="wav2lip_video")

        audio_file = st.file_uploader("Upload Audio", type=["wav", "mp3"], key="wav2lip_audio")

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

    # Model Management Tab
    with tab4:
        st.title("Model Management")
        st.write("Specify the paths for the required models.")

        # Model Path Inputs
        st.subheader("Model Paths")
        model_path = st.text_input(
            "Path to Merged_JuggernautXL_Realistic_Vision.safetensors",
            value="models/Merged_JuggernautXL_Realistic_Vision.safetensors"
        )
        controlnet_path = st.text_input(
            "Path to ControlNet Model",
            value="lllyasviel/sd-controlnet-canny"
        )
        face_swap_path = st.text_input(
            "Path to inswapper_128.onnx",
            value="models/inswapper_128.onnx"
        )
        wav2lip_path = st.text_input(
            "Path to wav2lip.pth",
            value="checkpoints/wav2lip.pth"
        )
        wav2lip_gan_path = st.text_input(
            "Path to wav2lip_gan.pth",
            value="checkpoints/wav2lip_gan.pth"
        )

        # Save paths to session state
        if st.button("Save Model Paths"):
            st.session_state["model_path"] = model_path
            st.session_state["controlnet_path"] = controlnet_path
            st.session_state["face_swap_path"] = face_swap_path
            st.session_state["wav2lip_path"] = wav2lip_path
            st.session_state["wav2lip_gan_path"] = wav2lip_gan_path
            st.success("Model paths saved successfully!")

if __name__ == "__main__":
    main()