import streamlit as st
from diffusers import StableDiffusionPipeline, ControlNetModel, StableDiffusionControlNetPipeline
from controlnet_aux import CannyDetector
import torch
from PIL import Image
import os
from diffusers.schedulers import DPMSolverMultistepScheduler
import insightface
import cv2
import numpy as np

# Set page config
st.set_page_config(
    page_title="AI Avatar Generator",
    page_icon="👤",
    layout="wide"
)

# Set title
st.title("AI Avatar Generator using Realistic Vision v6.0")



@st.cache_resource
def load_model():
    model_path = "models/realisticVisionV60B1_v51HyperVAE.safetensors"
    
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
def load_controlnet_model():
    """Load ControlNet model for pose and edge detection"""
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny",
        torch_dtype=torch.float16
    )
    return controlnet

@st.cache_resource
def load_canny_pipe():
    """Load the complete pipeline with ControlNet"""
    model_path = "models/realisticVisionV60B1_v51HyperVAE.safetensors"
    controlnet = load_controlnet_model()
    
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
def load_face_swap_model():
    face_swapper = insightface.model_zoo.get_model('models/inswapper_128.onnx')
    face_analyser = insightface.app.FaceAnalysis()
    face_analyser.prepare(ctx_id=0, det_size=(640, 640))
    return face_swapper, face_analyser

def process_canny_reference(reference_image, low_threshold=100, high_threshold=200):
    """Process reference image with Canny edge detection"""
    canny = CannyDetector()
    reference_np = np.array(reference_image)
    canny_image = canny(reference_np, low_threshold, high_threshold)
    return Image.fromarray(canny_image)

def swap_face(source_img, target_img):
    face_swapper, face_analyser = load_face_swap_model()
    
    source_cv2 = cv2.cvtColor(np.array(source_img), cv2.COLOR_RGB2BGR)
    target_cv2 = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)
    
    source_face = face_analyser.get(source_cv2)
    target_face = face_analyser.get(target_cv2)
    
    if len(source_face) == 0 or len(target_face) == 0:
        raise Exception("No face detected in one or both images")
    
    result = face_swapper.get(target_cv2, target_face[0], source_face[0], paste_back=True)
    return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

def generate_avatar(prompt, negative_prompt, num_images, guidance_scale, steps, height, width):
    pipe = load_model()
    
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

def generate_avatar_with_controlnet(prompt, negative_prompt, reference_image, num_images, guidance_scale, steps, height, width, canny_threshold):
    """Generate avatar with ControlNet guidance"""
    pipe = load_canny_pipe()
    
    # Process reference image with Canny
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
    """Upscale the generated image by a given scale factor"""
    if scale_factor == 1:
        return image
    
    original_width, original_height = image.size
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)
    
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

def main():
    # Sidebar controls
    with st.sidebar:
        st.header("Generation Settings")

        age_range = st.selectbox(
            "Character Age Range",
            options=[
                "Young Adult (20-30)",
                "Adult (30-40)",
                "Middle Age (40-50)",
                "Mature (50-60)",
                "Senior (60+)"
            ],
            help="Select the approximate age range for your generated character"
        )

        # Preset prompts dictionary
        preset_prompts = {
    "Professional Businessman": f"Clear facial features, looking in front, {age_range} professional businessman wearing formal suit, confident pose, corporate background, sharp jawline, well-groomed, studio lighting",
    "Corporate Executive": f"Clear facial features, looking in front, {age_range} executive portrait, professional attire, confident stance, modern office background, leadership presence, well-lit",
    "Tech Professional": f"Clear facial features, looking in front, {age_range} tech professional, casual business attire, modern workspace, friendly expression, startup environment, natural lighting",
    "Medical Professional": f"Clear facial features, looking in front, {age_range} medical professional, wearing white coat, professional setting, trustworthy expression, clean background, professional lighting",
    "Academic Professor": f"Clear facial features, looking in front, {age_range} distinguished professor, intellectual appearance, professional attire, library or office background, scholarly look",
    "Fashion Model": f"Clear facial features, looking in front, {age_range} fashion model portrait, high-end clothing, studio lighting, professional pose, magazine style",
    "Creative Artist": f"Clear facial features, looking in front, {age_range} artistic portrait, creative professional, stylish appearance, studio lighting, expressive face, modern creative space",
    "Casual Portrait": f"Clear facial features, looking in front, {age_range} natural casual portrait, relaxed pose, friendly expression, outdoor lighting, authentic look"
}
        
        source_image = st.file_uploader(
            "Upload Source Face Image",
            type=['jpg', 'jpeg', 'png'],
            help="Upload an image containing the face you want to swap onto the generated avatars"
        )
        
        enable_face_swap = st.checkbox(
            "Enable Face Swap",
            help="Check this to enable face swapping with your uploaded image"
        )
        
        st.header("Pose and Style Reference")
        style_reference = st.file_uploader(
            "Upload Style Reference Image",
            type=['jpg', 'jpeg', 'png'],
            help="Upload an image containing the pose and style you want to mimic"
        )
        
        if style_reference:
            reference_img = Image.open(style_reference)
            st.image(reference_img, caption="Style Reference", use_column_width=True)
            
            canny_threshold = st.slider(
                "Edge Detection Sensitivity",
                min_value=0,
                max_value=500,
                value=(100, 200),
                help="Adjust the sensitivity of edge detection"
            )
        
        # Prompt selection
        prompt_preset = st.selectbox(
            "Prompt Presets",
            options=["Custom Prompt"] + list(preset_prompts.keys()),
            help="Select a preset prompt or choose 'Custom Prompt' to write your own"
        )

        if prompt_preset == "Custom Prompt":
            prompt = st.text_area(
                "Custom Prompt",
                value=f"Clear facial features, {age_range} Professional looking person, looking in front",
                help="Describe the avatar you want to generate"
            )
        else:
            prompt = st.text_area(
                "Prompt (Pre-filled)",
                value=preset_prompts[prompt_preset],
                help="You can modify the preset prompt if needed",
                height=100
            )
        
        negative_prompt = st.text_area(
            "Negative Prompt",
            value="ugly, deformed, noisy, blurry, low quality, cartoon, anime, illustration, painting, drawing, art, disfigured, mutation, extra limbs",
            help="Describe what you don't want in the image"
        )
        
        num_images = st.slider(
            "Number of Images",
            min_value=1,
            max_value=30,
            value=1
        )
        
        guidance_scale = st.slider(
            "CFG Scale",
            min_value=1.0,
            max_value=3.0,
            value=1.5,
            step=0.1
        )
        
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

        if steps[1] > 30:
            st.warning("Higher step counts will significantly increase generation time.")
        
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
            format_func=lambda x: x[0],
            help="Select the aspect ratio for your generated images"
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
            format_func=lambda x: x[0],
            help="Select the upscaling factor for your generated images"
        )

        if upscale_factor[1] > 2:
            st.warning("High upscaling factors may require more processing time and memory.")
        
        generate_button = st.button("Generate Avatar")
    
    # Main content area
    if generate_button:
        with st.spinner("Generating your avatar..."):
            try:
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
                        canny_threshold=canny_threshold
                    )
                else:
                    images = generate_avatar(
                        prompt,
                        negative_prompt,
                        num_images,
                        guidance_scale,
                        steps[1],
                        height=aspect_ratio[1][1],
                        width=aspect_ratio[1][0]
                    )
                
                cols = st.columns(2)
                for idx, image in enumerate(images):
                    upscaled_image = upscale_image(image, upscale_factor[1])
                    
                    if enable_face_swap and source_image is not None:
                        try:
                            source_img = Image.open(source_image)
                            source_cv2 = cv2.cvtColor(np.array(source_img), cv2.COLOR_RGB2BGR)
                            target_cv2 = cv2.cvtColor(np.array(upscaled_image), cv2.COLOR_RGB2BGR)
                            
                            face_swapper, face_analyser = load_face_swap_model()
                            source_face = face_analyser.get(source_cv2)
                            target_face = face_analyser.get(target_cv2)
                            
                            if len(source_face) == 0:
                                st.warning(f"No face detected in source image")
                            elif len(target_face) == 0:
                                st.warning(f"No face detected in generated image {idx + 1}")
                            else:
                                upscaled_image = swap_face(source_img, upscaled_image)
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

if __name__ == "__main__":
    main()
