import streamlit as st
from gtts import gTTS
import os
import tempfile
import base64
from pydub import AudioSegment
import io

def safe_adjust_pitch(audio_segment, semitones):
    try:
        semitones = max(min(semitones, 6), -6)
        new_sample_rate = int(audio_segment.frame_rate * (2.0 ** (semitones / 12.0)))
        return audio_segment._spawn(audio_segment.raw_data, overrides={'frame_rate': new_sample_rate})
    except Exception as e:
        st.error(f"Error in pitch adjustment: {str(e)}")
        return audio_segment

def safe_adjust_speed(audio_segment, speed):
    try:
        speed = max(min(speed, 1.5), 0.75)
        return audio_segment.speedup(playback_speed=speed)
    except Exception as e:
        st.error(f"Error in speed adjustment: {str(e)}")
        return audio_segment

def text_to_speech(text, language, voice_profile, custom_pitch, custom_speed):
    try:
        tts = gTTS(text=text, lang=language, slow=False)
        
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        
        sound = AudioSegment.from_mp3(audio_bytes)
        
        st.info(f"Initial audio duration: {len(sound)/1000:.2f} seconds")
        
        if voice_profile != "Default":
            profile = voice_profiles[voice_profile]
            sound = safe_adjust_pitch(sound, profile["pitch"] + custom_pitch)
            sound = safe_adjust_speed(sound, profile["speed"] * custom_speed)
        else:
            sound = safe_adjust_pitch(sound, custom_pitch)
            sound = safe_adjust_speed(sound, custom_speed)
        
        st.info(f"Final audio duration: {len(sound)/1000:.2f} seconds")
        
        output_bytes = io.BytesIO()
        sound.export(output_bytes, format="mp3")
        output_bytes.seek(0)
        
        return output_bytes
    except Exception as e:
        st.error(f"Error in text_to_speech function: {str(e)}")
        return None

def get_binary_file_downloader_html(bin_file, file_label='File'):
    bin_str = base64.b64encode(bin_file.getvalue()).decode()
    href = f'<a href="data:audio/mp3;base64,{bin_str}" download="audio.mp3">Download {file_label}</a>'
    return href

voice_profiles = {
    "Default": {"pitch": 0, "speed": 1.0},
    "Child": {"pitch": 2, "speed": 1.05},
    "Old Man": {"pitch": -2, "speed": 0.95},
    "Robot": {"pitch": -1, "speed": 1.0},
    "Chipmunk": {"pitch": 3, "speed": 1.1},
    "Giant": {"pitch": -3, "speed": 0.9},
}

def main():
    st.title("Multilingual Text-to-Speech Converter")

    text = st.text_area("Enter the text you want to convert to speech:", height=150)

    col1, col2 = st.columns(2)

    with col1:
        languages = {
            'English': 'en', 'Spanish': 'es', 'French': 'fr', 'German': 'de',
            'Italian': 'it', 'Portuguese': 'pt', 'Hindi': 'hi', 'Japanese': 'ja',
            'Korean': 'ko', 'Chinese': 'zh-CN'
        }
        selected_language = st.selectbox("Select Language", list(languages.keys()))

    with col2:
        selected_voice = st.selectbox("Select Voice Profile", list(voice_profiles.keys()))

    st.subheader("Fine-tune Voice (Optional)")
    custom_pitch = st.slider("Adjust Pitch", min_value=-6, max_value=6, value=0, step=1, 
                      help="Adjust pitch in semitones. Negative values lower the pitch, positive values raise it.")
    
    speed_options = {'Slower': 0.9, 'Normal': 1.0, 'Faster': 1.1}
    custom_speed = st.select_slider("Adjust Speed", options=list(speed_options.keys()), value='Normal')

    if st.button("Convert to Speech"):
        if text:
            with st.spinner("Converting text to speech..."):
                audio_bytes = text_to_speech(text, languages[selected_language], 
                                             voice_profile=selected_voice,
                                             custom_pitch=custom_pitch, 
                                             custom_speed=speed_options[custom_speed])
            
            if audio_bytes:
                st.audio(audio_bytes, format='audio/mp3')
                st.markdown(get_binary_file_downloader_html(audio_bytes, 'Audio'), unsafe_allow_html=True)
                st.success("Conversion complete! You can play the audio above or download it.")
            else:
                st.error("Failed to generate audio. Please try again with different settings.")
        else:
            st.warning("Please enter some text to convert.")

if __name__ == "__main__":
    main()