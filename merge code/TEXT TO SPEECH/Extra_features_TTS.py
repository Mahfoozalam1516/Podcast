import streamlit as st
import pyttsx3
import os
import tempfile
import base64
from io import BytesIO

def get_available_voices():
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    return {voice.name: voice.id for voice in voices}

def text_to_speech(text, voice_id, rate, volume):
    engine = pyttsx3.init()
    engine.setProperty('voice', voice_id)
    engine.setProperty('rate', rate)
    engine.setProperty('volume', volume)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        engine.save_to_file(text, fp.name)
        engine.runAndWait()
        return fp.name

def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:audio/mpeg;base64,{bin_str}" download="audio.mp3">Download {file_label}</a>'
    return href

def main():
    st.title("Multilingual Text-to-Speech Converter")

    # Text input
    text = st.text_area("Enter the text you want to convert to speech:")

    # Get available voices
    voices = get_available_voices()
    
    # Voice selection
    selected_voice = st.selectbox("Select Voice", list(voices.keys()))

    # Rate (speed) selection
    rate = st.slider("Speech Rate", 50, 300, 150, 10)

    # Volume selection
    volume = st.slider("Volume", 0.0, 1.0, 0.5, 0.1)

    if st.button("Convert to Speech"):
        if text:
            try:
                audio_file = text_to_speech(text, voices[selected_voice], rate, volume)
                
                # Read the audio file
                with open(audio_file, 'rb') as f:
                    audio_bytes = f.read()
                
                # Play the audio
                st.audio(audio_bytes, format='audio/mp3')
                
                # Offer download option
                st.markdown(get_binary_file_downloader_html(audio_file, 'Audio'), unsafe_allow_html=True)
                
                # Clean up the temporary file
                os.unlink(audio_file)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please enter some text to convert.")

if __name__ == "__main__":
    main()