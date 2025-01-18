import streamlit as st
import asyncio
import edge_tts
import io
import base64

# Set page config at the very beginning
st.set_page_config(page_title="Advanced Multilingual Text-to-Speech Converter", page_icon="🎤")

# Extended dictionary of voices with language codes
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
            
            # Create a download button for the audio file
            st.markdown(get_binary_file_downloader_html(audio_data, 'audio.wav'), unsafe_allow_html=True)
    else:
        st.warning("Please enter some text to convert.")

st.sidebar.header("About")
st.sidebar.info("This app uses edge-tts to convert text to speech in multiple languages with adjustable speech rates.")
st.sidebar.header("Supported Languages")
languages = set(lang.split(' - ')[0].split(' (')[0] for lang in VOICES.keys())
st.sidebar.write("\n".join(f"- {lang}" for lang in sorted(languages)))

st.sidebar.header("Features")
st.sidebar.markdown("""
- Multiple languages and voices
- Adjustable speech rate
- Male and female voices for most languages
- Download option for generated audio
""")