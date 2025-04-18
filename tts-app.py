import gradio as gr
from TTS.api import TTS
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

#Loading the Coqui TTS Model
model_name = TTS.list_models()[0]
tts = TTS(model_name)

#Defining Voice Selection
available_speakers = [
    "Daisy Studious", "Sofia Hellen", "Asya Anara",
    "Eugenio Mataracı", "Viktor Menelaos", "Damien Black"      
]

#Defining Localization Options
available_languages = ["US English", "Spanish (LatAm)"]

#Defining Variables to Hold Selected Voice and Localization
selected_speaker = available_speakers[0]
selected_language = available_languages[0]

# Create the output directory if it doesn't exist
os.makedirs("output", exist_ok=True)
last_generated_audio = None
last_generated_text = ""

def trim_text(text, max_length=30):
    """
    Trim the text to a maximum length and add ellipsis if it exceeds the limit.
    """
    return text[:max_length] + "..." if len(text) > max_length else text

# Main Speech Synthesis Function
def generate_speech_with_timestamps(text, speaker, language):
    global last_generated_audio, last_generated_text
    output_path = "output/generated_speech.wav"
    start_time = time.time()

    
    #Generating the speech and save it to a WAV file
    tts.tts_to_file(
        text=text,
        speaker=speaker,
        language='en' if language == "US English" else 'es',
        file_path=output_path
    )

    end_time = time.time()
    duration = round(end_time - start_time, 2)
    last_generated_audio = output_path
    last_generated_text = text

    # Calculating the length of the generated speech
    samplerate, data = wavfile.read(output_path)
    speech_length = len(data) / samplerate

    return output_path, len(text.split()), speaker, language, round(speech_length, 2), duration

# Waveform Function
def generate_waveform():
    global last_generated_audio, last_generated_text

    # Check if a valid audio file exists
    if not last_generated_audio or not os.path.exists(last_generated_audio):
        return None, "No valid audio file found to generate waveform."

    # Read Audio File and Create Time Axis
    samplerate, data = wavfile.read(last_generated_audio)
    time_axis = np.linspace(0, len(data) / samplerate, num=len(data))

    # Plotting the Waveform 
    fig, ax = plt.subplots(figsize=(8, 4), facecolor='#1E1E1E')  # Dark background

    # Plotting the Waveform 
    ax.plot(time_axis, data, color='cyan', alpha=0.8, linewidth=1.2)

    ax.set_facecolor('#2E2E2E')  
    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)  
    ax.spines['bottom'].set_color('white')  
    ax.spines['left'].set_color('white')  
    ax.tick_params(axis='x', colors='white')  
    ax.tick_params(axis='y', colors='white')  
    ax.set_xlabel("Time (seconds)", color='white')  
    ax.set_ylabel("Amplitude", color='white')  

    
    trimmed_text = trim_text(last_generated_text)
    ax.set_title(f"Waveform for text input: '{trimmed_text}'", color='white', fontsize=14)


    waveform_image_path = "output/waveform.png"
    plt.savefig(waveform_image_path, transparent=True)
    plt.close()

    return waveform_image_path, "Waveform generated successfully!"

# Button Click Event Handler
def generate_speech(text, speaker, language):
    if not text:
        return None, "Please enter some text to generate speech.", "", gr.update(interactive=False)

    audio_path, word_count, speaker_name, lang, speech_length, duration = generate_speech_with_timestamps(text, speaker, language)

    # Format the text box content
    data_info = f"Word Count: {word_count}\nVoice: {speaker_name}\nLocalization: {lang}\nLength of Speech: {speech_length} seconds\nGeneration Duration: {duration} seconds"

    return audio_path, data_info, "Speech generation successful!", gr.update(interactive=True)

# Gradio Interface Setup
def setup_interface():
    with gr.Blocks() as app:

        gr.Markdown("# 🗣️ Text-to-Speech GenAI")
        gr.Markdown("Convert text to speech with support for different languages and speakers.")

        with gr.Row():
            with gr.Column():
                text_input = gr.Textbox(label="Enter Text", placeholder="Type your text here...", lines=3)

                with gr.Row():
                    speaker_dropdown = gr.Dropdown(choices=available_speakers, value=selected_speaker, label="Select Voice")
                    language_radio = gr.Radio(choices=available_languages, value=selected_language, label="Select Localization")


            with gr.Column():
                data_info_display = gr.Textbox(label="Data Info", interactive=False, lines=5)
                status_message = gr.Textbox(label="Status", interactive=False)

        with gr.Row():
            with gr.Column():
                audio_output = gr.Audio(label="Generated Speech", interactive=False)
                generate_button = gr.Button("Generate Speech")

            with gr.Column():
                waveform_output = gr.Image(label="Waveform")
                generate_waveform_button = gr.Button("Generate Waveform", interactive=False)

        generate_button.click(
            generate_speech, 
            inputs=[text_input, speaker_dropdown, language_radio], 
            outputs=[audio_output, data_info_display, status_message, generate_waveform_button]
        )

        generate_waveform_button.click(
            generate_waveform, 
            outputs=[waveform_output, status_message]
        )

    return app

if __name__ == "__main__":
    app = setup_interface()
    app.launch(share=True)