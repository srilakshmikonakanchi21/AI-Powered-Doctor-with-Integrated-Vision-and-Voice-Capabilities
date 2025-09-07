import os
import gradio as gr
from dotenv import load_dotenv
import logging

from think import encode_image, analyze_image_with_query
from user_voice import transcribe_with_groq
from assistant_voice import text_to_speech_with_gtts_old

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# ✅ Debug: check if keys are loaded
print("OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY"))
print("GROQ_API_KEY:", os.getenv("GROQ_API_KEY"))

# Check for required API key
if not os.getenv("GROQ_API_KEY") or not os.getenv("OPENAI_API_KEY"):
    raise ValueError("API keys missing! Please check your .env file.")

MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

def process_input(audio_file_path, image_file_path):
    try:
        logging.info("Processing input...")
        speech_to_text_output = ""
        assistant_response = "Image not provided."

        # Process audio if provided
        if audio_file_path:
            logging.info("Transcribing audio...")
            speech_to_text_output = transcribe_with_groq("whisper-large-v3", audio_file_path)
        else:
            speech_to_text_output = "No audio input provided."
            
        # Process image if provided
        if image_file_path:
            logging.info("Encoding image...")
            encoded_image = encode_image(image_file_path)
            logging.info("Analyzing image with AI...")
            assistant_response = analyze_image_with_query(speech_to_text_output, encoded_image, MODEL)

        # Convert assistant response to speech
        logging.info("Converting text to speech...")
        output_audio_path = "assistant_response.mp3"
        text_to_speech_with_gtts_old(assistant_response, output_audio_path)

        logging.info("Processing completed successfully.")
        return speech_to_text_output, assistant_response, output_audio_path

    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        logging.error(error_message)
        return "Error processing input", error_message, None


# ✅ Gradio UI
with gr.Blocks(title="AI Doctor with Vision and Voice") as demo:
    gr.Markdown("## AI Doctor with Vision and Voice\nUpload a medical image and speak your question. The AI doctor will analyze the image and respond both in text and voice.")

    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(sources=["microphone"], type="filepath", label="Speak your question")
            image_input = gr.Image(type="filepath", label="Upload medical image")

        with gr.Column():
            question_output = gr.Textbox(label="Your Question (Speech to Text)")
            doctor_output = gr.Textbox(label="Doctor's Response")
            voice_output = gr.Audio(label="Doctor's Voice Response", type="filepath")

    submit_btn = gr.Button("Submit")

    submit_btn.click(
        fn=process_input,
        inputs=[audio_input, image_input],
        outputs=[question_output, doctor_output, voice_output]
    )

if __name__ == "__main__":
    demo.launch(debug=True)
