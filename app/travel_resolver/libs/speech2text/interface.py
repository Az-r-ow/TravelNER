import gradio as gr
from transformers import pipeline
import numpy as np

transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")

def transcribe(audio):
    """
    Transcribe audio into text
    """
    sr, y = audio
    
    # Convert to mono if stereo
    if y.ndim > 1:
        y = y.mean(axis=1)
        
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))

    return transcriber({"sampling_rate": sr, "raw": y})["text"]  
    
css = """
<style>
    .gradio-container {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100vh;
    }
</style>
"""

with gr.Blocks() as interface:
    gr.HTML(css)
    gr.Interface(
        fn=transcribe,
        inputs=gr.Audio(),  # Utilisation par défaut de l'entrée audio (microphone)
        outputs="text"
    )

interface.launch()