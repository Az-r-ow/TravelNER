import gradio as gr
from transformers import pipeline
import numpy as np

from travel_resolver.libs.pathfinder.CSVTravelGraph import CSVTravelGraph
from travel_resolver.libs.pathfinder.graph import Graph

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
    
def getDijkstraResult(depart, destination):
    """
    Args: 
        depart (str): station name
        destination (str): station name
    Generate dijkstraGraph and find the shortest way for the destination
    Returns:
        (str): Time of the shortest travel found
    """
    dijkstraGraphData = CSVTravelGraph("../data/sncf/timetables.csv", "Dijkstra")
    graph = Graph(dijkstraGraphData.data)
    distances = graph.RunDijkstra(depart)
    if(destination in distances): return str(distances[destination]) + " minutes"
    return "Temps non trouvé"

def submit(audio, departV, destinationV):
    promptValue = transcribe(audio)
    return { 
        prompt: gr.TextArea(label="Prompt", value=promptValue, visible=True),
        depart: gr.Textbox(label="Départ", visible=True),
        destination: gr.Textbox(label="Destination", visible=True),
        timeDijkstra: gr.Textbox(label="Temps trouvé", value=getDijkstraResult(departV, destinationV))
    }
def clear(): 
    return {
        timeDijkstra:gr.HTML("<p>Aucun prompt renseigné</p>"),
        timeAStar:gr.HTML("<p>Aucun prompt renseigné</p>"),
        prompt: gr.TextArea(label="Prompt", visible=False),
        depart:gr.Textbox(label="Départ", visible=False, value="Gare de Amiens"),
        destination: gr.Textbox(label="Destination", visible=False, value="Gare de Jeumont")
    }
                
with gr.Blocks() as interface:
    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            audio=gr.Audio(sources="microphone")
            prompt= gr.TextArea(label="Prompt", visible=False)
            depart= gr.Textbox(label="Départ", visible=False, value="Gare de Amiens")
            destination= gr.Textbox(label="Destination", visible=False, value="Gare de Jeumont")
            submitButton = gr.Button("Rechercher")
        with gr.Column(scale=2, min_width=300):
            with gr.Tab("Dijkstra"):
                timeDijkstra=gr.HTML("<p>Aucun prompt renseigné</p>")
            with gr.Tab("AStar"):
                timeAStar=gr.HTML("<p>Aucun prompt renseigné</p>")
    submitButton.click(
        submit,
        [audio, depart, destination],
        [prompt, depart, destination, timeDijkstra],
    )
    audio.clear(
        clear,
        [],
        [timeDijkstra, timeAStar, prompt, depart, destination]
    )
interface.launch()

