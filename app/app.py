import gradio as gr
from transformers import pipeline
import numpy as np
import pandas as pd
import torch

from travel_resolver.libs.pathfinder.CSVTravelGraph import CSVTravelGraph
from travel_resolver.libs.pathfinder.graph import Graph

transcriber = pipeline(
    "automatic-speech-recognition", model="openai/whisper-base.en", device="cpu"
)


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


def getCSVTravelGraph():
    """
    Generate Graph with the csv dataset
    Returns:
        (Graph): Graph
    """
    graphData = CSVTravelGraph("./data/sncf/timetables.csv")
    return Graph(graphData.data)


def getDijkstraResult(depart, destination):
    """
    Args:
        depart (str): station name
        destination (str): station name
    Generate dijkstraGraph and find the shortest way for the destination
    Returns:
        (str): Time of the shortest travel found
    """
    graph = getCSVTravelGraph()
    path, cost = graph.RunDijkstraBetweenTwoNodes(depart, destination)
    if destination in cost:
        return [path, str(cost[destination]) + " minutes"]
    return [[], "Temps non trouvé"]


def getAStarResult(depart, destination):
    """
    Args:
        depart (str): station name
        destination (str): station name
    Generate AStarGraph and find the shortest way for the destination
    Returns:
        (str): Time of the shortest travel found
    """
    graph = getCSVTravelGraph()
    heuristic = graph.RunDijkstra(destination)
    path, cost = graph.RunAStar(depart, destination, heuristic)
    if destination in cost:
        return [path, str(cost[destination]) + " minutes"]
    return [[], "Temps non trouvé"]


def getStationsByCityName(city: str):
    data = pd.read_csv("../data/sncf/gares_info.csv", sep=",")
    stations = tuple(data[data["Commune"] == city]["Nom de la gare"])
    return stations


def handle_audio(audio):
    promptAudio = transcribe(audio)
    return (
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(value=promptAudio),
        gr.update(value="PARIS"),
        gr.update(value="MONTPELLIER"),
    )


def handle_file(file):
    if file is not None:
        with open(file.name, "r") as f:
            file_content = f.read()
    else:
        file_content = "Aucun fichier uploadé."
    return (
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(value=file_content),
        gr.update(value="PARIS"),
        gr.update(value="MONTPELLIER"),
    )


def handle_back():
    return gr.update(visible=False), gr.update(visible=True)


def handleCityChange(city):
    stations = getStationsByCityName(city)
    return gr.update(choices=stations, value=stations[0], interactive=True)


def handleStationChange(departureStation, destinationStation):
    if departureStation and destinationStation:
        dijkstraPath, dijkstraCost = getDijkstraResult(
            departureStation, destinationStation
        )
        dijkstraPathFormatted = "\n".join(
            [f"{i + 1}. {elem}" for i, elem in enumerate(dijkstraPath)]
        )
        AStarPath, AStarCost = getAStarResult(departureStation, destinationStation)
        AStarPathFormatted = "\n".join(
            [f"{i + 1}. {elem}" for i, elem in enumerate(AStarPath)]
        )
        print(dijkstraPathFormatted)
        return (
            gr.update(value=dijkstraCost),
            gr.update(value=dijkstraPathFormatted, lines=len(dijkstraPath)),
            gr.update(value=AStarCost),
            gr.update(value=AStarPathFormatted, lines=len(AStarPath)),
        )
    return (
        gr.HTML("<p>Aucun prompt renseigné</p>"),
        gr.update(value=""),
        gr.HTML("<p>Aucun prompt renseigné</p>"),
        gr.update(value=""),
    )


with gr.Blocks(css="#back-button {width: fit-content}") as interface:
    with gr.Column() as promptChooser:
        with gr.Row():
            audio = gr.Audio(label="Fichier audio")
            file = gr.File(
                label="Fichier texte", file_types=["text"], file_count="single"
            )
    with gr.Column(visible=False) as content:
        backButton = gr.Button("← Back", elem_id="back-button")
        with gr.Row():
            with gr.Column(scale=1, min_width=300) as parameters:
                prompt = gr.Textbox(label="Prompt")
                departureCity = gr.Textbox(label="Ville de départ")
                destinationCity = gr.Textbox(label="Ville de de destination")
            with gr.Column(scale=2, min_width=300) as result:
                with gr.Row():
                    departureStation = gr.Dropdown(label="Gare de départ")
                    destinationStation = gr.Dropdown(label="Gare d'arrivée")
                with gr.Tab("Dijkstra"):
                    timeDijkstra = gr.HTML("<p>Aucun prompt renseigné</p>")
                    dijkstraPath = gr.Textbox(label="Chemin emprunté")

                with gr.Tab("AStar"):
                    timeAStar = gr.HTML("<p>Aucun prompt renseigné</p>")
                    AstarPath = gr.Textbox(label="Chemin emprunté")
    audio.change(
        handle_audio,
        inputs=[audio],
        outputs=[
            content,
            promptChooser,
            prompt,
            departureCity,
            destinationCity,
        ],  # On rend la section "content" visible
    )
    file.upload(
        handle_file,
        inputs=[file],
        outputs=[
            content,
            promptChooser,
            prompt,
            departureCity,
            destinationCity,
        ],  # On rend la section "content" visible
    )
    backButton.click(handle_back, inputs=[], outputs=[content, promptChooser])
    departureCity.change(
        handleCityChange, inputs=[departureCity], outputs=[departureStation]
    )
    destinationCity.change(
        handleCityChange, inputs=[destinationCity], outputs=[destinationStation]
    )
    departureStation.change(
        handleStationChange,
        inputs=[departureStation, destinationStation],
        outputs=[timeDijkstra, dijkstraPath, timeAStar, AstarPath],
    )
    destinationStation.change(
        handleStationChange,
        inputs=[departureStation, destinationStation],
        outputs=[timeDijkstra, dijkstraPath, timeAStar, AstarPath],
    )
interface.launch()
