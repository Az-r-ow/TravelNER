import gradio as gr
from transformers import pipeline
import numpy as np
import pandas as pd
from travel_resolver.libs.nlp.ner.models import BiLSTM_NER, LSTM_NER, CamemBERT_NER

# import torch
from travel_resolver.libs.nlp.ner.data_processing import process_sentence
from travel_resolver.libs.pathfinder.CSVTravelGraph import CSVTravelGraph
from travel_resolver.libs.pathfinder.graph import Graph
import time

transcriber = pipeline(
    "automatic-speech-recognition", model="openai/whisper-base", device="cpu"
)

models = {"LSTM": LSTM_NER(), "BiLSTM": BiLSTM_NER(), "CamemBERT": CamemBERT_NER()}

entities_label_mapping = {1: "LOC-DEP", 2: "LOC-ARR"}

with gr.Blocks(css="#back-button {width: fit-content}") as demo:
    with gr.Column() as promptChooser:
        with gr.Row():
            audio = gr.Audio(label="Fichier audio")
            file = gr.File(
                label="Fichier texte", file_types=["text"], file_count="single"
            )

        model = gr.Dropdown(
            label="Modèle NER", choices=models.keys(), value="CamemBERT"
        )

    @gr.render(inputs=[audio, file, model], triggers=[model.change])
    def handle_model_change(audio, file, model):
        if audio:
            render_tabs([transcribe(audio)], model, gr.Progress())
        elif file:
            with open(file.name, "r") as f:
                sentences = f.read().split("\n")
                render_tabs(sentences, model, gr.Progress())

    @gr.render(inputs=[audio, model], triggers=[audio.change])
    def handle_audio(audio, model, progress=gr.Progress()):
        progress(0, "Analyzing audio...")
        promptAudio = transcribe(audio)

        time.sleep(1)

        render_tabs([promptAudio], model, progress)

    @gr.render(
        inputs=[file, model],
        triggers=[file.upload],
    )
    def handle_file(file, model, progress=gr.Progress()):
        progress(0, desc="Analyzing file...")
        time.sleep(1)
        if file is not None:
            with open(file.name, "r") as f:
                progress(0.33, desc="Reading file...")
                file_content = f.read()
                rows = file_content.split("\n")
                sentences = [row for row in rows if row]
                render_tabs(sentences, model, progress)


def handle_back():
    audio.clear()
    file.clear()
    return (gr.update(visible=False), gr.update(visible=True))


def handleCityChange(city):
    stations = getStationsByCityName(city)
    return gr.update(choices=stations, value=stations[0], interactive=True)


def handleCityChange(city):
    stations = getStationsByCityName(city)
    return gr.update(choices=stations, value=stations[0], interactive=True)


def formatPath(path):
    return "\n".join([f"{i + 1}. {elem}" for i, elem in enumerate(path)])


def handleStationChange(departureStation, destinationStation):
    if departureStation and destinationStation:
        dijkstraPath, dijkstraCost = getDijkstraResult(
            departureStation, destinationStation
        )
        dijkstraPathFormatted = formatPath(dijkstraPath)
        AStarPath, AStarCost = getAStarResult(departureStation, destinationStation)
        AStarPathFormatted = formatPath(AStarPath)
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
    graphData = CSVTravelGraph("../data/sncf/timetables.csv")
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


def getEntitiesPositions(text, entity):
    start_idx = text.find(entity)
    end_idx = start_idx + len(entity)

    return start_idx, end_idx


def getDepartureAndArrivalFromText(text: str, model: str):
    entities = models[model].get_entities(text)
    if not isinstance(entities, list):
        entities = entities.tolist()
    tokenized_sentence = process_sentence(text, return_tokens=True)

    dep = None
    arr = None

    if 1 in entities:
        dep_idx = entities.index(1)
        dep = tokenized_sentence[dep_idx]
        start, end = getEntitiesPositions(text, dep)
        dep = {
            "entity": entities_label_mapping[1],
            "word": dep,
            "start": start,
            "end": end,
        }

    if 2 in entities:
        arr_idx = entities.index(2)
        arr = tokenized_sentence[arr_idx]
        start, end = getEntitiesPositions(text, arr)
        arr = {
            "entity": entities_label_mapping[2],
            "word": arr,
            "start": start,
            "end": end,
        }

    return dep, arr


def render_tabs(sentences: list[str], model: str, progress_bar: gr.Progress):
    idx = 0
    with gr.Tabs() as tabs:
        for sentence in progress_bar.tqdm(sentences, desc="Processing sentences..."):
            with gr.Tab(f"Sentence {idx}"):
                dep, arr = getDepartureAndArrivalFromText(sentence, model)
                entities = []
                for entity in [dep, arr]:
                    if entity:
                        entities.append(entity)

                # Format the classified entities
                departureCityValue = dep["word"].upper() if dep else ""
                arrivalCityValue = arr["word"].upper() if arr else ""

                # Get the available stations
                departureStations = getStationsByCityName(departureCityValue)
                departureStationValue = (
                    departureStations[0] if departureStations else ""
                )
                arrivalStations = getStationsByCityName(arrivalCityValue)
                arrivalStationValue = arrivalStations[0] if arrivalStations else ""

                dijkstraPathValues = []
                AStarPathValues = []
                timeDijkstraValue = "<p>Aucun prompt renseigné</p>"
                timeAStarValue = "<p>Aucun prompt renseigné</p>"

                # Get the paths and time for the two algorithms
                if departureStationValue and arrivalStationValue:
                    dijkstraPathValues, timeDijkstraValue = getDijkstraResult(
                        departureStationValue, arrivalStationValue
                    )
                    AStarPathValues, timeAStarValue = getAStarResult(
                        departureStationValue, arrivalStationValue
                    )

                dijkstraPathFormatted = formatPath(dijkstraPathValues)
                AStarPathFormatted = formatPath(AStarPathValues)

                with gr.Row():
                    with gr.Column(scale=1, min_width=300):
                        gr.HighlightedText(
                            value={"text": sentence, "entities": entities}
                        )
                        departureCity = gr.Textbox(
                            label="Ville de départ",
                            value=departureCityValue,
                        )
                        arrivalCity = gr.Textbox(
                            label="Ville d'arrivée",
                            value=arrivalCityValue,
                        )
                    with gr.Column(scale=2, min_width=300):
                        with gr.Row():
                            departureStation = gr.Dropdown(
                                label="Gare de départ",
                                choices=departureStations,
                                value=departureStationValue,
                            )
                            arrivalStation = gr.Dropdown(
                                label="Gare d'arrivée",
                                choices=arrivalStations,
                                value=arrivalStationValue,
                            )
                        with gr.Tab("Dijkstra"):
                            timeDijkstra = gr.HTML(value=timeDijkstraValue)
                            dijkstraPath = gr.Textbox(
                                label="Chemin emprunté",
                                value=dijkstraPathFormatted,
                                lines=len(dijkstraPathValues),
                            )

                        with gr.Tab("AStar"):
                            timeAStar = gr.HTML(value=timeAStarValue)
                            AstarPath = gr.Textbox(
                                label="Chemin emprunté",
                                value=AStarPathFormatted,
                                lines=len(AStarPathValues),
                            )

                        departureCity.change(
                            handleCityChange,
                            inputs=[departureCity],
                            outputs=[departureStation],
                        )
                        arrivalCity.change(
                            handleCityChange,
                            inputs=[arrivalCity],
                            outputs=[arrivalStation],
                        )
                        departureStation.change(
                            handleStationChange,
                            inputs=[departureStation, arrivalStation],
                            outputs=[timeDijkstra, dijkstraPath, timeAStar, AstarPath],
                        )
                        arrivalStation.change(
                            handleStationChange,
                            inputs=[departureStation, arrivalStation],
                            outputs=[timeDijkstra, dijkstraPath, timeAStar, AstarPath],
                        )

                    idx += 1


if __name__ == "__main__":
    demo.launch()
