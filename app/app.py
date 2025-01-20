import gradio as gr
from transformers import pipeline
import numpy as np
import pandas as pd
from travel_resolver.libs.nlp.ner.models import BiLSTM_NER, LSTM_NER, CamemBERT_NER
from helpers.global_vars import entities_label_mapping, PROGRESS, HTML_COMPONENTS
from travel_resolver.libs.nlp.ner.data_processing import process_sentence
from travel_resolver.libs.pathfinder.CSVTravelGraph import CSVTravelGraph
from travel_resolver.libs.pathfinder.graph import Graph
import time
import plotly.graph_objects as go

transcriber = pipeline(
    "automatic-speech-recognition", model="openai/whisper-base", device="cpu"
)

models = {"LSTM": LSTM_NER(), "BiLSTM": BiLSTM_NER(), "CamemBERT": CamemBERT_NER()}


def handle_audio(audio, model, progress=gr.Progress()):
    progress(
        0,
    )
    promptAudio = transcribe(audio)

    print(f"prompt : {promptAudio}")

    time.sleep(1)

    return render_tabs([promptAudio], model, progress)


def handle_file(file, model, progress=gr.Progress()):
    print("file upload")
    progress(0, desc=PROGRESS.ANALYZING_FILE.value)
    time.sleep(1)
    if file is not None:
        with open(file.name, "r") as f:
            progress(0.33, desc=PROGRESS.READING_FILE.value)
            file_content = f.read()
            rows = file_content.split("\n")
            sentences = [row for row in rows if row]
            return render_tabs(sentences, model, progress)


tabs_components = []

with gr.Blocks() as demo:
    with gr.Column():
        with gr.Row():
            audio = gr.Audio(label="Fichier audio", interactive=True)
            file = gr.File(
                label="Fichier texte",
                file_types=["text"],
                file_count="single",
                interactive=True,
            )

        model = gr.Dropdown(
            label="Modèle NER",
            choices=models.keys(),
            value="CamemBERT",
            interactive=True,
        )

    @gr.render(
        inputs=[audio, file, model], triggers=[audio.change, file.upload, model.change]
    )
    def handle_changes(audio, file, model):
        if audio:
            return handle_audio(audio, model)
        elif file:
            return handle_file(file, model)


def handleCityChange(city):
    stations = getStationsByCityName(city)
    return gr.update(
        choices=[station["Nom de le gare"] for station in stations],
        value=stations[0]["Nom de la gare"],
        interactive=True,
    )


def formatPath(path):
    return "\n".join([f"{i + 1}. {elem}" for i, elem in enumerate(path)])


def plotMap(stationsInformation: dict = None):
    stationNames = stationsInformation["stations"] if stationsInformation else []
    stationsLat = stationsInformation["lat"] if stationsInformation else []
    stationsLon = stationsInformation["lon"] if stationsInformation else []

    plt = go.Figure(
        go.Scattermapbox(
            lat=stationsLat,
            lon=stationsLon,
            mode="markers+lines",
            marker=go.scattermapbox.Marker(size=14),
            text=stationNames,
        )
    )

    # France's default coordinates
    defaultLat = 46.227638
    defaultLon = 2.213749

    centerLat = stationsLat[0] if stationsLat else defaultLat
    centerLon = stationsLon[0] if stationsLon else defaultLon

    plt.update_layout(
        mapbox_style="open-street-map",
        mapbox=dict(
            center=go.layout.mapbox.Center(lat=centerLat, lon=centerLon),
            pitch=0,
            zoom=3,
        ),
    )

    return plt


def handleStationChange(departureStation, destinationStation):
    if departureStation and destinationStation:
        dijkstraPath, dijkstraCost = getDijkstraResult(
            departureStation, destinationStation
        )
        dijkstraPathFormatted = formatPath(dijkstraPath)
        AStarPath, AStarCost = getAStarResult(departureStation, destinationStation)
        AStarStationsInformation = getStationsInformation(AStarPath)
        AStarPathFormatted = formatPath(AStarPath)
        return (
            gr.update(value=dijkstraCost),
            gr.update(value=dijkstraPathFormatted, lines=len(dijkstraPath)),
            gr.update(value=AStarCost),
            gr.update(value=AStarPathFormatted, lines=len(AStarPath)),
            plotMap(AStarStationsInformation),
        )
    return (
        gr.HTML(HTML_COMPONENTS.NO_PROMPT.value),
        gr.update(value=""),
        gr.HTML(HTML_COMPONENTS.NO_PROMPT.value),
        gr.update(value=""),
        gr.update(visible=None),
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
    stations = data[data["Commune"] == city]
    return dict(
        stations=stations["Nom de la gare"].to_list(),
        lat=stations["Latitude"].to_list(),
        lon=stations["Longitude"].to_list(),
    )


def getStationsInformation(stations: list[str]):
    data = pd.read_csv("../data/sncf/gares_info.csv", sep=",")
    data = data[data["Nom de la gare"].isin(stations)]
    return dict(
        stations=data["Nom de la gare"].to_list(),
        lat=data["Latitude"].to_list(),
        lon=data["Longitude"].to_list(),
    )


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
    with gr.Column() as tabs:
        for sentence in progress_bar.tqdm(sentences, desc=PROGRESS.PROCESSING.value):
            with gr.Tab(f"Sentence {idx}"):
                dep, arr = getDepartureAndArrivalFromText(sentence, model)
                print(f"dep: {dep}, arr: {arr}")
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
                    departureStations["stations"][0]
                    if len(departureStations["stations"])
                    else ""
                )
                arrivalStations = getStationsByCityName(arrivalCityValue)
                arrivalStationValue = (
                    arrivalStations["stations"][0]
                    if len(arrivalStations["stations"])
                    else ""
                )

                dijkstraPathValues = []
                AStarPathValues = []
                AStarStationsInformation = None
                timeDijkstraValue = HTML_COMPONENTS.NO_PROMPT.value
                timeAStarValue = HTML_COMPONENTS.NO_PROMPT.value

                # Get the paths and time for the two algorithms
                if departureStationValue and arrivalStationValue:
                    dijkstraPathValues, timeDijkstraValue = getDijkstraResult(
                        departureStationValue, arrivalStationValue
                    )
                    AStarPathValues, timeAStarValue = getAStarResult(
                        departureStationValue, arrivalStationValue
                    )
                    AStarStationsInformation = getStationsInformation(AStarPathValues)

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
                                choices=departureStations["stations"],
                                value=departureStationValue,
                                allow_custom_value=True,
                            )
                            arrivalStation = gr.Dropdown(
                                label="Gare d'arrivée",
                                choices=arrivalStations["stations"],
                                value=arrivalStationValue,
                                allow_custom_value=True,
                            )

                        plt = plotMap(AStarStationsInformation)

                        map = gr.Plot(plt, min_width=300)

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
                            outputs=[
                                timeDijkstra,
                                dijkstraPath,
                                timeAStar,
                                AstarPath,
                                map,
                            ],
                        )
                        arrivalStation.change(
                            handleStationChange,
                            inputs=[departureStation, arrivalStation],
                            outputs=[
                                timeDijkstra,
                                dijkstraPath,
                                timeAStar,
                                AstarPath,
                                map,
                            ],
                        )

                    idx += 1


if __name__ == "__main__":
    demo.launch()
