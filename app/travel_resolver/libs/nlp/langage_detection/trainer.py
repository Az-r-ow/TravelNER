import csv
import joblib
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
import travel_resolver.libs.nlp.langage_detection.variables as var


def read_data():
    """
    Retreive and format data from csv input files
    """
    x, y = [], []
    i = 1
    for lang in var.CORRESP_LANG:
        first = True
        current_file = "../../../../data/langage_detection/trainset/"
        current_file += lang + "_trainset.csv"
        with open(current_file, "r") as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                if not first:
                    x.append(np.array(row, dtype=np.float64))
                    y.append(i)
                else:
                    first = False
        i += 1
    return train_test_split(np.array(x), y, test_size=0.2, random_state=5)


def train():
    """
    Train the model and generate a backup.
    """
    x_train, x_test, y_train, y_test = read_data()

    model = SGDClassifier()
    model.fit(x_train, y_train)
    joblib.dump(model, "model.sav")

    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
    print("Accuracy: {:.2f}%".format(accuracy * 100))
