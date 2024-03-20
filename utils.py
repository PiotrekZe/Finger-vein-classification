import json
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score


def calculate_metrics(targets_list, predictions_list):
    accuracy = accuracy_score(targets_list, predictions_list)
    recall = recall_score(targets_list, predictions_list, average='weighted', zero_division=0)
    precision = precision_score(targets_list, predictions_list, average='weighted', zero_division=0)
    f1 = f1_score(targets_list, predictions_list, average='weighted', zero_division=0)

    return accuracy, recall, precision, f1


def create_plot():
    pass


def save_model_results():
    pass


def read_config(path):
    with open(path, "r") as file:
        data = json.load(file)
    return data
