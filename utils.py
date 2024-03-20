import json

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt


def calculate_metrics(targets_list, predictions_list):
    accuracy = accuracy_score(targets_list, predictions_list)
    recall = recall_score(targets_list, predictions_list, average='weighted', zero_division=0)
    precision = precision_score(targets_list, predictions_list, average='weighted', zero_division=0)
    f1 = f1_score(targets_list, predictions_list, average='weighted', zero_division=0)

    return accuracy, recall, precision, f1


def create_plot(train_data, test_data, plot_name):
    train_data = np.array(train_data) * 100
    test_data = np.array(test_data) * 100
    plt.figure(figsize=(8, 6))
    plt.plot((range(len(train_data))), train_data, label="Train set", linewidth=2)
    plt.plot((range(len(test_data))), test_data, label="Test set", linewidth=2)
    plt.legend(loc="lower right")
    plt.grid()
    plt.xlabel('Epochs')
    plt.ylabel(f'{plot_name}')
    plt.yticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    return plt


def save_model_results(path, lists):
    list_train_loss = lists["train_loss"]
    list_train_accuracy = lists["train_accuracy"]
    list_train_recall = lists["train_recall"]
    list_train_precision = lists["train_precision"]
    list_train_f1 = lists["train_f1"]

    list_test_loss = lists["test_loss"]
    list_test_accuracy = lists["test_accuracy"]
    list_test_recall = lists["test_recall"]
    list_test_precision = lists["test_precision"]
    list_test_f1 = lists["test_f1"]

    loss_plt = create_plot(list_train_loss, list_test_loss, "Loss value")
    loss_plt.savefig(f"{path}/lossplt.png")
    accuracy_plt = create_plot(list_train_accuracy, list_test_accuracy, "Accuracy [%]")
    accuracy_plt.savefig(f"{path}/accuracy.png")
    recall_plt = create_plot(list_train_recall, list_test_recall, "Recall [%]")
    recall_plt.savefig(f"{path}/recall.png")
    precision_plt = create_plot(list_train_precision, list_test_precision, "Precision [%]")
    precision_plt.savefig(f"{path}/precision.png")
    f1_plt = create_plot(list_train_f1, list_test_f1, "F1-score [%]")
    f1_plt.savefig(f"{path}/f1.png")


def read_config(path):
    with open(path, "r") as file:
        data = json.load(file)
    return data
