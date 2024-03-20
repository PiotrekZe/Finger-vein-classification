import Model
import Dataset
import VeinDataset
import RunModel
import utils

from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim


def main():
    config_data = utils.read_config("config_file.json")

    learning_rate = config_data['model']['learning_rate']
    weight_decay = config_data['model']['weight_decay']
    batch_size = config_data['model']['batch_size']
    epochs = config_data['model']['epochs']
    device = config_data['model']['device']

    path = config_data['file']['path']
    width = config_data['file']['width']
    height = config_data['file']['height']
    path_to_save = config_data['file']['path_to_save']
    num_classes = config_data['file']['classes']

    dataset = Dataset.Dataset(path, width, height)
    X_train, X_test, y_train, y_test = dataset.read_dataset()
    train_dataset = VeinDataset.VeinDataset(X_train, y_train)
    test_dataset = VeinDataset.VeinDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = Model.FingerNet(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    run_model = RunModel.RunModel(epochs, device, train_loader, test_loader)

    for epoch in range(epochs):
        print(f"Epoch: {epoch}/{epochs}")
        train_running_loss, train_accuracy, train_recall, train_precision, train_f1 = run_model.train_model(model,
                                                                                                            criterion,
                                                                                                            optimizer)
        test_running_loss, test_accuracy, test_recall, test_precision, test_f1 = run_model.test_model(model, criterion)


if __name__ == '__main__':
    main()
