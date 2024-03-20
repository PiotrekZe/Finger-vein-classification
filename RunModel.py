import numpy as np
import torch
import utils


class RunModel:
    def __init__(self, epochs, device, train_loader, test_loader):
        self.epochs = epochs
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader

    def train_model(self, model, criterion, optimizer):
        model.train()
        running_loss = 0
        predictions_list = []
        targets_list = []
        for images, targets in self.train_loader:
            images = images.to(self.device)
            targets = targets.to(self.device)

            optimizer.zero_grad()
            batch_outputs = model(images)
            loss = criterion(batch_outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, batch_outputs = torch.max(batch_outputs, 1)
            predictions_list.append(np.array(batch_outputs.cpu().detach()))
            targets_list.append(np.array(targets.cpu().detach()))

        targets_list = np.concatenate(targets_list)
        predictions_list = np.concatenate(predictions_list)
        running_loss = running_loss / len(self.train_loader)
        accuracy, recall, precision, f1 = utils.calculate_metrics(targets_list, predictions_list)

        print(f"Training. Loss: {running_loss}. Accuracy: {accuracy}")
        return running_loss, accuracy, recall, precision, f1

    def test_model(self, model, criterion):
        model.eval()
        running_loss = 0
        predictions_list = []
        targets_list = []
        with torch.no_grad():
            for images, targets in self.test_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)

                batch_outputs = model(images)
                loss = criterion(batch_outputs, targets)

                running_loss += loss.item()
                _, batch_outputs = torch.max(batch_outputs, 1)
                predictions_list.append(np.array(batch_outputs.cpu().detach()))
                targets_list.append(np.array(targets.cpu().detach()))

        targets_list = np.concatenate(targets_list)
        predictions_list = np.concatenate(predictions_list)
        running_loss = running_loss / len(self.train_loader)
        accuracy, recall, precision, f1 = utils.calculate_metrics(targets_list, predictions_list)

        print(f"Testing. Loss: {running_loss}. Accuracy: {accuracy}")
        return running_loss, accuracy, recall, precision, f1
