from data.file_manager import save_torch_data
import torch
import numpy as np

from utils.cuda import has_cuda
from train.training_config import TrainingConfig


class ModelTrainer():

    @staticmethod
    def train(model, optimizer, train_loader, test_loader, training_config: TrainingConfig):

        losses = []
        train_accuracies = []
        test_accuracies = []

        best_accuracy = 0
        best_epoch = 0

        model.train()

        epochs = training_config['epochs']
        starting_epoch = training_config['starting_epoch']
        loss_fnc = training_config['loss_fnc']
        save_path = training_config['save_path']
        save_name = training_config['save_name']
        log_frequency = training_config['log_frequency']

        for epoch in range(starting_epoch, epochs):

            epoch_losses = []
            epoch_num_train_correct_predictions = 0
            epoch_num_train_total_predictions = 0

            for batch_idx, (data, target) in enumerate(train_loader):

                if has_cuda():
                    data, target = data.cuda(), target.cuda()

                optimizer.zero_grad()

                predicted = model(data)[0]

                loss = loss_fnc(predicted, target)
                epoch_losses.append(loss.item())

                loss.backward()
                optimizer.step()

                _, predicted_indices = predicted.max(1)

                train_accuracy, num_correct_predictions, num_predictions = ModelTrainer.get_accuracy(
                    predicted_indices, target)

                epoch_num_train_correct_predictions += num_correct_predictions
                epoch_num_train_total_predictions += num_predictions

                if log_frequency > 0 and batch_idx % log_frequency == 1:
                    print('\rTrain Epoch: {}/{} [{:5}/{} ({:3.0f}%)]\tLoss: {:.6f}\t Train Accuracy: {}/{} ({:3.4f}%)'.format(
                        epoch + 1,
                        epochs,
                        (batch_idx+1) * len(data),
                        len(train_loader.dataset),
                        100.0 * (batch_idx+1) / len(train_loader),
                        loss.cpu(),
                        num_correct_predictions,
                        num_predictions,
                        train_accuracy*100,
                    ), end='')

            average_epoch_loss = np.mean(epoch_losses)
            losses.append((epoch, average_epoch_loss))

            epoch_train_accuracy = epoch_num_train_correct_predictions / epoch_num_train_total_predictions
            train_accuracies.append((epoch, epoch_train_accuracy))

            if test_loader is not None:
                test_accuracy, test_num_correct_predictions, test_num_predictions = ModelTrainer.validate(
                    model, test_loader)
                test_accuracies.append((epoch, test_accuracy))

                new_best = False

                if test_accuracy > best_accuracy:
                    best_accuracy = test_accuracy
                    best_epoch = epoch
                    new_best = True

                print('\nTrain Epoch: {}/{} [{:5}/{} ({:3.0f}%)]\tLoss: {:.6f}\t Test Accuracy: {}/{} ({:3.4f}%){}'.format(
                    epoch+1,
                    epochs,
                    len(train_loader.dataset),
                    len(train_loader.dataset),
                    100.0 * (batch_idx+1) / len(train_loader),
                    loss.cpu(),
                    test_num_correct_predictions,
                    test_num_predictions,
                    test_accuracy*100,
                    ' [NEW BEST!]' if new_best else '',
                ))

            if save_path is not None:
                save_dir = save_torch_data(save_path, save_name, model, optimizer, epoch)
                print(f'Saved model to {save_dir}')

        print(f'Best accuracy: {best_accuracy} (Epoch {best_epoch})')
        return losses, train_accuracies, test_accuracies

    @staticmethod
    def validate(model, test_loader):
        test_data = test_loader.dataset.data.float()
        test_data = test_data[:, None, ...]  # Expand dimension of data
        test_targets = test_loader.dataset.targets

        model.eval()

        output = model(test_data)[0]
        _, output_indices = output.max(1)

        accuracy, num_correct_predictions, num_predictions = ModelTrainer.get_accuracy(output_indices, test_targets)
        return accuracy, num_correct_predictions, num_predictions

    @staticmethod
    def get_accuracy(outputs, labels):
        equalities = torch.eq(outputs, labels).cpu()

        num_predictions = equalities.size()[0]
        num_correct_predictions = equalities.sum().item()
        accuracy = num_correct_predictions/num_predictions

        return accuracy, num_correct_predictions, num_predictions
