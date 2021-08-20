import numpy as np
import torch


def add_filename_extension(filename, compressed=False):
    target_extension = '.npz' if compressed else '.npy'
    if filename[-4:] != target_extension:
        filename += target_extension

    return filename


def save_training_results(loss_data, train_data, test_data, filename='training_results_data.npz'):
    filename = add_filename_extension(filename, compressed=True)
    np.savez(filename, loss_data=loss_data, train_data=train_data, test_data=test_data)

    return filename


def load_training_results(filename='training_results_data.npz'):
    filename = add_filename_extension(filename, compressed=True)

    results = np.load(filename)
    loss_data = results['loss_data']
    train_data = results['train_data']
    test_data = results['test_data']

    return loss_data, train_data, test_data


def save_classification_data(class_data, filename='class_data.npy'):
    filename = add_filename_extension(filename, compressed=False)
    np.save(filename, class_data)

    return filename


def load_classification_data(filename):
    filename = add_filename_extension(filename, compressed=False)
    class_data = np.load(filename, allow_pickle=True).item()
    return class_data


def save_torch_data(save_path, save_name, model, optimizer, epoch):
    save_dir = f'{save_path}/{save_name}_epoch_{epoch}.pth'

    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, save_dir)

    return save_dir


def load_torch_data(filepath: str, model, optimizer):
    loaded_state = torch.load(filepath)

    model.load_state_dict(loaded_state['model'])
    optimizer.load_state_dict(loaded_state['optimizer'])
    epoch = loaded_state['epoch']

    return epoch
