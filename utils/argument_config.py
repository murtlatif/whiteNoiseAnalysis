from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--name', default='model', dest='model_name', help='Name of the model')
    parser.add_argument('-d', '--seed', type=int, help='Manual seed for generating numbers')
    parser.add_argument('-s', '--save-model', action='store_true', help='Should save the model parameters')
    parser.add_argument('-l', '--load-model', help='Path to saved model parameters')
    parser.add_argument('-S', '--save-training', action='store_true', help='Save the training data')
    parser.add_argument('-p', '--hide-plot', action='store_false', dest='plot_data', help='Plot the training data')
    parser.add_argument('-r', '--learning-rate', type=float, help='Learning rate of the optimizer', default=1e-3)
    parser.add_argument('-e', '--epochs', type=int, help='Number of epochs to train', default=15)
    parser.add_argument('-g', '--gamma', type=float, default=0.5)
    parser.add_argument('-t', '--train', action='store_true', help='Train the model if this flag is set')
    parser.add_argument('-N', '--wn-num-batches', type=int, default=10000,
                        help='Number of white noise batches for analysis')
    parser.add_argument('-W', '--wn-batch-size', type=int, default=100,
                        help='Number of samples per white noise batch')
    parser.add_argument('-y', '--summary', action='store_true', help='Show model summary at the end')
    parser.add_argument('--large', action='store_true', help='Use the larger CNN model')
    parser.add_argument('-b', '--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('-B', '--test-batch-size', type=int, default=32, help='Batch size for testing')
    parser.add_argument('-f', '--log-frequency', type=int, default=50, help='Frequency of training logs')
    parser.add_argument('-F', '--fashion', action='store_true', help='Use the Fashion MNIST dataset')
    parser.add_argument('-L', '--load-class-data', help='Path to saved classification data')
    parser.add_argument('--show-samples', action='store_true', help='Show sample stimulus images for each class')
    parser.add_argument('--classification-analysis', action='store_true', help='Perform classification analysis')
    parser.add_argument('--spike-triggered-analysis', action='store_true', help='Perform spike triggered analysis')

    args = parser.parse_args()
    return args
