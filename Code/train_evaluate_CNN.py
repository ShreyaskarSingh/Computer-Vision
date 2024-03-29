from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from ConvNet import ConvNet
import argparse
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


def train(model, device, train_loader, optimizer, criterion, epoch, batch_size, output_txt):
    # Set model to train mode before each epoch
    model.train()

    # Empty list to store losses
    losses = []
    correct = 0

    # Iterate over entire training samples (1 epoch)
    for batch_idx, batch_sample in enumerate(train_loader):
        data, target = batch_sample

        # Push data/label to correct device
        data, target = data.to(device), target.to(device)

        # Reset optimizer gradients. Avoids grad accumulation (accumulation used in RNN).
        optimizer.zero_grad()

        # Do forward pass for current set of data
        output = model(data)

        loss = criterion(output, target)

        # Computes gradient based on final loss
        loss.backward()

        # Store loss
        losses.append(loss.item())

        # Optimize model parameters based on learning rate and gradient
        optimizer.step()

        # Get predicted index by selecting maximum log-probability
        pred = output.argmax(dim=1, keepdim=True)

        correct += pred.eq(target.view_as(pred)).sum().item()

    train_loss = float(np.mean(losses))
    train_acc = 100. * correct / ((batch_idx + 1) * batch_size)
    output_txt.write('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        float(np.mean(losses)), correct, (batch_idx + 1) * batch_size,
                                         100. * correct / ((batch_idx + 1) * batch_size)))
    print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        float(np.mean(losses)), correct, (batch_idx + 1) * batch_size,
                                         100. * correct / ((batch_idx + 1) * batch_size)))

    return train_loss, train_acc


def test(model, device, test_loader, output_txt):
    model.eval()

    losses = []
    correct = 0

    # Set torch.no_grad() to disable gradient computation and backpropagation
    with torch.no_grad():
        for batch_idx, sample in enumerate(test_loader):
            data, target = sample
            data, target = data.to(device), target.to(device)
            output = model(data)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(output, target)

            losses.append(loss.item())

            pred = output.argmax(dim=1, keepdim=True)

            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss = float(np.mean(losses))
    accuracy = 100. * correct / len(test_loader.dataset)
    output_txt.write('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))

    return test_loss, accuracy


def run_main(FLAGS):
    # Check if cuda is available
    use_cuda = torch.cuda.is_available()

    # Set proper device based on cuda availability
    file = open("Steps" + str(FLAGS.mode), 'w+')
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Torch device selected: ", device)

    model = ConvNet(FLAGS.mode).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=FLAGS.learning_rate)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load datasets for training and testing
    # Inbuilt datasets available in torchvision (check documentation online)
    dataset1 = datasets.MNIST('./data/', train=True, download=True,
                              transform=transform)
    dataset2 = datasets.MNIST('./data/', train=False,
                              transform=transform)
    train_loader = DataLoader(dataset1, batch_size=FLAGS.batch_size,
                              shuffle=True, num_workers=4)
    test_loader = DataLoader(dataset2, batch_size=FLAGS.batch_size,
                             shuffle=False, num_workers=4)

    best_accuracy = 0.0
    train_loss1 = []
    test_loss1 = []
    train_accuracy1 = []
    test_accuracy1 = []
    epoch_array = []

    epoch_array.append(0)
    train_loss1.append(0)
    test_loss1.append(0)
    train_accuracy1.append(0)
    test_accuracy1.append(0)

    # Run training for n_epochs specified in config
    for epoch in range(1, FLAGS.num_epochs + 1):
        print("Epoch No.", epoch)
        train_loss, train_accuracy = train(model, device, train_loader,
                                           optimizer, criterion, epoch, FLAGS.batch_size, file)
        test_loss, test_accuracy = test(model, device, test_loader, file)

        train_loss1.append(100. * train_loss)
        test_loss1.append(100. * test_loss)
        train_accuracy1.append(train_accuracy)
        test_accuracy1.append(test_accuracy)
        epoch_array.append(epoch)

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy

    plt.plot(epoch_array, train_loss1, color='g', label="Train Loss")
    plt.plot(epoch_array, test_loss1, color='r', label="Test Loss")
    plt.xlabel("EPOCH")
    plt.ylabel("LOSS")
    plt.title("MODE" + str(FLAGS.mode))
    plt.legend()
    plt.show()

    plt.plot(epoch_array, train_accuracy1, color='g', label="Train Accuracy")
    plt.plot(epoch_array, test_accuracy1, color='r', label="Test Accuracy")
    plt.xlabel("EPOCH")
    plt.ylabel("Accuracy")
    plt.title("MODE" + str(FLAGS.mode))
    plt.legend()
    plt.show()

    file.write("accuracy is {:2.2f}".format(best_accuracy))
    print("Training and evaluation finished")
    print("accuracy is {:2.2f}".format(best_accuracy))
    print("Training and evaluation finished")
    file.close()


if __name__ == '__main__':
    # Set parameters for Sparse Autoencoder
    parser = argparse.ArgumentParser('CNN Exercise.')
    parser.add_argument('--mode',
                        type=int, default=3,
                        help='Select mode between 1-5.')
    parser.add_argument('--learning_rate',
                        type=float, default=0.03,
                        help='Initial learning rate.')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=60,
                        help='Number of epochs to run trainer.')
    parser.add_argument('--batch_size',
                        type=int, default=10,
                        help='Batch size. Must divide evenly into the dataset sizes.')
    parser.add_argument('--log_dir',
                        type=str,
                        default='logs',
                        help='Directory to put logging.')

    FLAGS = None
    FLAGS, unparsed = parser.parse_known_args()

    run_main(FLAGS)

