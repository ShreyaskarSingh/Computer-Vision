import time
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self, mode):
        super(ConvNet, self).__init__()
        #Conv layers
        self.conv_layer1 = nn.Conv2d(1, 40, 5)
        self.conv_layer2 = nn.Conv2d(40, 40, 5)

        # Step1
        self.fc_step1 = nn.Linear(28 * 28, 100)       #MNIST dataset image size = 28x28 #100 neurons
        self.fc1_step1_op = nn.Linear(100, 10)        #i/p->100 Neurons; o/p-> #10 way classifier 0-9

        # Step2
        self.fc_step2 = nn.Linear(40 * 4 * 4, 100)    # image dimensions after pooling

        # Step4
        self.fc_step4 = nn.Linear(100, 100)

        # Step5
        self.dropout = nn.Dropout(0.5)
        self.fc_step5_1 = nn.Linear(40 * 4 * 4, 1000)
        self.fc_step5_2 = nn.Linear(1000, 1000)
        self.fc_step5_op = nn.Linear(1000, 10)

        if mode == 1:
            self.forward = self.model_1
        elif mode == 2:
            self.forward = self.model_2
        elif mode == 3:
            self.forward = self.model_3
        elif mode == 4:
            self.forward = self.model_4
        elif mode == 5:
            self.forward = self.model_5
        else:
            print("Invalid mode ", mode, "selected. Select between 1-5")
            exit(0)

    # Baseline model. step 1
    def model_1(self, X):
        X = X.view(-1, 28 * 28 * 1)
        X = F.sigmoid(self.fc_step1(X))
        return self.fc1_step1_op(X)

    # Use two convolutional layers.
    def model_2(self, X):
        X = self.conv_layer1(X)
        X = F.sigmoid(X)
        X = F.max_pool2d(X, 2)
        X = self.conv_layer2(X)
        X = F.sigmoid(X)
        X = F.max_pool2d(X, 2)
        X = X.view(10, -1)
        X = self.fc_step2(X)
        X = F.sigmoid(X)
        return self.fc1_step1_op(X)

    # Replace sigmoid with ReLU.
    def model_3(self, X):
        X = self.conv_layer1(X)
        X = F.relu(X)
        X = F.max_pool2d(X, 2)
        X = self.conv_layer2(X)
        X = F.relu(X)
        X = F.max_pool2d(X, 2)
        X = X.view(10, -1)
        X = self.fc_step2(X)
        X = F.relu(X)
        return self.fc1_step1_op(X)

    # Add one extra fully connected layer.
    def model_4(self, X):
        X = self.conv_layer1(X)
        X = F.relu(X)
        X = F.max_pool2d(X, 2)
        X = self.conv_layer2(X)
        X = F.relu(X)
        X = F.max_pool2d(X, 2)
        X = X.view(10, -1)
        X = self.fc_step2(X)
        X = F.relu(X)
        X = self.fc_step4(X)
        return self.fc1_step1_op(X)

    # Use Dropout now.
    def model_5(self, X):
        # ======================================================================
        # Two convolutional layers + two fully connected layers, with ReLU.
        # and  + Dropout
        X = self.conv_layer1(X)
        X = F.relu(X)
        X = F.max_pool2d(X, 2)
        X = self.conv_layer2(X)
        X = F.relu(X)
        X = F.max_pool2d(X, 2)
        X = X.view(10, -1)
        X = self.dropout(X)
        X = self.fc_step5_1(X)
        X = F.relu(X)
        X = self.fc_step5_2(X)
        X = F.relu(X)
        return self.fc_step5_op(X)
