
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

device = ('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1,),(0.3,))])

train_set = datasets.EMNIST('DATA_EMNIST/raw', split='letters', train=False, download=True, transform=transform)
trainLoader = DataLoader(train_set, batch_size=64, shuffle=True)
test_set = datasets.EMNIST('DATA_EMNIST/',split='letters', train=False, download=True, transform=transform)
testLoader = DataLoader(test_set, batch_size=64, shuffle=True)

num_letters = 27

training_data = enumerate(trainLoader)
batch_idx, (images, labels) = next(training_data)

num_folds = 3
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
train_loss_fold = []
test_loss_fold = []

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.convolutional_neural_network_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=12, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(24 * 7 * 7, out_features=64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=64, out_features=num_letters)
        )
    def forward(self, x):
        x = self.convolutional_neural_network_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        x = F.log_softmax(x, dim=1)
        return x

for fold, (train_indices, test_indices) in enumerate(kf.split(train_set)):
    train_fold = torch.utils.data.Subset(train_set, train_indices)
    test_fold = torch.utils.data.Subset(train_set, test_indices)
    trainLoader_fold = DataLoader(train_fold, batch_size=64, shuffle=True)
    testLoader_fold = DataLoader(test_fold, batch_size=64, shuffle=True)

    model = Network()
    model.to(device)
    print(model)

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    epochs = 10
    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for idx, (images, labels) in enumerate(trainLoader_fold):

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            output = model(images)
            loss = criterion(output, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        else:
            model.eval()
            test_loss = 0

            with torch.no_grad():
                for images, labels in testLoader_fold:
                    images = images.to(device)
                    labels = labels.to(device)

                    log_probabilities = model(images)
                    test_loss += criterion(log_probabilities, labels)

                    probabilities = torch.exp(log_probabilities)
                    top_prob, top_class = probabilities.topk(1, dim=1)
                    predictions = top_class == labels.view(*top_class.shape)

            train_losses.append(train_loss / len(trainLoader_fold))
            test_losses.append(test_loss / len(testLoader_fold))

            print("Epoch: {}/{}  ".format(fold + 1, epoch+1,epochs),
                  "Training loss: {:.4f}  ".format(train_loss / len(trainLoader_fold)),
                  "Testing loss: {:.4f}  ".format(test_loss / len(testLoader_fold)))

    train_loss_fold.append(train_losses)
    test_loss_fold.append(test_losses)

average_train_loss = np.mean(train_loss_fold, axis=0)
average_test_loss = np.mean(test_loss_fold, axis=0)

plt.plot(average_train_loss, label='Training Loss')
plt.plot(average_test_loss, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()

