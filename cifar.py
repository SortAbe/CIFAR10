import math
import torch
import torchvision

from torch import nn
from torchvision import transforms

from matplotlib import pyplot

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Parameters
epochs = 4
batch_size = 4
learning_rate = 0.001

#Dataset
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

train_set = torchvision.datasets.CIFAR10(root='./data2', train=True, download=True, transform=transform)
test_set = torchvision.datasets.CIFAR10(root='./data2', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')

#Convolutional Neural Network
class ConvNet(nn.Module):

    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 4) # 3 color channels, 6 output channel size , 5 kernel size
        self.pool = nn.MaxPool2d(2, 2) # 2 kernel size , 2 stride
        self.conv2 = nn.Conv2d(6, 16, 4)
        self.fc1 = nn.Linear(400, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc_out = nn.Linear(200, 10)
        self.af = nn.LeakyReLU()

    def forward(self, x):
        out = self.pool(self.af(self.conv1(x)))
        out = self.pool(self.af(self.conv2(out)))
        #out = out.view(-1, 16*4*4) # Flatten output of Convolutional
        out = torch.flatten(out, 1)
        out = self.af(self.fc1(out))
        out = self.af(self.fc2(out))
        out = self.fc_out(out)
        return out

    def count(self):
        size = 0
        print(self.state_dict().keys())
        for key in self.state_dict().keys():
            size += math.prod(list(self.state_dict()[key].size()))
        print(size)

    def save(self):
        torch.save(self.state_dict(), 'parameters2')

    def load(self):
        self.load_state_dict(torch.load('parameters2'))

model = ConvNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

n_steps = len(train_loader)
def train():
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Original shape: [4, 3, 32, 32] = 4, 3, 1024
            # input_layer: 3 input channels, 6 output channels, 5 kernel size
            images = images.to(device)
            labels = labels.to(device)

            #Forward
            outputs = model(images)
            loss = criterion(outputs, labels)

            #Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 1_000 == 0:
                print(f'Epoch: {epoch}/{epochs}, step: {i}/{n_steps}, loss: {loss.item():.4f}')
    print('\nTraining DONE!\n')

train()

with torch.no_grad():
    n_correct = 0
    n_samples = 0

    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        # max return (value, index)
        _,predicts = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicts == labels).sum().item()

        for i in range(batch_size):
            label = labels[i]
            predict = predicts[i]
            if label == predict:
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    accuracy = 100 * n_correct / n_samples
    print(accuracy)
    for i in range(len(classes)):
        print(f'{classes[i]}: {100 * n_class_correct[i]/n_class_samples[i]}')
model.save()
model.count()
