import torch
import torch.nn as nn
from torchvision.transforms import transforms
# SINCE THE MNIST DATASET IS IN 28X28, PADDING OF 2 IS PUT ON C1 LAYER
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5,self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        self.c1 = nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5,stride=1,padding=2)
        self.c2 = nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5,stride=1,padding=0)
        self.c3 = nn.Conv2d(in_channels=16,out_channels=120,kernel_size=5,stride=1,padding=0)
        # self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=2,stride=2)
        # self.fc0 = nn.Linear(in_features=400,out_features=120)
        self.fc1 = nn.Linear(in_features=120,out_features=84)
        self.fc2 = nn.Linear(in_features=84,out_features=10)

    def forward(self, img):
        x = self.quant(img)
        # in 1x32x32 out 6x28x28
        x = self.c1(x)
        print('c1',x.int_repr())
        # in 6x28x28 out 6x14x14
        # replace tanh with relu
        x = self.relu(self.max_pool(x))

        # in 6x14x14 out 16x10x10
        x = self.c2(x)
        # in 16x10x10 out 16x5x5
        # replace tanh with relu
        x = self.relu(self.max_pool(x))

        # # in 16x5x5 out 400x1x1
        # x = torch.flatten(x,1)
        # x = self.tanh(self.fc0(x))

        # in 16x5x5 out 120x1x1
        # replace tanh with relu
        x = self.relu(self.c3(x))
        x = torch.flatten(x,1)

        # replace tanh with relu
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.dequant(x)
        return x
    
