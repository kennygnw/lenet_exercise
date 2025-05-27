import lenet_model
import torch
from pathlib import Path
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
import matplotlib.pyplot as plt

# from lenet_train import mnist_transforms, loss_fn, accuracy
MODEL_NAME = "lenet5_mnist_2146.pth"
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# device-agnostic setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# load test dataset
BATCH_SIZE = 32
test_dataset  = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transforms.ToTensor())
imgs = torch.stack([img for img, _ in test_dataset], dim=0)
mean = imgs.view(1, -1).mean(dim=1)    # or imgs.mean()
std = imgs.view(1, -1).std(dim=1)     # or imgs.std()
# create Transformation (converting from Image class to Tensor and normalize)
mnist_transforms = transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize(mean=mean, std=std)])

test_dataset  = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=mnist_transforms)
class_names = test_dataset.classes

test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)
# Loading the saved model
net = lenet_model.LeNet5()
net.load_state_dict(torch.load(MODEL_SAVE_PATH))
test_loss, test_acc = 0, 0

net.to(device)

net.eval()

# set loss function and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=net.parameters(), lr=0.0001)
accuracy = Accuracy(task='multiclass', num_classes=10)

with torch.inference_mode():
    for X, y in test_dataloader:
        X, y = X.to(device), y.to(device)
        y_pred = net(X)
        
        test_loss += loss_fn(y_pred, y)
        test_acc += accuracy(y_pred, y)
        
    test_loss /= len(test_dataloader)
    test_acc /= len(test_dataloader)

print(f"Test loss: {test_loss: .5f}| Test acc: {test_acc: .5f}")

#  See random images with their labels
torch.manual_seed(42)  # setting random seed
fig = plt.figure(figsize=(12, 4))

rows, cols = 2, 6
for i in range(1, (rows * cols) + 1):
    random_idx = torch.randint(0, len(test_dataset), size=[1]).item()
    img, label_gt = test_dataset[random_idx]
    img_temp = img.unsqueeze(dim=0).to(device)
    # print(img.shape)
    label_pred = torch.argmax(net(img_temp))
    fig.add_subplot(rows, cols, i)
    img = img.permute(1, 2, 0)    # CWH --> WHC
    plt.imshow(img, cmap='gray')
    if label_pred == label_gt:
        plt.title(class_names[label_pred], color='g') # for correct prediction
    else:
        plt.title(class_names[label_pred], color='r') # for incorrect prediction
plt.axis(False)
plt.tight_layout()
plt.show()
