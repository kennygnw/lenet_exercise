import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import lenet_model
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from pathlib import Path
from datetime import datetime
import time
from torch.utils.tensorboard import SummaryWriter
import os
USE_LOGGING =False
if USE_LOGGING:
    import logging

time_start = datetime.now().strftime('%H%M')
if USE_LOGGING:
    # Setup basic logger
    logging.basicConfig(filename=f"train_events_{time_start}.log", level=logging.INFO)
    def log_event(event: str):
        logging.info(f"{time.time():.2f},{event}")
    # load MNIST DATASET
    # if first time running, no /data folder with dataset, set download to True
train_val_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=False, transform=transforms.ToTensor())
class_names = train_val_dataset.classes

# Calculate mean and std of the train dataset
imgs = torch.stack([img for img, _ in train_val_dataset], dim=0)
mean = imgs.view(1, -1).mean(dim=1)    # or imgs.mean()
std = imgs.view(1, -1).std(dim=1)     # or imgs.std()
# create Transformation (converting from Image class to Tensor and normalize)
mnist_transforms = transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize(mean=mean, std=std)])
mnist_trainset = torchvision.datasets.MNIST(root="./data", train=True, download=False, transform=mnist_transforms)
# split to train dataset and validation dataset
train_size = int(0.8 * len(mnist_trainset))
val_size = len(mnist_trainset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset=mnist_trainset, lengths=[train_size, val_size])

# load dataset and set number of data per batch
BATCH_SIZE = 32
train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True)

net = lenet_model.LeNet5()

# set loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=net.parameters(), lr=0.0001)
accuracy = Accuracy(task='multiclass', num_classes=10)

# device-agnostic setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
accuracy = accuracy.to(device)
lenet_model = net.to(device)
# CREATE LOGGER FOR LOSS AND ACCURACY
log_dir = os.path.join("runs", time_start)
writer = SummaryWriter(log_dir)

# train
EPOCHS = 25
for epoch in range(EPOCHS):
    if USE_LOGGING:
        log_event(f"EPOCH_START:{epoch}")
    # Training loop
    train_loss, train_acc = 0.0, 0.0
    lenet_model.train()
    for X, y in train_dataloader:
        X, y = X.to(device), y.to(device)
        
        y_pred = lenet_model(X)
        
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        
        acc = accuracy(y_pred, y)
        train_acc += acc
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if USE_LOGGING:
        log_event(f"EPOCH_END:{epoch}")
    train_loss /= len(train_dataloader)
    train_acc /= len(train_dataloader)
        
    # Validation loop
    val_loss, val_acc = 0.0, 0.0
    lenet_model.eval()
    if USE_LOGGING:
        log_event(f"VALIDATION_START:{epoch}")
    with torch.inference_mode():
        for X, y in val_dataloader:
            X, y = X.to(device), y.to(device)
            
            y_pred = lenet_model(X)
            
            loss = loss_fn(y_pred, y)
            val_loss += loss.item()
            
            acc = accuracy(y_pred, y)
            val_acc += acc
        if USE_LOGGING:
            log_event(f"VALIDATION_END:{epoch}")
        val_loss /= len(val_dataloader)
        val_acc /= len(val_dataloader)
    writer.add_scalars(main_tag="Loss", tag_scalar_dict={"train/loss": train_loss, "val/loss": val_loss}, global_step=epoch)
    writer.add_scalars(main_tag="Accuracy", tag_scalar_dict={"train/acc": train_acc, "val/acc": val_acc}, global_step=epoch)
    
    print(f"Epoch: {epoch}| Train loss: {train_loss: .5f}| Train acc: {train_acc: .5f}| Val loss: {val_loss: .5f}| Val acc: {val_acc: .5f}")
if USE_LOGGING:
    log_event("TRAINING_DONE")
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

MODEL_NAME = f"lenet5_mnist_{time_start}.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# Saving the model
print(f"Saving the model: {MODEL_SAVE_PATH}")
torch.save(obj=net.state_dict(), f=MODEL_SAVE_PATH)
writer.close()
