import lenet_model
import torch
from pathlib import Path
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
import matplotlib.pyplot as plt
import os
from torchvision.utils import save_image

# from lenet_train import mnist_transforms, loss_fn, accuracy
MODEL_NAME = "quantized_lenet5_mnist_20250703_1003.pth"

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
print(f"MNIST mean: {mean.item():.6f}")
print(f"MNIST std:  {std.item():.6f}")
# create Transformation (converting from Image class to Tensor and normalize)
mnist_transforms = transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize(mean=mean, std=std)])

test_dataset  = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=mnist_transforms)
class_names = test_dataset.classes

test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)

# # Loading the saved model
# net = lenet_model.LeNet5()
# net.load_state_dict(torch.load(MODEL_SAVE_PATH))

# Step 1: Re-create model and set QAT config
model_fp32 = lenet_model.LeNet5()
model_fp32.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
torch.quantization.prepare_qat(model_fp32 , inplace=True)

# Step 2: Convert to quantized version
net = torch.quantization.convert(model_fp32 .eval(), inplace=False)
print(net)
# Step 3: Load quantized weights
net.load_state_dict(torch.load(MODEL_SAVE_PATH))


test_loss, test_acc = 0, 0

net.to(device)

net.eval()

# # set loss function and optimizer
# loss_fn = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(params=net.parameters(), lr=0.0001)
# accuracy = Accuracy(task='multiclass', num_classes=10)

# with torch.inference_mode():
#     for X, y in test_dataloader:
#         X, y = X.to(device), y.to(device)
#         y_pred = net(X)
        
#         test_loss += loss_fn(y_pred, y)
#         test_acc += accuracy(y_pred, y)
        
#     test_loss /= len(test_dataloader)
#     test_acc /= len(test_dataloader)

# print(f"Test loss: {test_loss: .5f}| Test acc: {test_acc: .5f}")

#  See random images with their labels
torch.manual_seed(42)  # setting random seed

# Create output folders
output_img_dir = Path("results/images")
output_img_dir.mkdir(parents=True, exist_ok=True)
output_txt_path = Path("results/results.txt")

# Clear or create the results.txt file
with open(output_txt_path, 'w') as f:
    f.write("idx, ground_truth, predicted\n")

rows, cols = 2, 6
fig = plt.figure(figsize=(12, 4))

for i in range(1, (rows * cols) + 1):
    random_idx = torch.randint(0, len(test_dataset), size=[1]).item()
    img, label_gt = test_dataset[random_idx]
    # print(img)
    img_tensor = img.unsqueeze(dim=0).to(device)

    # get the stuff with the highest confidence(?)
    # label_pred = torch.argmax(net(img_tensor)).item()
    with torch.no_grad():
        output = net(img_tensor)
        prob = torch.softmax(output, dim=1)[0]  # shape: [10]
        label_pred = torch.argmax(prob).item()
    # === Save image to PNG ===
    img_filename = f"img_{i:02d}.png"
    save_image(img, output_img_dir / img_filename)

    # === Write result to text file ===
    with open(output_txt_path, 'a') as f:
        # f.write(f"{img_filename}, {class_names[label_gt]}, {class_names[label_pred]}\n")
        prob_str = ", ".join([f"{class_names[i]}: {prob[i]:.4f}" for i in range(10)])
        f.write(f"{img_filename}, GT: {class_names[label_gt]}, Pred: {class_names[label_pred]} → [{prob_str}]\n")    # === Plot image ===
    fig.add_subplot(rows, cols, i)
    img_display = img.permute(1, 2, 0)  # CxHxW → HxWxC
    plt.imshow(img_display, cmap='gray')
    color = 'g' if label_pred == label_gt else 'r'
    plt.title(class_names[label_pred], color=color)
    plt.axis('off')

plt.tight_layout()
plt.show()


# rows, cols = 2, 6
# for i in range(1, (rows * cols) + 1):
#     random_idx = torch.randint(0, len(test_dataset), size=[1]).item()
#     img, label_gt = test_dataset[random_idx]
#     img_temp = img.unsqueeze(dim=0).to(device)
#     # print(img.shape)
#     label_pred = torch.argmax(net(img_temp))
#     fig.add_subplot(rows, cols, i)
#     img = img.permute(1, 2, 0)    # CWH --> WHC
#     plt.imshow(img, cmap='gray')
#     if label_pred == label_gt:
#         plt.title(class_names[label_pred], color='g') # for correct prediction
#     else:
#         plt.title(class_names[label_pred], color='r') # for incorrect prediction
# plt.axis(False)
# plt.tight_layout()
# plt.show()
