# import

import os
import torch
from torch import nn

import torchvision
from torchvision import models
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

# import torchmetrics
from torchmetrics.classification import MulticlassAccuracy

# device Agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load dataset
# Define main data directory
DATA_DIR = 'tiny-imagenet-200' # Original images come in shapes of [3,64,64]

# Define training and validation data paths
TRAIN_DIR = os.path.join(DATA_DIR, 'train') 
VALID_DIR = os.path.join(DATA_DIR, 'val')


# Create separate validation subfolders for the validation images based on
# their labels indicated in the val_annotations txt file
val_img_dir = os.path.join(VALID_DIR, 'images')

# Open and read val annotations text file
fp = open(os.path.join(VALID_DIR, 'val_annotations.txt'), 'r')
data = fp.readlines()

# Create dictionary to store img filename (word 0) and corresponding
# label (word 1) for every line in the txt file (as key value pair)
val_img_dict = {}
for line in data:
    words = line.split('\t')
    val_img_dict[words[0]] = words[1]
fp.close()

# # Display first 10 entries of resulting val_img_dict dictionary
# {k: val_img_dict[k] for k in list(val_img_dict)[:10]}

# Create subfolders (if not present) for validation images based on label,
# and move images into the respective folders
for img, folder in val_img_dict.items():
    newpath = (os.path.join(val_img_dir, folder))
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    if os.path.exists(os.path.join(val_img_dir, img)):
        os.rename(os.path.join(val_img_dir, img), os.path.join(newpath, img))

image_transform_pretrain = transforms.Compose([
                transforms.Resize(128), # Resize images to 128 x 128
                # transforms.CenterCrop(56), # Center crop image
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),  # Images to tensors
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])
])


train_dataset = datasets.ImageFolder(TRAIN_DIR, transform = image_transform_pretrain)
valid_dataset = datasets.ImageFolder(val_img_dir, transform = image_transform_pretrain)


num_classes = len(train_dataset.classes)
# num_classes

# Define batch size for DataLoaders
BATCH_SIZE = 32

# Create DataLoaders for pre-trained models (normalized based on specific requirements)
train_dataloader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True, drop_last = True)

valid_dataloader = DataLoader(valid_dataset, batch_size = BATCH_SIZE, drop_last = True)

# print(len(train_dataloader), len(valid_dataloader))



# Build model

# Loss and acuuracy function
# Loss
loss_fn = nn.CrossEntropyLoss()

# Accuracy
accuracy_fn = MulticlassAccuracy(num_classes = num_classes).to(device)


# Train and Test loop
# train
def train_loop(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer, 
               accuracy_fn, device: torch.device = device):
  
  model.train()

  train_loss, train_acc = 0, 0

  for batch, (x_train, y_train) in enumerate(dataloader):

    if device == 'cuda':
      x_train, y_train = x_train.to(device), y_train.to(device)

    
    # 1. Forward
    y_pred = model(x_train)

    # 2. Loss
    loss = loss_fn(y_pred, y_train)
    acc = accuracy_fn(y_train, torch.argmax(y_pred, dim = 1))
    # print("train")
    # print(f"acutal: {y_train}")
    # print(f"pred: {torch.argmax(y_pred, dim = 1)}")
    train_loss += loss
    train_acc += acc

    # 3. optimizer zero grad
    optimizer.zero_grad()

    # 4. Backward
    loss.backward()

    # 5. optimizer step
    optimizer.step()

  train_loss /= len(dataloader)
  train_acc /= len(dataloader)

  return train_loss, train_acc


def test_loop(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module, accuracy_fn, device: torch.device = device):
  test_loss, test_acc = 0, 0

  model.eval()
  with torch.inference_mode():
    for x_test, y_test in dataloader:
      if device == 'cuda':
        x_test, y_test = x_test.to(device), y_test.to(device)

      # 1. Forward
      test_pred = model(x_test)

      # 2. Loss
      test_loss += loss_fn(test_pred, y_test)
      test_acc += accuracy_fn(y_test, torch.argmax(test_pred, dim = 1))
      # print("test")
      # print(f"acutal: {y_test}")
      # print(f"pred: {torch.argmax(test_pred, dim = 1)}")

    test_loss /= len(dataloader)
    test_acc /= len(dataloader)

  return test_loss, test_acc

# plot the loss and accuracy 
def plotplot(train_losses, test_losses, train_acces, test_acces, file_name):
  plt.figure(figsize = (25,8))
  # plt.plot(range(1, epoches*(len(train_dataloader)*BATCH_SIZE) + 1), train)
  plt.subplot(1,2,1)
  plt.plot(range(len(train_losses)),train_losses, label = "Train Loss")
  plt.plot(range(len(test_losses)),test_losses, label = "Test Loss")
  plt.xlabel("Epoches")
  plt.ylabel("Loss")
  plt.title("Loss vs Epoches")
  plt.legend()

  plt.subplot(1,2,2)
  plt.plot(range(len(train_acces)),train_acces, label = "Train Accuracy")
  plt.plot(range(len(test_acces)),test_acces, label = "Test Accuracy")
  plt.xlabel("Epoches")
  plt.ylabel("Accuracy")
  plt.title("Accuracy vs Epoches")
  plt.legend()

  plt.savefig(file_name)


# model define
# with random weights
# resnet50 = models.resnet50(pretrained = False, progress = False, num_classes = num_classes).to(device)
# resnet50

print()
print("\nRANDOM WEIGHTS")
torch.manual_seed(64)
torch.cuda.manual_seed(64)


resnet50 = models.resnet50(pretrained = False, progress = False, num_classes = num_classes).to(device)
# resnet50


# optimizer function
optimizer = torch.optim.Adam(params = resnet50.parameters(), lr = 1e-3)

train_losses, test_losses = [], []
train_acces, test_acces = [], []

# train model
epoches = 11

torch.manual_seed(64)
torch.cuda.manual_seed(64)
for epoch in tqdm(range(epoches)):
  print()
  print(f"Epoch: {epoch + 1}")
  train_loss, train_acc = train_loop(model = resnet50, dataloader = train_dataloader,
                                     loss_fn = loss_fn, optimizer = optimizer, 
                                     accuracy_fn = accuracy_fn, device = device)
  test_loss, test_acc = test_loop(model = resnet50, dataloader = valid_dataloader,
                                  loss_fn = loss_fn, accuracy_fn = accuracy_fn,
                                  device = device)
  print(f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}  ||  Train Accuracy: {train_acc:.4f} | Test Accuracy: {test_acc:.4f}")

  train_losses.append(train_loss.item())
  test_losses.append(test_loss.item())
  train_acces.append(train_acc.item())
  test_acces.append(test_acc.item())


plotplot(train_losses, test_losses, train_acces, test_acces, "random_weights.jpg")


# with Xavier initilization
# x_resnet50 = models.resnet50(pretrained = False, progress = False, num_classes = num_classes).to(device)
# for name, param in x_resnet50.named_parameters():
#     if 'weight' in name:
#         torch.nn.init.xavier_uniform_(param)
# x_resnet50

print()
print("\nXAVIER WEIGHTS")
torch.manual_seed(64)
torch.cuda.manual_seed(64)


x_resnet50 = models.resnet50(pretrained = False, progress = False, num_classes = num_classes).to(device)
for name, param in x_resnet50.named_parameters():
    
    if 'weight' in name:
        torch.nn.init.xavier_uniform_(param.unsqueeze(0))
# resnet50



# optimizer function
optimizer = torch.optim.Adam(params = x_resnet50.parameters(), lr = 1e-3)

train_losses, test_losses = [], []
train_acces, test_acces = [], []

# train model
epoches = 11

torch.manual_seed(64)
torch.cuda.manual_seed(64)
for epoch in tqdm(range(epoches)):
  print()
  print(f"Epoch: {epoch + 1}")
  train_loss, train_acc = train_loop(model = x_resnet50, dataloader = train_dataloader,
                                     loss_fn = loss_fn, optimizer = optimizer, 
                                     accuracy_fn = accuracy_fn, device = device)
  test_loss, test_acc = test_loop(model = x_resnet50, dataloader = valid_dataloader,
                                  loss_fn = loss_fn, accuracy_fn = accuracy_fn,
                                  device = device)
  print(f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}  ||  Train Accuracy: {train_acc:.4f} | Test Accuracy: {test_acc:.4f}")

  train_losses.append(train_loss.item())
  test_losses.append(test_loss.item())
  train_acces.append(train_acc.item())
  test_acces.append(test_acc.item())


plotplot(train_losses, test_losses, train_acces, test_acces, "xavier_weights.jpg")


# with He initilization
# h_resnet50 = models.resnet50(pretrained = False, progress = False, num_classes = num_classes).to(device)
# for name, param in h_resnet50.named_parameters():
#     if 'weight' in name:
#         torch.nn.init.kaiming_uniform_(param, a=0, mode='fan_in', nonlinearity='relu')
# h_resnet50

print()
print("\nHE WEIGHTS")
torch.manual_seed(64)
torch.cuda.manual_seed(64)


h_resnet50 = models.resnet50(pretrained = False, progress = False, num_classes = num_classes).to(device)
for name, param in h_resnet50.named_parameters():
    if 'weight' in name:
        torch.nn.init.kaiming_uniform_(param.unsqueeze(0), a=0, mode='fan_in', nonlinearity='relu')
# resnet50



# optimizer function
optimizer = torch.optim.Adam(params = h_resnet50.parameters(), lr = 1e-3)

train_losses, test_losses = [], []
train_acces, test_acces = [], []

# train model
epoches = 11

torch.manual_seed(64)
torch.cuda.manual_seed(64)
for epoch in tqdm(range(epoches)):
  print()
  print(f"Epoch: {epoch + 1}")
  train_loss, train_acc = train_loop(model = h_resnet50, dataloader = train_dataloader,
                                     loss_fn = loss_fn, optimizer = optimizer, 
                                     accuracy_fn = accuracy_fn, device = device)
  test_loss, test_acc = test_loop(model = h_resnet50, dataloader = valid_dataloader,
                                  loss_fn = loss_fn, accuracy_fn = accuracy_fn,
                                  device = device)
  print(f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}  ||  Train Accuracy: {train_acc:.4f} | Test Accuracy: {test_acc:.4f}")

  train_losses.append(train_loss.item())
  test_losses.append(test_loss.item())
  train_acces.append(train_acc.item())
  test_acces.append(test_acc.item())


plotplot(train_losses, test_losses, train_acces, test_acces, "he_weights.jpg")