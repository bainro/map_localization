import os
import time
import torch
import torchvision
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from dataloader import LocalizeDataset
from tempfile import TemporaryDirectory
from torchvision import datasets, models
from torchvision import transforms as T

cudnn.benchmark = True

train_trans = T.Compose([
    T.Resize(224, antialias=True),
    # T.RandomResizedCrop(224, antialias=True),
    # T.RandomHorizontalFlip(),
    T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    T.ColorJitter(brightness=.5, hue=.3),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
val_trans = T.Compose([
    T.Resize(224, antialias=True),
    # T.CenterCrop(224),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

image_datasets = {}
image_datasets['train'] = LocalizeDataset('./data/nongen', train=True, transform=train_trans, target_size=256)
image_datasets['val'] = LocalizeDataset('./data/nongen', train=False, transform=val_trans, target_size=256)
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

assert torch.cuda.is_available(), "GPU not found!"
device = torch.device("cuda:0")

def imshow(inp, title=None):
    """Display image for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# Get a batch of training data
inputs, _x, _y = next(iter(dataloaders['train']))

# Make a grid from batch
inputs = inputs[:8,...]
out = torchvision.utils.make_grid(inputs)

imshow(out, title="Verify the images loaded correctly!")
plt.show()

def train_model(model, criterion, optimizer, scheduler, num_epochs=1):
    since = time.time()

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_loss = 1e8

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0

                # Iterate over data.
                for inputs, x, y in dataloaders[phase]:
                    inputs = inputs.to(device)
                    x = x.to(device)
                    y = y.to(device)
                    labels = torch.stack([x, y])
                    labels = torch.squeeze(labels, -1)
                    labels = torch.transpose(labels, 0, 1)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                if phase == 'train' and scheduler != None:
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f}')

                # deep copy the model
                if phase == 'val' and epoch_loss < best_loss:
                    best_loss = epoch_loss
                    torch.save(model.state_dict(), best_model_params_path)

            print("")

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))
    return model

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, x, y) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            x = x.to(device)
            y = y.to(device)
            labels = torch.stack([x, y])
            labels = torch.squeeze(labels, -1)
            labels = torch.transpose(labels, 0, 1)

            outputs = model(inputs)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {outputs[j]}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

model_conv = torchvision.models.resnet18(weights='IMAGENET1K_V1')
for param in model_conv.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Sequential(
     nn.Linear(num_ftrs, 2), 
     nn.Sigmoid()
)

model_conv = model_conv.to(device)

criterion = nn.MSELoss()

# Observe that only parameters of final layer are being optimized as
# opposed to before.
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.01, momentum=0.9)
# optimizer_conv = optim.Adam(model_conv.fc.parameters(), lr=0.1)

# Decay LR by a factor of 0.1 every 7 epochs
#lr_scheduler = None
lr_schedule = lr_scheduler.StepLR(optimizer_conv, step_size=6, gamma=0.1)

model_conv = train_model(model_conv, criterion, optimizer_conv,
                         lr_schedule, num_epochs=1)

visualize_model(model_conv)
plt.show()
