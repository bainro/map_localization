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
from torchvision import transforms as T
from torchvision import datasets, models


cudnn.benchmark = True

train_trans = T.Compose([
    T.Resize(224, antialias=True),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
val_trans = T.Compose([
    T.Resize(224, antialias=True),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

image_datasets = {}
image_datasets['train'] = LocalizeDataset('./data/nongen', train=True, transform=train_trans)
image_datasets['val'] = LocalizeDataset('./data/nongen', train=False, transform=val_trans)
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
    plt.pause(0.001)  # pause a bit for the plots

# Get a batch of training data
inputs, _x, _y = next(iter(dataloaders['train']))
inputs = inputs[:8,...]
out = torchvision.utils.make_grid(inputs)
imshow(out, title="Verify the images loaded correctly!")
plt.show()

def train_model(model, criterion, optimizer, scheduler, num_epochs=1):
    start_time = time.time()
    best_loss = 1e8
    best_model_params_path = "./best_model_params.pt"

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
                labels = torch.stack([x, y], dim=1)
                labels = torch.squeeze(labels, -1)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    # backward
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
                print("new best!")

        print("")

    time_elapsed = time.time() - start_time
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    # load best model weights
    model.load_state_dict(torch.load(best_model_params_path))
    return model

def visualize_model(model, num_images=1):
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

            '''
            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//4, 4, images_so_far)
                ax.axis('off')
                o_x, o_y = outputs[j].cpu().tolist()
                gt_x, gt_y = labels[j].cpu().tolist()
                ax.set_title(f'Predicted: ({o_x:.3f},{o_y:.3f}) \nGT: ({gt_x:.3f},{gt_y:.3f})')
                imshow(inputs.cpu().data[j])
           '''

            f, (ax1, ax2) = plt.subplots(1, 2, width_ratios=[1, 3])
            ax1.axis('off')
            o_x, o_y = outputs[0].cpu().tolist()
            gt_x, gt_y = labels[0].cpu().tolist()
            ax1.set_title("First Person View")
            imshow(inputs.cpu().data[0])
            
            #ax = plt.subplot(1, 2, 2, width_ratios=[1, 4])
            ax2.axis('off')
            o_x, o_y = outputs[0].cpu().tolist()
            gt_x, gt_y = labels[0].cpu().tolist()
            map_img = plt.imread('./data/2nd_blueprint.png')
            plt.imshow(map_img, 
                       resample=False, 
                       interpolation='none', 
                       cmap='gray', 
                       vmin=0, 
                       vmax=255)

            # denormalize (i.e. convert back to px space)
            o_x = o_x * map_img.shape[1]
            o_y = o_y * map_img.shape[0]
            gt_x = gt_x * map_img.shape[1]
            gt_y = gt_y * map_img.shape[0]
            plt.scatter(x=o_x, y=o_y, c=[[1.,0.,0.,1.]], s=18)
            plt.scatter(x=gt_x, y=gt_y, c=[[0.,1.,0.,1.]], s=18)
            plt.pause(0.001)
            if True: return
            
            '''
                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
            '''
        model.train(mode=was_training)

model_conv = torchvision.models.resnet18(weights='IMAGENET1K_V1')
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Sequential(
     nn.Linear(num_ftrs, 2), 
     nn.Sigmoid(),
)

model_conv = model_conv.to(device)

# def weighted_mse_loss(input, target, weight):
    # return torch.sum(weight * (input - target) ** 2)

criterion = nn.MSELoss()
optimizer_conv = optim.SGD(model_conv.parameters(), lr=0.1, momentum=0.9)
lr_schedule = lr_scheduler.StepLR(optimizer_conv, step_size=45, gamma=0.1)
model_conv = train_model(model_conv, criterion, optimizer_conv,
                         lr_schedule, num_epochs=0)

visualize_model(model_conv)
plt.show()
