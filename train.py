import os
import time
import copy
import torch
import sys
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset
from torchvision import models, transforms
from PIL import Image
from scipy import spatial

IMAGES_DIR = "JPEGImages_128x128/"
NUM_PREDS = 85
RESNET_INPUT_SIZE = 224

class AnimalsDataset(Dataset):
    def __init__(self, images_file, classes, matrix, transform=None):
        self.transform = transform
        self.classes = classes
        self.matrix = matrix
        self.images = []
        self.labels = []
        ifile = open(images_file, "r")
        for line in ifile:
            image, label = line.split(" ")
            self.images.append(image)
            self.labels.append(label[:-1])
        ifile.close()
        cfile = open(images_file, "r")

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):

        if type(idx) == torch.Tensor:
            idx = idx.item()

        img_name = os.path.join(IMAGES_DIR, self.images[idx])

        image = Image.open(img_name)
        label = self.labels[idx]
        label_num = self.classes[label]
        
        pred_vec = self.matrix[label_num]
        if self.transform:
             image = self.transform(image)
        return image, torch.FloatTensor(pred_vec), label_num

def prediction(matrix, predicates):
    tree = spatial.KDTree(matrix)
    predictions = []
    for p in predicates:
        _, pred = tree.query(p.detach())
        predictions.append(pred)
    return torch.IntTensor(predictions)
        
def train_model(model, dataloaders, criterion, optimizer, device, matrix, num_epochs=25):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0

            all_data = len(dataloaders[phase].dataset)
            done = 0
        
            for inputs, predicates, labels in dataloaders[phase]:
                sys.stdout.write("\r{}/{}".format(done, all_data))
                inputs = inputs.to(device)
                predicates = predicates.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    sigmoid = nn.Sigmoid()
                    outputs = sigmoid(outputs)
                    loss = criterion(outputs, predicates)
                    preds = prediction(matrix, outputs)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                done += len(inputs)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print() 

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model, val_acc_history

def main():
    text = open('predicate-matrix-binary.txt').read()
    rows = text.split("\n")[:-1]
    matrix = [list(map(int, row.split(' '))) for row in rows]

    classes = {}
    classes_text = open('classes.txt').read()
    for i, row in enumerate(classes_text.split('\n')[:-1]):
        _, animal = row.split('\t')
        classes[animal] = i

    train_classes_text = open('trainclasses.txt').read()

    train_matrix = []
    train_classes = {}
    for i, c in enumerate(train_classes_text.split('\n')[:-1]):
        train_classes[c] = i
        train_matrix.append(matrix[classes[c]])
    
    model_ft = models.resnet18(pretrained=True)
    for name, param in model_ft.named_parameters():
        param.requires_grad = False


    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, NUM_PREDS)

    dataset = AnimalsDataset(images_file="train_images.txt", classes=classes, matrix=matrix, transform=transforms.Compose([
        transforms.ToTensor()
    ]))
    train_len = int(0.8 * len(dataset))
    val_len = len(dataset) - train_len
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_len, val_len])

    batch_size = 4
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    dataloaders = {
        'train': train_loader,
        'val': val_loader,
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    params_to_update = []

    print("Params to learn:")

    model_ft = model_ft.to(device)
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t", name)

    num_epochs = 1
    criterion = nn.MSELoss()
    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    model_ft, hist = train_model(model_ft, dataloaders, criterion, optimizer_ft, device, matrix, num_epochs=num_epochs)
    ohist = [h.cpu().numpy() for h in hist]

    plt.title("Validation Accuracy vs. Number of Training Epochs")
    plt.xlabel("Training Epochs")
    plt.ylabel("Validation Accuracy")

    plt.plot(range(1, num_epochs + 1), ohist)
    plt.ylim((0, 1.))
    plt.xticks(np.arange(1, num_epochs+1, 1.0))
    plt.savefig('val_acc.png')
    torch.save(model_ft.state_dict, 'model.pth')

if __name__ == '__main__':
    main()