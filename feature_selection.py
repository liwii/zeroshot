import torch.nn as nn
import torch
from train import AnimalsDataset, NUM_PREDS, prediction
from torchvision import models, transforms
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import numpy as np
import sys

def select_features(model, dataloaders, device, matrix):
    predicates = 85

    all_labels = {
        'train': np.empty((0)),
        'val': np.empty((0))
    }

    all_outputs = {
        'train': np.empty((0, predicates)),
        'val': np.empty((0, predicates))
    }


    for phase in ['train', 'val']:

        all_data = len(dataloaders[phase].dataset)

        done = 0
    
        for inputs, predicates, labels in dataloaders[phase]:
            sys.stdout.write("\r{}/{}".format(done, all_data))
            inputs = inputs.to(device)
            predicates = predicates.to(device)

            with torch.no_grad():
                outputs = model(inputs)
                sigmoid = nn.Sigmoid()
                outputs = sigmoid(outputs)
                all_labels[phase] = np.concatenate([all_labels[phase], labels.numpy()])
                all_outputs[phase] = np.concatenate([all_outputs[phase], outputs.numpy()])

            done += len(inputs)

    torch.save(all_labels, 'all_labels.pth')
    torch.save(all_outputs, 'all_outputs.pth')

    clf = ExtraTreesClassifier(n_estimators=100)
    clf = clf.fit(all_outputs['train'], all_labels['train'])
    torch.save(clf, 'forest.pth')

    features_list = []
    train_acc = []
    val_acc = []

    matrix = np.array(matrix)
    for i in range(1, 9):
        features = i * 10
        selector = SelectFromModel(clf, threshold=-np.inf, max_features=features)
        matrix_selected = selector.transform(matrix)
        outputs_selected = selector.transform(all_outputs['train'])
        predicted_train = prediction(matrix_selected, outputs_selected)
        acc_train = np.sum(all_labels['train'] == predicted_train) / len(all_labels['train'])
        outputs_selected_val = selector.transform(all_outputs['val'])
        predicted_val = prediction(matrix_selected, outputs_selected_val)
        acc_val = np.sum(all_labels['val'] == predicted_val) / len(all_labels['val'])
        print('With {} features, Training Accuracy: {}, Validation Accuracy: {}'.format(features, acc_train, acc_val))
        features_list.append(features)
        train_acc.append(acc_train)
        val_acc.append(acc_val)

    return features_list, train_acc, val_acc


def main(model):
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
    model_ft.load_state_dict(torch.load(model))
    model_ft.eval()

    dataset = AnimalsDataset(images_file="train_images.txt", classes=train_classes, matrix=train_matrix, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]))
    train_len = int(0.8 * len(dataset))
    val_len = len(dataset) - train_len
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_len, val_len])

    batch_size = 16
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    dataloaders = {
        'train': train_loader,
        'val': val_loader,
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = model_ft.to(device)

    features_list, train_acc, val_acc = select_features(model_ft, dataloaders, device, train_matrix)

    plt.title("Accuracy with different number of features")
    plt.xlabel("Selected Features")
    plt.ylabel("Accuracy")

    plt.plot(features_list, train_acc, label="train")
    plt.plot(features_list, val_acc, label="val")
    plt.xticks(np.arange(10, 90, 10))
    plt.savefig('select_features.png')

if __name__ == '__main__':
    model = sys.argv[1]
    main(model)