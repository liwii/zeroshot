import torch.nn as nn
import torch
from train import AnimalsDataset, NUM_PREDS, prediction
from torchvision import models, transforms

def predict(model, dataloader, device, matrix, classes):
    result = []

    for inputs, predicates, labels in dataloader:
        inputs = inputs.to(device)
        predicates = predicates.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            sigmoid = nn.Sigmoid()
            outputs = sigmoid(outputs)
            preds = prediction(matrix, outputs)

        for pred in preds:
            c = classes[pred.item()]
            print(c)
            result.append(c)

    return result

def main():
    text = open('predicate-matrix-binary.txt').read()
    rows = text.split("\n")[:-1]
    matrix = [list(map(int, row.split(' '))) for row in rows]

    classes = {}
    classes_text = open('classes.txt').read()
    for i, row in enumerate(classes_text.split('\n')[:-1]):
        _, animal = row.split('\t')
        classes[animal] = i

    test_classes_text = open('testclasses.txt').read()

    test_matrix = []
    test_classes = test_classes_text.split('\n')[:-1]
    test_classes_dict = {}
    for i, c in enumerate(test_classes):
        test_classes_dict[c] = i
        test_matrix.append(matrix[classes[c]])
    
    model_ft = models.resnet18(pretrained=True)
    for name, param in model_ft.named_parameters():
        param.requires_grad = False


    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, NUM_PREDS)
    model_ft.load_state_dict(torch.load('model.pth'))
    model_ft.eval()

    images = []
    ifile = open('test_images.txt', "r")
    for line in ifile:
        image, _ = line.split(" ")
        images.append(image)

    dataset = AnimalsDataset(images_file="test_images.txt", classes=test_classes_dict, matrix=test_matrix, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]))

    batch_size = 16
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = model_ft.to(device)

    predicted = predict(model_ft, dataloader, device, test_matrix, test_classes)

    with open("test_images_predicted", "w") as fp:
        for image, c in zip(images, predicted):
            fp.write("{}\t{}\n".format(image, c))

if __name__ == '__main__':
    main()