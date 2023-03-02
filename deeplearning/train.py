import json

import torch
import torch.optim as optim
import torchvision.transforms as transforms

from models.NN import NN
from models.CNN import CNN
from utils import datasetUtils
from utils import trainer
from dataset_reader.customDatasetImages import CustomDatasetImages
from configs.configs_reader.configReader import config_map

import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# seperator for print statement
sepr = [80,"-"]

# Hyper parameter 
# --------------------------------------------------------------------------------------------------
def load_hyperparameter(path):
    with open(path,"r+") as file:
        data = json.load(file)
    config = config_map[data['type']](data)

    print(f"\n{sepr[1]*sepr[0]}\nHyperparameters: \n{sepr[1]*sepr[0]}")
    for key,value in config.__dict__.items():
        print(f"{key}: {value}")
    print(f"{sepr[1]*sepr[0]}\n")
    return config

# --------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    path = r"C:\Siddhesh\Programming\Machine Learning\DeepLearning\deeplearning\configs\hyperparameter.json"
    param = load_hyperparameter(path)

    # Load train & test data
    image_size = tuple(param.input_shape[:-1])
    transform = transforms.Compose([
        transforms.Resize(image_size), 
        transforms.ToTensor(),
        transforms.ColorJitter(brightness=0.5),
        transforms.RandomHorizontalFlip(p=0.5)
    ])
    dataset = CustomDatasetImages(root_dir=param.dataset_path, transform=transform)
    train_loader,test_loader = datasetUtils.load_custom_data(param.batch_size, dataset, 0.85)

    # Define Model
    model = CNN(param.input_shape, param.num_classes, param.feature_extractor).to(device)

    # Loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=param.learning_rate)
    model = trainer.train_model(model, criterion, optimizer, train_loader, param)

    print(f"Accuracy on training set: {trainer.check_accuracy(model, train_loader):.2f}")
    print(f"Accuracy on testing set: {trainer.check_accuracy(model, test_loader):.2f}")