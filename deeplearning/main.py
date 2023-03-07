import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from models.CNN import CNN
from utils import others
from utils import trainer
from utils import datasetUtils
from utils import modelUtils
from dataset_reader.customDatasetImages import CustomDatasetImages

import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    path = r"C:\Siddhesh\Programming\Machine Learning\DeepLearning\deeplearning\configs\hyperparameter.json"
    param = others.load_hyperparameter(path)

    # # Load train & test data
    # image_size = tuple(param.input_shape[:-1])
    # transform = transforms.Compose([
    #     transforms.Resize(image_size), 
    #     transforms.ToTensor(),
    #     transforms.ColorJitter(brightness=0.5),
    #     transforms.RandomHorizontalFlip(p=0.5),
    #     transforms.RandomRotation(degrees=[30,45]),
    # ])
    # dataset = CustomDatasetImages(root_dir=param.dataset_path, transform=transform)
    # train_loader,test_loader = datasetUtils.load_custom_data(param.batch_size, dataset, 0.85)

    # # Define Model
    # model = CNN(param.input_shape, param.num_classes, param.feature_extractor).to(device)

    # # Loss and optimizer
    # criterion = torch.nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=param.learning_rate)
    # model = trainer.train_model(model, criterion, optimizer, train_loader, param)

    # print(f"Accuracy on training set: {trainer.check_accuracy(model, train_loader):.2f}")
    # print(f"Accuracy on testing set: {trainer.check_accuracy(model, test_loader):.2f}")

    data_path = r"C:\Siddhesh\Programming\Machine Learning\DeepLearning\dataset\birds_species_classification\val"
    model_path = r"C:\Siddhesh\Programming\Machine Learning\DeepLearning\saved_models\birds_species_classification_(02-03-2023-19-37-13)\model_24.pth"
    transform = transforms.Compose([
        transforms.Resize((256, 256)), 
        transforms.ToTensor()
    ])
    
    dataset = CustomDatasetImages(root_dir=data_path, transform=transform)
    val_loader = DataLoader(dataset, batch_size=2)
    model = modelUtils.load_checkpoint_CNN(model_path, param)
    print(f"Accuracy on training set: {trainer.check_accuracy(model, val_loader):.2f}")