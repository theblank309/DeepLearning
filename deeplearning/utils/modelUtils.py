import os
import torch
from models.NN import NN
from models.CNN import CNN

from datetime import datetime

model_map = {
    "NN":NN,
    "CNN":CNN
}

def get_model_savepath(param):
    dataset_name = os.path.basename(param.dataset_path)
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y-%H-%M-%S")
    model_dir = f"{dataset_name}_({dt_string})"
    os.makedirs(f"{param.save_path}\\{model_dir}", exist_ok=True)
    full_model_dir = f"{param.save_path}\\{model_dir}"
    return full_model_dir

# Save model checkpoint
# --------------------------------------------------------------------------------------------------
def save_checkpoint(model, epoch, full_model_dir):
    checkpoint = {"state_dict":model.state_dict()}
    save_path = f"{full_model_dir}\\model_{epoch}.pth"
    torch.save(checkpoint, save_path)

# Load model checkpoint
# --------------------------------------------------------------------------------------------------
def load_checkpoint(path, param):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path)
    print(f"Load Model Type: {param.model_type}")
    model_class = model_map[param.model_type]
    model = model_class(input_shape=param.input_shape, num_classes=param.num_classes).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    return model