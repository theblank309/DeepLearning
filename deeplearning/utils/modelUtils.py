import torch
from models.NN import NN
from models.CNN import CNN

model_map = {
    "NN":NN,
    "CNN":CNN
}

# Save model checkpoint
# --------------------------------------------------------------------------------------------------
def save_checkpoint(model, param):
    checkpoint = {"state_dict":model.state_dict()}
    save_path = f"{param.save_path}\\{param.model_type}_{param.save_mode}.pth"
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