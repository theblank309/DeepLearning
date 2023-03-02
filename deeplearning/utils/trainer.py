import traceback
import warnings
import torch

from utils.progressbar import ProgressBar
from utils import modelUtils

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Exception handling
# --------------------------------------------------------------------------------------------------
def Error_Handler(func):
    def Inner_Function(*args, **kwargs):
        try:
            progressbar = ProgressBar()
            args = tuple(list(args) + [progressbar])
            model = func(*args, **kwargs)
            return model
        except Exception as error:
            progressbar.failed()
            traceback.print_exc()
    return Inner_Function

# Select model using save mode
# --------------------------------------------------------------------------------------------------
def select_checkpoint(selected_model, current_model, selected_loss, current_loss, epoch):
    if current_loss < selected_loss or epoch == 0:
        return current_model, current_loss
    return selected_model, selected_loss

# Train model (Intialized model, loss and optimizer)
# --------------------------------------------------------------------------------------------------
@Error_Handler
def train_model(model, criterion, optimizer, train_loader, param, progressbar):

    selected_model = None
    selected_loss = 0
    model_dir = modelUtils.get_model_savepath(param)
    
    for epoch in range(param.epochs):
        batches_acc = []
        batches_loss = []
        progressbar.pbar(epoch+1, param.epochs, len(train_loader))

        for batch_idx, (x,y) in enumerate(train_loader):

            # Get data to cuda if possible
            data = x.to(device=device)
            targets = y.to(device=device)

            # Forward
            scores = model(data)
            loss = criterion(scores, targets)
            batches_loss.append(loss.item())

            # Backward
            optimizer.zero_grad()
            loss.backward()

            # Gradient descent or adam step
            optimizer.step()

            batch_acc = check_accuracy(model, [(x,y)])
            batches_acc.append(batch_acc)
            progressbar.update(loss.item(), batch_acc)
        
        final_acc = sum(batches_acc)/len(batches_acc)
        final_loss = sum(batches_loss)/len(batches_loss)

        progressbar.update(final_loss, final_acc, update_value=0)
        progressbar.close()

        selected_model, selected_loss = select_checkpoint(selected_model, model, selected_loss, final_loss, epoch)
        modelUtils.save_checkpoint(selected_model, epoch, model_dir)
    modelUtils.save_checkpoint(model, "last_model", model_dir)

    return model

# Check accuracy of model
# --------------------------------------------------------------------------------------------------
def check_accuracy(model, loader):

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        # Loop through the data
        for x, y in loader:

            # Move data to device
            x = x.to(device=device)
            y = y.to(device=device)

            # Forward pass
            scores = model(x)
            _, predictions = scores.max(1)

            # Calculate
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    model.train()
    return float(num_correct) / num_samples * 100