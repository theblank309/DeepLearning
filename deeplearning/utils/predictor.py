# Predict output through model
# --------------------------------------------------------------------------------------------------
def predict_output(model, data):
    model.eval()
    data = data.to(device="cuda")
    output = model(data)
    model.train()
    return output