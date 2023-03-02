import json
from configs.configs_reader.configReader import config_map

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