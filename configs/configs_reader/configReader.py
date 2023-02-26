class ConfigReader():

    def __init__(self, data):

        # Always requirement
        self.model_type = data.get("type")
        self.input_shape = data.get("input_shape")
        self.num_classes = data.get("num_classes")
        self.save_path = data.get("save_path")
        self.dataset_path = data.get("dataset_path")

        # Default value
        self.learning_rate = data.get("learning_rate", 0.01)
        self.batch_size = data.get("batch_size", 1)
        self.epochs = data.get("epochs", 10)
        self.save_mode = data.get("save_mode", "last_model")

class NNConfigReader(ConfigReader):

    def __init__(self,data) -> None:
        super(NNConfigReader,self).__init__(data)

class CNNConfigReader(ConfigReader):

    def __init__(self,data) -> None:
        super(CNNConfigReader,self).__init__(data)

# Config Map
#----------------------------------------------------------------------------------------------------------
config_map = {
    "NN":NNConfigReader,
    "CNN":CNNConfigReader
}