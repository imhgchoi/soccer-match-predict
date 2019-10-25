

class BaseModel():
    def __init__(self, config, dataset):
        self.config = config
        self.dataset = dataset

        self.set_params()

    def set_params(self):
        print('parameters are not initialized for model')
        raise NotImplementedError

    def train(self):
        print('train logic is not implemented')
        raise NotImplementedError

    def predict(self, data):
        print('prediction logic is not implemented')
        raise NotImplementedError