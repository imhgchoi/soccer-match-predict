


class Trainer():
    def __init__(self, config, model):
        self.config = config
        self.model = model

    def train(self):
        self.model.preprocess()
        self.model.train()
        return self.model