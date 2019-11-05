

class BaseModel():
    def __init__(self, config, dataset):
        self.config = config
        self.dataset = dataset

        self.set_params()

    def set_params(self):
        raise NotImplementedError('parameters are not initialized for model')

    def preprocess(self):
        '''
        Should alter the dataset.train_set and dataset.test_set to the final data format used for training

        self.dataset.trainX = ?
        self.dataset.trainY = ?
        self.dataset.testX = ?
        self.dataset.testY = ?
        '''
        raise NotImplementedError('preprocessing logic is not implemented')

    def train(self):
        raise NotImplementedError('train logic is not implemented')

    def predict(self):
        raise NotImplementedError('prediction logic is not implemented')