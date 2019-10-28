

class BaseModel():
    def __init__(self, config, dataset):
        self.config = config
        self.dataset = dataset

        self.set_params()

    def set_params(self):
        print('parameters are not initialized for model')
        raise NotImplementedError

    def preprocess(self):
        '''
        Should alter the dataset.train_set and dataset.test_set to the final data format used for training

        self.dataset.trainX = ?
        self.dataset.trainY = ?
        self.dataset.testX = ?
        self.dataset.testY = ?
        '''
        print('preprocessing logic is not implemented')
        raise NotImplementedError

    def train(self):
        print('train logic is not implemented')
        raise NotImplementedError

    def predict(self):
        print('prediction logic is not implemented')
        raise NotImplementedError