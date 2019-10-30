import numpy as np
import matplotlib.pyplot as plt
from models.base_model import BaseModel
from sklearn.preprocessing import MinMaxScaler

class LinReg(BaseModel):
    def __init__(self, config, dataset):
        super(LinReg, self).__init__(config, dataset)
        '''
        This simple Linear Regression module is for demonstrative example.
        The model performance will not be guaranteed
        '''


    def set_params(self):
        self.w = self.w_init()
        self.alpha = self.config.linreg_alpha

    def w_init(self) :
        if self.config.linreg_w_init == 'uniform' :
            weight = np.random.uniform(-2,2,size=[22, 2])

        return weight

    def preprocess(self):
        use_cols = ['FTHG','FTAG','home_wins', 'home_draws', 'home_losses', 'home_goals', 'home_oppos_goals',
                    'home_shots', 'home_oppos_shots', 'home_shotontarget', 'home_oppos_shotontarget',
                    'away_wins', 'away_draws', 'away_losses', 'away_goals', 'away_oppos_goals', 'away_shots',
                    'away_oppos_shots', 'away_shotontarget', 'away_oppos_shotontarget','Hodds','Dodds','Aodds']

        train = self.dataset.train_set[use_cols]
        test = self.dataset.test_set[use_cols]

        # separate X, Y Dataset
        trainY = np.array(train.iloc[:,:2])
        trainX = np.array(train.iloc[:,2:])
        testY = np.array(test.iloc[:,:2])
        testX = np.array(test.iloc[:,2:])

        # apply min max scaling
        scaler = MinMaxScaler()
        trainX = scaler.fit_transform(trainX)
        testX = scaler.transform(testX)

        # add ones for bias term
        trainX = np.concatenate((trainX, np.ones([trainX.shape[0],1])), axis=1)
        testX = np.concatenate((testX, np.ones([testX.shape[0],1])), axis=1)

        # save it in the dataset object
        self.dataset.trainX = trainX
        self.dataset.trainY = trainY
        self.dataset.testX = testX
        self.dataset.testY = testY

    def train(self):
        iter = 0
        losses = []
        past_MSE = [9999999,9999999]
        while True :
            iter+=1
            out = np.matmul(self.dataset.trainX, self.w)
            error_matrix = (out - self.dataset.trainY)**2
            MSE = 1/(2*out.shape[0]) * np.sum(error_matrix, axis=0)
            gradient = 1/out.shape[0] * np.matmul(np.transpose(out - self.dataset.trainY), self.dataset.trainX)

            self.w = self.w - self.alpha * np.transpose(gradient)
            print('iter {} : Home loss - {}  /  Away loss - {}'.format(iter, MSE[0],MSE[1]))

            loss_avg = (MSE[0] + MSE[1])/2
            losses.append(loss_avg)

            if np.mean(past_MSE - MSE) < self.config.linreg_tolerance :
                break
            past_MSE = MSE.copy()

        # visualize descending loss value
        plt.plot(losses)
        plt.title('linear regression gradient descent loss')
        plt.ylabel('avg MSE')
        plt.xlabel('epoch')
        plt.savefig('./out/linreg_loss.png')
        plt.close()




    def predict(self):
        '''
        Refers to parameters and produce predictions
        :return: predicted values for train and test data
        '''
        train_out = np.round(np.matmul(self.dataset.trainX, self.w))
        test_out = np.round(np.matmul(self.dataset.testX, self.w))

        return train_out, test_out

