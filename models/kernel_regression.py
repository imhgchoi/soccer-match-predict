import numpy as np
from models.base_model import BaseModel
from sklearn.preprocessing import MinMaxScaler
from scipy import stats


class KernelRegression(BaseModel):
    def __init__(self, config, dataset):
        super(KernelRegression, self).__init__(config, dataset)


    def set_params(self):
        self.sd = self.config.kernelreg_std

    def preprocess(self):
        use_cols = ['FTHG','FTAG','home_wins', 'home_draws', 'home_losses', 'home_goals', 'home_oppos_goals',
                    'home_shots', 'home_oppos_shots', 'home_shotontarget', 'home_oppos_shotontarget',
                    'away_wins', 'away_draws', 'away_losses', 'away_goals', 'away_oppos_goals', 'away_shots',
                    'away_oppos_shots', 'away_shotontarget', 'away_oppos_shotontarget',
                    'home_oppos_wins', 'home_oppos_draws', 'home_oppos_losses', 'home_fouls', 'home_yellowcards',
                    'home_redcards', 'home_cornerkicks', 'home_oppos_cornerkicks', 'home_oppos_fouls',
                    'home_oppos_yellowcards', 'home_oppos_redcards', 'away_fouls', 'away_yellowcards', 'away_redcards',
                    'away_cornerkicks', 'away_oppos_cornerkicks', 'away_oppos_fouls', 'away_oppos_yellowcards',
                    'away_oppos_redcards', 'Hodds', 'Dodds', 'Aodds']

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

        # save it in the dataset object
        self.dataset.trainX = trainX
        self.dataset.trainY = trainY
        self.dataset.testX = testX
        self.dataset.testY = testY

    def gaussianKernel(self, x_L2, sd):
        K = np.exp(-(x_L2/sd)**2)
        return K

    def train(self):
        self.reference = self.dataset.trainX
        self.reference_label = self.dataset.trainY


    def predict(self):
        train_out = []
        print('predicting train...')
        for idx_, row1 in enumerate(self.dataset.trainX) :
            y_sum = 0
            w_sum = 0
            for idx, row2 in enumerate(self.dataset.trainX) :
                if idx_ == idx :
                    continue
                kernel_dis= self.gaussianKernel(np.linalg.norm(row1 - row2, axis=0, ord=2), self.sd)
                y_i= kernel_dis*self.dataset.trainY[idx]
                y_sum = y_sum + y_i
                w_sum = w_sum + kernel_dis
            y_hat = np.round(y_sum/w_sum)
            train_out.append(y_hat)

        test_out = []
        print('predicting test...')
        for row1 in self.dataset.testX:
            y_sum = 0
            w_sum = 0
            for idx, row2 in enumerate(self.dataset.trainX) :
                kernel_dis= self.gaussianKernel(np.linalg.norm(row1 - row2, axis=0, ord=2), self.sd)
                y_i= kernel_dis*self.dataset.trainY[idx]
                y_sum = y_sum + y_i
                w_sum = w_sum + kernel_dis
            
            y_hat = np.round(y_sum/w_sum)
            test_out.append(y_hat)
        return np.array(train_out), np.array(test_out)


