import numpy as np
from models.base_model import BaseModel
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class KNNRegressor(BaseModel):
    def __init__(self, config, dataset):
        super(KNNRegressor, self).__init__(config, dataset)


    def set_params(self):
        self.k = self.config.knn_k

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

    def train(self):
        self.reference = self.dataset.trainX
        self.reference_label = self.dataset.trainY


    def predict(self):
        train_out = []
        for row in self.dataset.trainX :
            euc_dists = np.linalg.norm(self.dataset.trainX - row, axis=1)
            indices = np.argpartition(euc_dists, self.k)[:self.k]
            k_predictions = self.dataset.trainY[indices]
            avged_prediction = np.round(np.mean(k_predictions, axis=0))
            train_out.append(avged_prediction)

        test_out = []
        for row in self.dataset.testX:
            euc_dists = np.linalg.norm(self.dataset.trainX - row, axis=1)
            indices = np.argpartition(euc_dists, self.k)[:self.k]
            k_predictions = self.dataset.trainY[indices]
            avged_prediction = np.round(np.mean(k_predictions, axis=0))
            test_out.append(avged_prediction)

        return np.array(train_out), np.array(test_out)

