import numpy as np
import matplotlib.pyplot as plt
from models.base_model import BaseModel
from sklearn.preprocessing import MinMaxScaler


class SVM(BaseModel):
    def __init__(self, config, dataset):
        super(SVM, self).__init__(config, dataset)

    def set_params(self):
        self.w = self.w_init()
        self.alpha = self.config.svm_alpha
        self.delta = self.config.svm_delta

    def w_init(self):
        if self.config.svm_w_init == 'uniform':
            weight = np.random.uniform(-2, 2, size=[41, 3])

        return weight

    def preprocess(self):
        use_cols = ['FTR', 'home_wins', 'home_draws', 'home_losses', 'home_goals', 'home_oppos_goals',
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
        trainY = np.array(train.iloc[:, 0])
        trainX = np.array(train.iloc[:, 1:])
        testY = np.array(test.iloc[:, 0])
        testX = np.array(test.iloc[:, 1:])

        # apply min max scaling
        scaler = MinMaxScaler()
        trainX = scaler.fit_transform(trainX)
        testX = scaler.transform(testX)

        # add ones for bias term
        trainX = np.concatenate((trainX, np.ones([trainX.shape[0], 1])), axis=1)
        testX = np.concatenate((testX, np.ones([testX.shape[0], 1])), axis=1)

        trainY = np.where(trainY == 'H', 0, trainY)
        trainY = np.where(trainY == 'D', 1, trainY)
        trainY = np.where(trainY == 'A', 2, trainY)
        trainY = trainY.astype('int64')


        testY = np.where(testY == 'H', 0, testY)
        testY = np.where(testY == 'D', 1, testY)
        testY = np.where(testY == 'A', 2, testY)
        testY = testY.astype('int64')


        # save it in the dataset object
        self.dataset.trainX = trainX
        self.dataset.trainY = trainY
        self.dataset.testX = testX
        self.dataset.testY = testY


    def train(self):
        iter = 0
        losses = []
        past_hloss = 9999999
        while True:
            iter += 1
            #if iter>100:
                #break
            gradient_w, loss = self.gradient(self.dataset.trainX, self.w, self.dataset.trainY)

            self.w = self.w - self.alpha * gradient_w
            print('iter {} : loss - {} '.format(iter, loss))

            losses.append(loss)

            if np.mean(past_hloss - loss) < self.config.svm_tolerance:
                break
            past_hloss = loss

        # visualize descending loss value
        plt.plot(losses)
        plt.title('svm gradient descent loss')
        plt.ylabel('HingeLoss')
        plt.xlabel('epoch')
        plt.savefig('./out/svm_loss.png')
        plt.close()

    def gradient(self, x, w, y):
        gradient_w = np.zeros_like(w)

        num_classes = w.shape[1]
        num_train = x.shape[0]
        loss = 0.0

        for i in range(num_train):
            # For each training example, we will count the
            # scores and keep track of the correct class score
            scores = x[i].dot(w)
            correct_class_score = scores[y[i]]

            for j in range(num_classes):
                # We the  n compare it for each class
                if j == y[i]:
                    continue

                # Compute the margin
                margin = scores[j] - correct_class_score + self.delta

                # If margin is greater than zero, the class
                # contributes to the loss. And the gradient
                # is computed
                if margin > 0:
                    loss += margin
                    gradient_w[:, y[i]] -= x[i, :]
                    gradient_w[:, j] += x[i, :]
        loss = loss / num_train
        return gradient_w, loss

    def predict(self):
        train_out = np.matmul(self.dataset.trainX, self.w)
        test_out = np.matmul(self.dataset.testX, self.w)

        train_out = np.argmax(train_out, axis=1)
        test_out = np.argmax(test_out, axis=1)


        return train_out, test_out
