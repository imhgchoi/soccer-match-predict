import pandas as pd
import datetime

class Dataset():
    def __init__(self, config):
        self.config = config

        self.load()


    def load(self):
        data = pd.read_csv(self.config.datadir+'epl.csv')

        # modify date format
        data['Date'] = data['Date'].apply(lambda x : datetime.datetime.strptime(x, '%d/%m/%y').strftime('%Y-%m-%d'))

        # split data
        train, test = self.split(data)
        self.train_set = train
        self.test_set = test


    def preprocess(self):
        pass


    def split(self, data):
        train = data[pd.to_datetime(data['Date']).dt.year.apply(lambda x : x not in self.config.test_years)]
        test = data[pd.to_datetime(data['Date']).dt.year.apply(lambda x : x in self.config.test_years)]
        return train, test


    def get_data_info(self):
        print('train set size : {}'.format(self.train_set.shape))
        print('test set size : {}'.format(self.test_set.shape))
        print('columns : \n {}'.format(list(self.train_set.columns)))
        print('data sample : \n {}'.format(self.train_set.head(10)))