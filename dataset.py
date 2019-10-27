import pandas as pd
import numpy as np
import datetime
from utils import conjunction, union

class Dataset():
    def __init__(self, config):
        self.config = config

        self.load()


    def load(self):
        data = pd.read_csv(self.config.datadir+'epl.csv')
        self.raw = data.copy()

        # preprocess data
        data = self.preprocess(data)

        # split data
        train, test = self.split(data)
        self.train_set = train
        self.test_set = test


    def preprocess(self, data):
        print('preprocessing data...')

        # modify date format
        data['Date'] = data['Date'].apply(lambda x : datetime.datetime.strptime(x, '%d/%m/%y').strftime('%Y-%m-%d'))

        # average out betting odds
        data['Hodds'] = np.mean(data[['B365H','BWH','GBH','IWH','LBH','SBH','WHH','SJH','VCH','BSH']],axis=1)
        data['Dodds'] = np.mean(data[['B365D','BWD','GBD','IWD','LBD','SBD','WHD','SJD','VCD','BSD']],axis=1)
        data['Aodds'] = np.mean(data[['B365A','BWA','GBA','IWA','LBA','SBA','WHA','SJA','VCA','BSA']],axis=1)

        # filter columns - meta data @ http://www.football-data.co.uk/notes.txt
        use_col = ['Date','HomeTeam','AwayTeam','FTHG','FTAG','FTR','HTHG','HTAG','HTR','Referee','HS','AS','HST','AST',
                   'HF','AF','HC','AC','HY','AY','HR','AR','Hodds','Dodds','Aodds']
        data = data[use_col]

        # accumulate histories
        # : referenced http://andrew.carterlunn.co.uk/programming/2018/02/20/beating-the-bookmakers-with-tensorflow.html
        acc_hist = {'home_wins' : [], 'home_draws' : [], 'home_losses' : [], 'home_goals' : [], 'home_oppos_goals' : [],
                    'home_shots' : [], 'home_oppos_shots' : [], 'home_shotontarget' : [], 'home_oppos_shotontarget' : [],
                    'away_wins' : [], 'away_draws' : [], 'away_losses' : [], 'away_goals' : [], 'away_oppos_goals' : [],
                    'away_shots' : [], 'away_oppos_shots' : [], 'away_shotontarget' : [], 'away_oppos_shotontarget' : []}

        for row in data.iterrows() :
            hometeam = row[1]['HomeTeam']
            awayteam = row[1]['AwayTeam']
            date = row[1]['Date']

            # filter matches with same playing teams
            temp1 = data[conjunction(data['HomeTeam']==hometeam, data['AwayTeam']==awayteam)]
            temp2 = data[conjunction(data['HomeTeam']==awayteam, data['AwayTeam']==hometeam)]
            temp = pd.concat([temp1, temp2], axis=0)
            history = temp[temp['Date']<date].sort_values(by='Date').tail(self.config.lookback_opp_matches)

            # if opponent history is too short, continue
            if len(history) < self.config.lookback_opp_matches :
                for key in list(acc_hist.keys()) :
                    acc_hist[key].append(np.nan)
                continue

            # compute average number of goals scored against opponent in the past N matches with the opponent
            home = history[history['HomeTeam'] == hometeam]
            away = history[history['AwayTeam'] == hometeam]
            home_sum = np.sum(home[['FTHG','FTAG','HS','AS','HST','AST']])
            away_sum = np.sum(away[['FTHG','FTAG','HS','AS','HST','AST']])


            # filter recent N matches of both home and away
            home = data[union(data['HomeTeam']==hometeam, data['AwayTeam']==hometeam)]
            home = home[home['Date']<date].sort_values(by='Date').tail(self.config.lookback_matches)
            away = data[union(data['HomeTeam']==awayteam, data['AwayTeam']==awayteam)]
            away = away[away['Date']<date].sort_values(by='Date').tail(self.config.lookback_matches)

            # if match history is too short, continue
            if len(home) < self.config.lookback_matches or len(away) < self.config.lookback_matches :
                for key in list(acc_hist.keys()) :
                    acc_hist[key].append(np.nan)
                continue

            home_home_sum = np.sum(home[home['HomeTeam']==hometeam][['FTHG','HS','HST']])
            home_away_sum = np.sum(home[home['AwayTeam']==hometeam][['FTAG','AS','AST']])
            away_home_sum = np.sum(away[away['HomeTeam']==awayteam][['FTHG','HS','HST']])
            away_away_sum = np.sum(away[away['AwayTeam']==awayteam][['FTAG','AS','AST']])

            acc_hist['home_oppos_goals'].append((home_sum['FTHG'] + away_sum['FTAG']) / self.config.lookback_opp_matches)
            acc_hist['away_oppos_goals'].append((home_sum['FTAG'] + away_sum['FTHG']) / self.config.lookback_opp_matches)
            acc_hist['home_oppos_shots'].append((home_sum['HS'] + away_sum['AS']) / self.config.lookback_opp_matches)
            acc_hist['away_oppos_shots'].append((home_sum['AS'] + away_sum['HS']) / self.config.lookback_opp_matches)
            acc_hist['home_oppos_shotontarget'].append((home_sum['HST'] + away_sum['AST']) / self.config.lookback_opp_matches)
            acc_hist['away_oppos_shotontarget'].append((home_sum['AST'] + away_sum['HST']) / self.config.lookback_opp_matches)
            acc_hist['home_goals'].append((home_home_sum['FTHG'] + home_away_sum['FTAG']) / self.config.lookback_matches)
            acc_hist['away_goals'].append((away_home_sum['FTHG'] + away_away_sum['FTAG']) / self.config.lookback_matches)
            acc_hist['home_shots'].append((home_home_sum['HS'] + home_away_sum['AS']) / self.config.lookback_matches)
            acc_hist['away_shots'].append((away_home_sum['HS'] + away_away_sum['AS']) / self.config.lookback_matches)
            acc_hist['home_shotontarget'].append((home_home_sum['HST'] + home_away_sum['AST']) / self.config.lookback_matches)
            acc_hist['away_shotontarget'].append((away_home_sum['HST'] + away_away_sum['AST']) / self.config.lookback_matches)

            # count ratio of wins / draws / losses in the past N matches
            res = []
            for r in home.iterrows() :
                if r[1]['HomeTeam'] == hometeam :
                    res.append(r[1]['FTR'])
                else :
                    if r[1]['FTR'] == 'A' :
                        res.append('H')
                    elif r[1]['FTR'] == 'H' :
                        res.append('A')
                    else :
                        res.append('D')
            acc_hist['home_wins'].append(res.count('H') / self.config.lookback_matches)
            acc_hist['home_draws'].append(res.count('D') / self.config.lookback_matches)
            acc_hist['home_losses'].append(res.count('A') / self.config.lookback_matches)

            res = []
            for r in away.iterrows() :
                if r[1]['HomeTeam'] == awayteam :
                    res.append(r[1]['FTR'])
                else :
                    if r[1]['FTR'] == 'A' :
                        res.append('H')
                    elif r[1]['FTR'] == 'H' :
                        res.append('A')
                    else :
                        res.append('D')
            acc_hist['away_wins'].append(res.count('H') / self.config.lookback_matches)
            acc_hist['away_draws'].append(res.count('D') / self.config.lookback_matches)
            acc_hist['away_losses'].append(res.count('A') / self.config.lookback_matches)

        acc_hist = pd.DataFrame(acc_hist)
        data = pd.concat([data, acc_hist], axis=1)
        data = data.dropna()

        return data

    def split(self, data):
        train = data[pd.to_datetime(data['Date']).dt.year.apply(lambda x : x not in self.config.test_years)]
        test = data[pd.to_datetime(data['Date']).dt.year.apply(lambda x : x in self.config.test_years)]
        return train, test


    def get_data_info(self):
        print('train set size : {}'.format(self.train_set.shape))
        print('test set size : {}'.format(self.test_set.shape))
        print('columns : \n {}'.format(list(self.train_set.columns)))
        print('data sample : \n {}'.format(self.train_set.head(10)))