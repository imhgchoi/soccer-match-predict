# Feature Specifications
#### The following describes what kind of features are included in dataset.train_set and dataset.test_set after preprocessing.

-----------

### English version
> You may refer to the link provided in README.md for the following features :\
  **Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR, HTHG, HTAG, HTR, Referee, HS, AS, HST, AST, HC, AC, HF, AF, HY, AY, HR, AR**

> **Hodds, Dodds, Aodds** : Average odds of betting sites for the specific match. Corresponds to odds for Home, Draw, Away

> **home_goals, away_goals** : Average goals scored in the most recent N matches of the Home(Away) team

> **home_oppos_goals, away_oppos_goals** : Average goals scored in the recent N matches between the Home and Away

> **home_oppos_wins, home_oppos_draws, home_oppos_losses** :\
 The ratio of wins, draws, losses of the Home team in the recent N matches against the Away team. 1 minus the value 
 naturally signifies the ratio of wins, draws, losses of the Away team in the recent N matches against the Home team.
 
> **home_wins, away_wins** : The win ratio in the most recent N matches of the Home(Away) team

> **home_losses, away_losses** : The loss ratio in the most recent N matches of the Home(Away) team

> **home_draws, away_draws** : The draw ratio in the most recent N matches of the Home(Away) team

> **home_shots, away_shots** : Average # of shootings in the most recent N matches of the Home(Away) team

> **home_oppos_shots, away_oppos_shots** : Average # of shootings in the recent N matches played against the same opponent

> **home_shotontarget, away_shotontarget** : Average # of shootings on target in the most recent N matches of the Home(Away) team

> **home_oppos_shotontarget, away_oppos_shotontarget** : Average # of shots on target in the recent N matches played against the same opponent

> **home_cornerkicks, away_cornerkicks** : Average # of cornerkicks in the most recent N matches of the Home(Away) team

> **home_oppos_cornerkicks, away_oppos_cornerkicks** : Average # of cornerkicks in the recent N matches played against the same opponent

> **home_fouls, away_fouls** : Average # of fouls committed in the most recent N matches of the Home(Away) team

> **home_oppos_fouls, away_oppos_fouls** : Average # of fouls committed in the recent N matches played against the same opponent

> **home_yellowcards, away_yellowcards** : Average # of yellowcards received in the most recent N matches of the Home(Away) team

> **home_oppos_yellowcards, away_oppos_yellowcards** : Average # of yellocards received in the recent N matches played against the same opponent

> **home_redcards, away_redcards** : Average  # of redcards received in the most recent N matches of the Home(Away) team

> **home_oppos_redcards, away_oppos_redcards** : Average # of redcards received in the recent N matches played against the same opponent


--------------------
### Korean version
> 다음 변수들은 README.md 에 링크되어있는 meta data를 참고하면 됩니다 :\
  **Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR, HTHG, HTAG, HTR, Referee, HS, AS, HST, AST, HC, AC, HF, AF, HY, AY, HR, AR**

> **Hodds, Dodds, Aodds** :\
 해당 경기에 대한 여러 베팅 회사들의 배당률 평균. 순서대로 Home 승리, 무승부, Away 승리인 경우에 대한 배당률 평균

> **home_goals, away_goals** : Home(Away)팀의 최근 N 경기에서의 득점 평균

> **home_oppos_goals, away_oppos_goals** : Home과 Away가 맞붙은 최근 N 경기에서 각 팀의 득점 평균

> **home_oppos_wins, home_oppos_draws, home_oppos_losses** :\
 Home과 Away가 맞붙은 최근 N 경기에서 Home 팀의 승리, 무승부, 패배 비율. 1에서 해당 값을 빼면 Away의 승리, 무승부, 패배 비율이 됨

> **home_wins, away_wins** : Home(Away)팀의 최근 N 경기에서의 승리 비율

> **home_losses, away_losses** : Home(Away)팀의 최근 N 경기에서의 패배 비율

> **home_draws, away_draws** : Home(Away)팀의 최근 N 경기에서의 무승부 비율

> **home_shots, away_shots** : Home(Away)팀의 최근 N 경기에서의 슛 개수 평균

> **home_oppos_shots, away_oppos_shots** : Home과 Away가 맞붙은 최근 N 경기에서 각 팀의 슛 개수 평균

> **home_shotontarget, away_shotontarget** : Home(Away)팀의 최근 N 경기에서의 유효슛 수 평균

> **home_oppos_shotontarget, away_oppos_shotontarget** : Home과 Away가 맞붙은 최근 N 경기에서 각 팀의 유효슛 평균

> **home_cornerkicks, away_cornerkicks** : Home(Away)팀의 최근 N 경기에서의 코너킥 개수 평균

> **home_oppos_cornerkicks, away_oppos_cornerkicks** : Home과 Away가 맞붙은 최근 N 경기에서 각 팀의 코너킥 수 평균

> **home_fouls, away_fouls** : Home(Away)팀의 최근 N 경기에서의 파울 수 평균

> **home_oppos_fouls, away_oppos_fouls** : Home과 Away가 맞붙은 최근 N 경기에서 각 팀의 파울 수 평균

> **home_yellowcards, away_yellowcards** : Home(Away)팀의 최근 N 경기에서 받은 옐로카드 수 평균 

> **home_oppos_yellowcards, away_oppos_yellowcards** : Home과 Away가 맞붙은 최근 N 경기에서 각 팀의 옐로카드 수 평균 

> **home_redcards, away_redcards** : Home(Away)팀의 최근 N 경기에서 받은 레드카드 수 평균

> **home_oppos_redcards, away_oppos_redcards** : Home과 Away가 맞붙은 최근 N 경기에서 각 팀의 레드카드 수 평균
