import pandas as pd
import numpy as np
import sys

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# from math import exp

def meanAbsolutePercentageError(yTrue, yPredict):
    yTrue, yPredict = np.array(yTrue), np.array(yPredict)
    return np.mean(np.abs((yTrue - yPredict) / yTrue)) * 100


# Keys for team offense file
# PF: points scored by team
# Y/P: yards per play
# FL: fumbles lost
# NY/A: net yards gained per pass attempt
# RY/A: net yards gained per rush attempt
# Sc%: percent of drives scored

# read in data
dfTeamsO = pd.read_csv('data/nfl_week16_team_offense_2018.csv', header=0, float_precision='round_trip')
dfTeamsD = pd.read_csv('data/nfl_week16_team_defense_2018.csv', header=0, float_precision='round_trip')
dfGames = pd.read_csv('data/nfl_week16_game_stats_2018.csv', header=0)

dfWeek16 = pd.read_csv('predict/nfl_week16_predict.csv', header=0)

# these are the features of interest from the teams file
Okeys = ['Y/P','TO','PYds','PTD','Int','RYds','RTD','OPenYds','Sc%','OTO%']
Dkeys = ['OY/P','FTO','OPYds','OPTD','FInt','ORYds','ORTD','DPenYds','OSc%','DTO%']
Wkeys = ['WY/P','WTO','WPYds','WPTD','WInt','WRYds','WRTD','WPenYds','WSc%','WTO%','WOY/P','WFTO','WOPYds','WOPTD','WFInt','WORYds','WORTD','WDPenYds','WOSc%','WDTO%']
Lkeys = ['LY/P','LTO','LPYds','LPTD','LInt','LRYds','LRTD','LPenYds','LSc%','LTO%','LOY/P','LFTO','LOPYds','LOPTD','LFInt','LORYds','LORTD','LDPenYds','LOSc%','LDTO%']
reversedKeys = Lkeys + Wkeys

# append each game from the season with the teams' stats
dfGames = pd.concat([dfGames, pd.DataFrame(0, index=np.arange(len(dfGames)), columns=Wkeys)], axis=1)
dfGames = pd.concat([dfGames, pd.DataFrame(0, index=np.arange(len(dfGames)), columns=Lkeys)], axis=1)
dfWeek16 = pd.concat([dfWeek16, pd.DataFrame(0, index=np.arange(len(dfWeek16)), columns=Wkeys)], axis=1)
dfWeek16 = pd.concat([dfWeek16, pd.DataFrame(0, index=np.arange(len(dfWeek16)), columns=Lkeys)], axis=1)

# get lengths
numTeams = len(dfTeamsO.index)
numGames = len(dfGames.index)
numPGames = len(dfWeek16.index)

# get keys
dfTempO = dfTeamsO[Okeys]
dfTempD = dfTeamsD[Dkeys]

# fill in prediction dataframe
for i in range(0,numPGames):
    pWinner = dfWeek16.at[i,'Winner']
    pLoser = dfWeek16.at[i,'Loser']

    pwinnerIdx = dfTeamsO.index[dfTeamsO['Tm'] == pWinner].tolist()
    ploserIdx = dfTeamsO.index[dfTeamsO['Tm'] == pLoser].tolist()

    pwinnerOStats = dfTempO.iloc[pwinnerIdx[0],:]
    ploserOStats = dfTempO.iloc[ploserIdx[0],:]
    pwinnerDStats = dfTempD.iloc[pwinnerIdx[0],:]
    ploserDStats = dfTempD.iloc[ploserIdx[0],:]

    pwinnerStats = pwinnerOStats.append(pwinnerDStats)
    ploserStats = ploserOStats.append(ploserDStats)

    for j in range(0,len(pwinnerStats)):
        dfWeek16.at[i,Wkeys[j]] = pwinnerStats[j]
        dfWeek16.at[i,Lkeys[j]] = ploserStats[j]

# indices to be filled starts at 8
# fill in values for the team stats
for i in range(0,numGames):
    # for every game
    # get the winner and loser
    winner = dfGames.at[i,'Winner']
    loser = dfGames.at[i,'Loser']

    # get index of winner and loser
    winnerIdx = dfTeamsO.index[dfTeamsO['Tm'] == winner].tolist()
    loserIdx = dfTeamsO.index[dfTeamsO['Tm'] == loser].tolist()

    # get the winner and loser stats from team stats
    winnerOStats = dfTempO.iloc[winnerIdx[0],:]
    loserOStats = dfTempO.iloc[loserIdx[0],:]
    winnerDStats = dfTempD.iloc[winnerIdx[0],:]
    loserDStats = dfTempD.iloc[loserIdx[0],:]

    winnerStats = winnerOStats.append(winnerDStats)
    loserStats = loserOStats.append(loserDStats)

    # for both teams
    # fill in values
    for j in range(0,len(winnerStats)):
        dfGames.at[i,Wkeys[j]] = winnerStats[j]
        dfGames.at[i,Lkeys[j]] = loserStats[j]

################################################################################
#              Code below is used for predicting the winning team              #
################################################################################

firstTeams = dfWeek16.iloc[:,0]
secondTeams = dfWeek16.iloc[:,1]
# get list of winning scores
yW = dfGames.iloc[:,2]
# get list of losing scores
yL = dfGames.iloc[:,3]
# get list of features to test
Xw = dfGames.iloc[:,8:]
Xl = Xw.ix[:, reversedKeys]

predictXW = dfWeek16.iloc[:,2:]
predictXL = predictXW.ix[:, reversedKeys]

xwTrain, xwTest, ywTrain, ywTest = train_test_split(Xw, yW, test_size=0.25, random_state=0)
xlTrain, xlTest, ylTrain, ylTest = train_test_split(Xl, yL, test_size=0.25, random_state=0)
# logreg = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial')
logregW = LogisticRegression()
logregL = LogisticRegression()

logregW.fit(xwTrain, ywTrain)
logregL.fit(xlTrain, ylTrain)

predictionsW = logregW.predict(predictXW)
predictionsL = logregL.predict(predictXL)

# print("\nThe training set is:\n")
# print(dfWeek16)

# predictions = logreg.predict(predictX)
# print("\nThe real winners were:\n")
# print(yTest)
# print("\nThe predicted first team scores are:\n")
# print("\n{}".format(predictionsW))
# # score = logregW.score(xwTest, ywTest)
# # print("\nAccuracy for predicting the winning score was: {:.2f}%".format(score * 100))
#
# print("\nThe predicted second team scores are:\n")
# print("\n{}".format(predictionsL))
# score = logregL.score(xlTest, ylTest)
# print("\nAccuracy for predicting the secon score was: {:.2f}%".format(score * 100))

for i in range(len(predictionsW)):
    if predictionsW[i] > predictionsL[i]:
        print("Winner of the {} game is {} by a score of {} to {}, line of {}".format(i + 1, firstTeams[i], predictionsW[i], predictionsL[i], predictionsW[i] - predictionsL[i]))
    else:
        print("Winner of the {} game is {} by a score of {} to {}, line of {}".format(i + 1, secondTeams[i], predictionsL[i], predictionsW[i], predictionsL[i] - predictionsW[i]))

# ################################################################################
# #              Code below is used for predicting the winning score             #
# ################################################################################
#
# # get list of winning scores
# y = dfGames.iloc[:,2]
#
# xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.25, random_state=0)
# # logreg = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial')
# logreg = LogisticRegression()
# logreg.fit(xTrain, yTrain)
#
# # print("\nThe training set is:\n")
# # print(xTest)
#
# predictions = logreg.predict(xTest)
# # print("\nThe real winninng scores were:\n")
# # print(yTest)
# # print("\nThe predicted winning scores are:\n")
# # print("\n{}".format(predictions))
# score = logreg.score(xTest, yTest)
# mape = meanAbsolutePercentageError(yTest, predictions)
# print("\nAccuracy for predicting the winning score was: {:.2f}%".format(score * 100))
# print("\nPercent error for predicting the winning score was: {:.2f}%".format(mape))
#
# ################################################################################
# #              Code below is used for predicting the losing score              #
# ################################################################################
#
# # get list of winning scores
# y = dfGames.iloc[:,3]
#
# xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.25, random_state=0)
# # logreg = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial')
# logreg = LogisticRegression()
# logreg.fit(xTrain, yTrain)
#
# # print("\nThe training set is:\n")
# # print(xTest)
#
# predictions = logreg.predict(xTest)
# # print("\nThe real winninng scores were:\n")
# # print(yTest)
# # print("\nThe predicted winning scores are:\n")
# # print("\n{}".format(predictions))
# score = logreg.score(xTest, yTest)
# mape = meanAbsolutePercentageError(yTest, predictions)
# print("\nAccuracy for predicting the losing score was: {:.2f}%".format(score * 100))
# print("\nPercent error for predicting the losing score was: {:.2f}%".format(mape))
