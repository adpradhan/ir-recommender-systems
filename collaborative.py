import numpy as np
import pandas as pd
from metrics import analyse
import time

def get_sim_matrix(mat):
  no_of_users = len(mat)
  simMatrix = np.corrcoef(mat)[:no_of_users+1, :no_of_users+1]
  return simMatrix

def collaborative_without_baseline(train,test,simMatrix,k):
  (no_of_users,no_of_items) = train.shape
  predictedTrain = np.zeros((no_of_users,no_of_items))
  for user in range(0,no_of_users):
    KUsers = np.argsort(-simMatrix[user])[:k]
    for item in range(0,no_of_items):
      # only predicting for places where test data exists
      if test[user][item] !=0 :
        # calculating weighted ratings
        simSum = 0 # denominator
        tempNum = 0 #numerator
        for CUser in KUsers:
          if train[CUser][item] != 0:
            tempNum += simMatrix[user][CUser] * train[CUser][item]
            simSum += simMatrix[user][CUser]   
        if simSum != 0:
          predictedTrain[user][item] = tempNum/simSum
  return predictedTrain

def collaborative_with_baseline(train,test,simMatrix,k):
  (no_of_users,no_of_items) = train.shape
  mean = 0
  rd_users = np.zeros(no_of_users)
  rd_items = np.zeros(no_of_items)
  cntr_users = np.zeros(no_of_users)
  cntr_items = np.zeros(no_of_items)
  for user in range(0,no_of_users):
    for item in range(0,no_of_items):
      if train[user][item] != 0:
        rd_users[user] += train[user][item]
        rd_items[item] += train[user][item]
        mean += train[user][item]
        cntr_users[user] += 1
        cntr_items[item] += 1

  for u in range(0,no_of_users):
    if cntr_users[u] == 0:
      cntr_users[u] = 1
  for i in range(0,no_of_items):
    if cntr_items[i] == 0:
      cntr_items[i] = 1
  total = np.sum(cntr_users)
  mean = (mean/total)
  rd_users = rd_users/cntr_users
  rd_users = rd_users - mean
  rd_items = rd_items/cntr_items
  rd_items = rd_items - mean
  
  baseline = np.zeros((no_of_users,no_of_items))
  for user in range(0,no_of_users):
      for item in range(0,no_of_items):
          baseline[user][item] = mean + rd_users[user] + rd_items[item]
  
  predictedTrain = np.zeros((no_of_users,no_of_items))
  for user in range(0,no_of_users):
    KUsers = np.argsort(-simMatrix[user])[:k]
    for item in range(0,no_of_items):
      # only predicting for places where test data exists
      if test[user][item] !=0 :
        # calculating weighted ratings
        simSum = 0 # denominator
        tempNum = 0 #numerator
        for CUser in KUsers:
          if train[CUser][item] != 0:
            tempNum += simMatrix[user][CUser] * (train[CUser][item]-baseline[CUser][item])
            simSum += simMatrix[user][CUser]   
        if simSum != 0:
          predictedTrain[user][item] = tempNum/simSum
          predictedTrain[user][item] += baseline[user][item]
  return predictedTrain

# k is number of nearest users to consider
def startCollaborative(train,test,k):
  print("Collaborative...")
  simMatrix = get_sim_matrix(train)
  print("Basic...")
  t1 = time.time()
  predictedTrain = collaborative_without_baseline(train,test,simMatrix,k)
  #print(predictedTrain.shape)
  (RMSE, SRC, PTOPK) = analyse(predictedTrain, test,10,3.5)
  t2 = time.time()
  print(RMSE, SRC, PTOPK, t2-t1)
  print("Baseline...")
  t3 = time.time()
  predictedTrain_bl = collaborative_with_baseline(train,test,simMatrix,k)
  # metrics rmse, src, precision on top k
  (RMSE, SRC, PTOPK) = analyse(predictedTrain_bl, test,10,3.5)
  t4 = time.time()
  print(RMSE, SRC,PTOPK, t4-t3)
  