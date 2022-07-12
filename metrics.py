import numpy as np


def analyse(predicted,test,k,t):
  no_of_users = len(predicted)
  no_of_items = len(predicted[0])
  squared_error=0
  totalr = 0
  for u in range(0,no_of_users):
    for i in range(0,no_of_items):
      if test[u][i] != 0:
        squared_error += (test[u][i]-predicted[u][i])**2
        totalr += 1
  rmse = (squared_error/totalr)**0.5
  src = 1-((6*squared_error)/(totalr * (totalr ** 2 - 1)))

  no_movies_rated = {}
  i = 0
  for user in test:
    no_movies_rated[i] = user[user>0].size
    i += 1
  i = 0

  user_prec = []
  for user in predicted:
    if no_movies_rated[i] < k:
      i+=1
      continue
    topk_i = (-user).argsort()[:k]
    topk_v = []
    for j in topk_i:
      topk_v.append((j,user[j]))

    rec = []
    for (ind,user[ind]) in topk_v:
      if(user[ind]>=t):
        rec.append((ind,user[ind]))
    cnt = 0
    for j in rec:
      if test[i][j[0]] >= t:
        cnt = cnt + 1
    if len(rec) > 0:
      user_prec.append(cnt/len(rec))
    i += 1
  return (rmse, src, (sum(user_prec)/len(user_prec)))