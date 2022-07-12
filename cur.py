import numpy as np
import time
from svd import svd_decomp
from svd import svd_decomp_90
from math import sqrt

def CUR(train,test,k):
  user_movie_mat = train
  mat_ssq = 0
  num_users = user_movie_mat.shape[0]
  num_movies = user_movie_mat[0].size
  for i in range(num_users):
    for j in range(num_movies):
      mat_ssq += user_movie_mat[i][j]**2
  
  user_prob = []
  for i in range(num_users):
    row_ssq = 0
    for j in range(num_movies):
      row_ssq += user_movie_mat[i][j]**2
    user_prob.append(row_ssq/mat_ssq)
  
  movie_prob = []
  for i in range(num_movies):
    col_ssq = 0
    for j in range(num_users):
      col_ssq += user_movie_mat[j][i]**2
    movie_prob.append(col_ssq/mat_ssq)

  sel_users = np.random.choice(num_users,k, replace=False, p=user_prob)
  sel_movies = np.random.choice(num_movies,k, replace=False, p=movie_prob)

  sel_movies.sort()
  sel_users.sort()
  C = []
  R = []
  for i in sel_users:
    R.append(list(user_movie_mat[i]/sqrt(k*user_prob[i])))
  for j in sel_movies:
    C.append(list(user_movie_mat[:,j]/sqrt(k*movie_prob[j])))
  Ct = np.transpose(C)
  W = []
  for i in sel_users:
    X=[]
    for j in sel_movies:
      X.append(user_movie_mat[i][j])
    W.append(np.array(X))
  W = np.array(W)
  #print(W.shape)

  x,yt,sigma = svd_decomp(W)#SVD of intersection
  sig_sq = np.zeros((sigma.shape[0],sigma.shape[0]))
  for i in range(sigma.shape[0]):
    sig_sq[i][i]=1/sigma[i][i]
  # sig_sq = np.linalg.pinv(sigma) #Moore Penrose Pseudo Inverse
  sig_sq = np.linalg.matrix_power(sig_sq, 2)#square of pseudo-inverse
  
  y = np.transpose(yt)
  xt = np.transpose(x)

  U = np.matmul(y, sig_sq)
  U = np.matmul(U, xt)
  return (Ct, R, U)
  
def CUR_90(train,test,k):
  user_movie_mat = train
  mat_ssq = 0
  num_users = user_movie_mat.shape[0]
  num_movies = user_movie_mat[0].size
  for i in range(num_users):
    for j in range(num_movies):
      mat_ssq += user_movie_mat[i][j]**2
  
  user_prob = []
  for i in range(num_users):
    row_ssq = 0
    for j in range(num_movies):
      row_ssq += user_movie_mat[i][j]**2
    user_prob.append(row_ssq/mat_ssq)
  
  movie_prob = []
  for i in range(num_movies):
    col_ssq = 0
    for j in range(num_users):
      col_ssq += user_movie_mat[j][i]**2
    movie_prob.append(col_ssq/mat_ssq)

  sel_users = np.random.choice(num_users,k, replace=False, p=user_prob)
  sel_movies = np.random.choice(num_movies,k, replace=False, p=movie_prob)

  sel_movies.sort()
  sel_users.sort()
  C = []
  R = []
  for i in sel_users:
    R.append(list(user_movie_mat[i]/sqrt(k*user_prob[i])))
  for j in sel_movies:
    C.append(list(user_movie_mat[:,j]/sqrt(k*movie_prob[j])))
  Ct = np.transpose(C)
  W = []
  for i in sel_users:
    X=[]
    for j in sel_movies:
      X.append(user_movie_mat[i][j])
    W.append(np.array(X))
  W = np.array(W)
  #print(W.shape)

  x,yt,sigma = svd_decomp(W)#SVD of intersection
  sig_sq = np.zeros((sigma.shape[0],sigma.shape[0]))
  for i in range(sigma.shape[0]):
    sig_sq[i][i]=1/sigma[i][i]
  # sig_sq = np.linalg.pinv(sigma) #Moore Penrose Pseudo Inverse
  sig_sq = np.linalg.matrix_power(sig_sq, 2)#square of pseudo-inverse
  
  y = np.transpose(yt)
  xt = np.transpose(x)

  U = np.matmul(y, sig_sq)
  U = np.matmul(U, xt)
  new_x, new_yt, new_sigma = svd_decomp_90(x,yt,sigma)
  new_sig_sq = np.linalg.pinv(new_sigma)
  # new_sig_sq = np.linalg.matrix_power(pinv_new_sigma, 2)
  new_y = np.transpose(new_yt)
  new_xt = np.transpose(new_x)
  U_90 = np.matmul(new_y, new_sig_sq)
  U_90 = np.matmul(U_90, new_xt)
  return (Ct, R, U_90)

def metric( ori, pred ):
  sum=0
  count=0
  for i in range(0,len(ori)):
    for j in range(0,len(ori[i])):
      sum+=(ori[i][j]-pred[i][j])**2
      count=count+1
  rmse_error=(sum/count)**0.5

  train = ori
  count=0
  sum=0
  for i in range(0,len(ori)):
    for j in range(0,len(ori[i])):
      sum=sum+(ori[i][j]-pred[i][j])**2
      count=count+1
  sum=6*sum
  temp=(count**3)-count
  src_error=1-(sum/temp)

  precision = prek(ori,train)

  return (rmse_error,src_error,precision)
  
def prek(ori,pred):
  k_mat=pred.tolist()
  count=0.00
  match=0.00
  for i in range(len(ori)):
    for j in range(len(ori[i])):
      count+=1
      a=int(round(ori[i][j]))
      b=int(round(k_mat[i][j]))
      if (a==b):
        match=match+1
  
  precision=(match*100)/count
  return precision/100
  
	
def CUR_driver(train,test):
  s1 = time.time()
  (Ct, R, U) = CUR(train,test,600)
  final = np.matmul(Ct, U)
  final = np.matmul(final, R)
  print('CUR')
  print(metric(train,final))
  s2 = time.time()
  print(s2-s1)

  s1 = time.time()
  (Ct, R, U_90) = CUR(train,test,600)
  final_90 = np.matmul(Ct, U_90)
  final_90 = np.matmul(final_90, R)
  print('CUR with 90%')
  print(metric(train,final_90))
  s2 = time.time()
  print(s2-s1)
  