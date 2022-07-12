# driver
import numpy as np
import pandas as pd
from preprocess import makeMatrix
from collaborative import startCollaborative
from svd import startSVD
from cur import CUR_driver
# driver temp
(train, test) = makeMatrix()
# train = train.to_numpy()
# test = train.to_numpy()
print(train.shape)
# startCollaborative(train, test, 50)
# 50 is number of collaborative neighbours
startCollaborative(train, test, 50)
startSVD(train, test)
CUR_driver(train, test)