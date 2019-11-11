import numpy as np
import tensorflow as tf

def compute_SPK(testlabel, predictlabel, K):
    sorted_predictlabel = np.sort(predictlabel)
    sorted_predictlabel = sorted_predictlabel[::-1]
    index = np.argsort(-predictlabel)  
    row1, col1 = np.shape(testlabel)
    row2, col2 = np.shape(sorted_predictlabel)

    SK = np.zeros(row1)
    PK = np.zeros(row1)

    if (row1 == row2) and (col1 == col2):
        for i in range(row1):
            for j in range(K):
                b = index[i][j]
                if testlabel[i][b] == '1':  
                    SK[i] = 1 
                    PK[i] += 1 
    else:
        print('something wrong with the dimension of testlabel and predictlabel')
    return SK, PK
    # return 1


