import numpy as np

def one_error(gnd, pred):
    gnd = gnd[:, 1:]

    pred_id = pred[:, 1]
    pred = pred[:, 1:]

    num_tweets = np.size(gnd, 0)

    sum1 = 0
    for i in range(num_tweets):
        cur_pred_id = i
        ind = np.argmax(pred[cur_pred_id])
        if gnd[i,ind] == '0':
            sum1 += 1

    one_error = sum1 / num_tweets

    return one_error