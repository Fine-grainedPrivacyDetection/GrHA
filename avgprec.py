import numpy as np

def avgprec(gnd, pred):
    gnd_id = gnd[:, 0]
    gnd = gnd[:, 1:]

    pred_id = pred[:, 0]
    pred_id = np.array(pred_id, dtype=int)
    pred = pred[:, 1:]

    num_tweets = np.size(gnd, 0)
    num_properties = np.size(gnd, 1)

    prec = np.zeros(num_tweets)

    for i in range(num_tweets):
        
        cur_pred_id = i
        ind = np.argsort(-pred[cur_pred_id])
        gnd = np.array(gnd, int)
        Y = np.sum(gnd[i])
        exist_pro_index = np.where(gnd[i]==1)[0]
        rank = np.zeros(num_properties)
        for j in exist_pro_index:
            rank[j] = np.where(ind==j)[0][0] + 1
        ind_rank = np.where(rank!=0)[0]
        sum_rank = 0
        for j in ind_rank:
            sum_rank += np.size(np.where((rank <= rank[j]) & (rank > 0))[0]) / rank[j]
        if Y == 0:
            prec[i] = 0
        else:
            prec[i] = sum_rank / Y
        if not(prec[i] >= 0):
            print(i, sum_rank, Y, gnd[i])
    tot = 0
    for i, num in enumerate(prec):
        tot += num
       
    ans = tot / num_tweets
    print('tot: ', tot)
    print('num_tweets: ', num_tweets)
    return ans