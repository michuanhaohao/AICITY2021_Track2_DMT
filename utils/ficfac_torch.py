import torch
import numpy as np
from copy import deepcopy


def compute_P2(qf, gf, gc, la=3.0):
    X = gf
    neg_vec = {}
    u_cams = np.unique(gc)
    P = {}
    for cam in u_cams:
        curX = gf[gc == cam]
        neg_vec[cam] = torch.mean(curX, axis=0)
        tmp_eye = torch.eye(X.shape[1]).cuda()
        P[cam] = torch.inverse(curX.T.matmul(curX)+curX.shape[0]*la*tmp_eye)
    return P, neg_vec


def meanfeat_sub(P, neg_vec, in_feats, in_cams):
    out_feats = []
    for i in range(in_feats.shape[0]):
        camid = in_cams[i]
        feat = in_feats[i] - neg_vec[camid]
        feat = P[camid].matmul(feat)
        feat = feat/torch.norm(feat, p=2)
        out_feats.append(feat)
    out_feats = torch.stack(out_feats)
    return out_feats


def mergesetfeat(X, cams, gX, gcams, beta=0.08, knn=30):
    for i in range(X.shape[0]):
        #  if i % 5000 == 0:
            #  print('merge:%d/%d' % (i, X.shape[0]))
        knnX = gX #[gcams != cams[i], :]
        sim = knnX.matmul(X[i, :])
        knnX = knnX[sim>0]
        sim = sim[sim>0]
        if len(sim) > 0:
            idx = torch.argsort(-sim)
            if len(sim)>2*knn:
                sim = sim[idx[:2*knn]]
                knnX = knnX[idx[:2*knn],:]
            else:
                sim = sim[idx]
                knnX = knnX[idx,:]
                knn = min(knn,len(sim))
            knn_pos_weight = torch.exp((sim[:knn]-1)/beta)
            knn_neg_weight = torch.ones(len(sim)-knn).cuda()
            knn_pos_prob = knn_pos_weight/torch.sum(knn_pos_weight)
            knn_neg_prob = knn_neg_weight/torch.sum(knn_neg_weight)
            X[i,:] += 0.2*(knn_pos_prob.matmul(knnX[:knn,:]) - knn_neg_prob.matmul(knnX[knn:,:]))
            X[i,:] /= torch.norm(X[i,:])
    return X


def run_fic(qf, gf, qcams, gcams, la=0.02):
    P, neg_vec = compute_P2(qf, gf, gcams, la)
    qf = meanfeat_sub(P, neg_vec, qf, qcams)
    gf = meanfeat_sub(P, neg_vec, gf, gcams)
    return qf, gf
