import numpy as np
import os

distmat_paths = [
        './logs/stage2/resnext101a_384/v1/dist_mat.npy',
        './logs/stage2/resnext101a_384/v2/dist_mat.npy',

        './logs/stage2/101a_384/v1/dist_mat.npy',
        './logs/stage2/101a_384/v2/dist_mat.npy',

        './logs/stage2/101a_384_recrop/v1/dist_mat.npy',
        './logs/stage2/101a_384_recrop/v2/dist_mat.npy',

        './logs/stage2/101a_384_spgan/v1/dist_mat.npy',
        './logs/stage2/101a_384_spgan/v2/dist_mat.npy',

        './logs/stage2/densenet169a_384/v1/dist_mat.npy',
        './logs/stage2/densenet169a_384/v2/dist_mat.npy',

        './logs/stage2/s101_384/v1/dist_mat.npy',
        './logs/stage2/s101_384/v2/dist_mat.npy',

        './logs/stage2/se_resnet101a_384/v1/dist_mat.npy',
        './logs/stage2/se_resnet101a_384/v2/dist_mat.npy',

        './logs/stage2/transreid_256/v1/dist_mat.npy',
        './logs/stage2/transreid_256/v2/dist_mat.npy',
        ]

distmat = np.zeros((1103,31238))
for i in distmat_paths:
    distmat += np.load(i)

sort_distmat_index = np.argsort(distmat, axis=1)
print(sort_distmat_index)
print('The shape of distmat is: {}'.format(distmat.shape))
save_path = './track2.txt'
with open(save_path,'w') as f:
    for item in sort_distmat_index:
        for i in range(99):
            f.write(str(item[i] + 1) + ' ')
        f.write(str(item[99] + 1) + '\n')

