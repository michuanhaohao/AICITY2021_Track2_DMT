import os
from torch.backends import cudnn
from utils.logger import setup_logger
from datasets import make_dataloader, get_trainloader_uda, get_testloader_uda
from model import make_model
from solver import make_optimizer, WarmupMultiStepLR
from loss import make_loss
from processor import do_uda_train
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import numpy as np
import os
import argparse
from timm.scheduler import create_scheduler
from config import cfg
from timm.data import Mixup
from sklearn.cluster import DBSCAN, KMeans
from utils.faiss_rerank import compute_jaccard_distance
from utils.faiss_rerank import batch_cosine_dist, cosine_dist
from utils.metrics import euclidean_distance

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def copy_state_dict(state_dict, model, strip=None):
    tgt_state = model.state_dict()
    copied_names = set()
    for name, param in state_dict.items():
        name = name.replace('module.', '')
        if strip is not None and name.startswith(strip):
            name = name[len(strip):]
        if name not in tgt_state:
            continue
        if isinstance(param, Parameter):
            param = param.data
        if param.size() != tgt_state[name].size():
            print('mismatch:', name, param.size(), tgt_state[name].size())
            continue
        tgt_state[name].copy_(param)
        copied_names.add(name)

    missing = set(tgt_state.keys()) - copied_names
    if len(missing) > 0:
        print("missing keys in state_dict:", missing)

    return model


class IterLoader:
    def __init__(self, loader, length=None):
        self.loader = loader
        self.length = length
        self.iter = None

    def __len__(self):
        if (self.length is not None):
            return self.length
        return len(self.loader)

    def new_epoch(self):
        self.iter = iter(self.loader)

    def next(self):
        try:
            return next(self.iter)
        except:
            self.iter = iter(self.loader)
            return next(self.iter)


def extract_features(model, data_loader, print_freq):
    model.eval()
    feats = []
    vids = []
    camids = []
    trkids = []
    with torch.no_grad():
        for i, (img, vid, camid, trkid, _) in enumerate(data_loader):
            img = img.to('cuda')
            feat = model(img)

            feats.append(feat)
            vids.extend(vid)
            camids.extend(camid)
            trkids.extend(trkid)

    feats = torch.cat(feats, dim=0)
    vids = torch.tensor(vids).cpu().numpy()
    camids = torch.tensor(camids).cpu().numpy()
    trkids = torch.tensor(trkids).cpu().numpy()

    return feats, vids, camids, trkids


def calc_distmat(feat):
    rerank_distmat = compute_jaccard_distance(feat, k1=30, k2=6, search_option=3)
    cosine_distmat = batch_cosine_dist(feat, feat).cpu().numpy()
    final_dist = rerank_distmat * 0.9 + cosine_distmat * 0.1

    return final_dist


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


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )

    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    #  parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        #  help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')
    parser.add_argument('--epochs', default=120, type=int)
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()
    
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)
    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)
    else:
        pass

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("reid_baseline", output_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    logger.info("EPS:%.f LA:%.f"%(cfg.STAGE2.EPS, cfg.STAGE2.LA))

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            #  logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    val_loader, num_query, testset = get_testloader_uda(cfg)
    aug_loader, num_query, _ = get_testloader_uda(cfg, aug=True)

    num_classes = 1500
    model = make_model(cfg, num_class=num_classes)
    initial_weights = torch.load(cfg.MODEL.PRETRAIN_PATH, map_location='cpu')
    copy_state_dict(initial_weights, model)


    #model_ema.classifier.weight.data.copy_(model.classifier.weight.data)
    if True:
        model.to(args.local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    for epoch in range(cfg.SOLVER.MAX_EPOCHS):

        if epoch % 3 == 0: # and epoch < 9) or (epoch % 6 == 0):
            target_features, target_labels, target_camids, target_trkids = extract_features(model, val_loader, print_freq=100)
            target_features = F.normalize(target_features, dim=1)
            aug_features, _, _, _ = extract_features(model, aug_loader, print_freq=100)
            aug_features = F.normalize(aug_features, dim=1)
            target_features = (aug_features + target_features) / 2.0
            P, neg_vec = compute_P2(target_features, target_features, target_camids, la=cfg.STAGE2.LA)
            target_features = meanfeat_sub(P, neg_vec, target_features, target_camids)

            #torch.save(target_features, 'target_features.pth')
            #target_features = torch.load('target_features.pth')

            gallery_trkids = target_trkids[num_query:]
            unique_trkids = sorted(list(set(gallery_trkids[gallery_trkids != -1])))
            gallery_features = target_features[num_query:]
            track_features = []
            for i, trkid in enumerate(unique_trkids):
                track_feature = torch.mean(gallery_features[gallery_trkids == trkid], dim=0, keepdim=True)
                tmp_indices = (gallery_trkids == trkid)
                gallery_features[tmp_indices] = gallery_features[tmp_indices] * 0.3 + track_feature * 0.7
            target_features[num_query:] = gallery_features

            final_dist = calc_distmat(target_features)
            final_dist[final_dist < 0.0] = 0.0
            final_dist[final_dist > 1.0] = 1.0
            cluster = DBSCAN(eps=cfg.STAGE2.EPS, min_samples=10, metric='precomputed', n_jobs=-1)
            pseudo_labels = cluster.fit_predict(final_dist)
            labelset = list(set(pseudo_labels[pseudo_labels >= 0]))
            idxs = np.where(np.in1d(pseudo_labels, labelset))
            psolabels = pseudo_labels[idxs]
            psofeatures = target_features[idxs]

            mean_features = []
            for label in labelset:
                mean_indices = (psolabels == label)
                mean_features.append(torch.mean(psofeatures[mean_indices], dim=0))
            mean_features = torch.stack(mean_features).cuda()

            num_classes = len(mean_features)
            model.num_classes = len(mean_features)
            model.classifier = nn.Linear(model.in_planes, len(mean_features), bias=False)
            model.classifier.weight = nn.Parameter(mean_features)

            del target_features
            
            pids = []
            new_dataset = []
            for i, (item, label) in enumerate(zip(testset, pseudo_labels)):
                if label == -1 or label not in labelset:
                    continue
                pids.append(label)
                new_dataset.append((item[0], label, item[2], item[3]))
            print('new class are {}, length of new dataset is {}'.format(len(set(pids)), len(new_dataset)))

        train_loader = IterLoader(get_trainloader_uda(cfg, new_dataset, num_classes))
        train_loader.new_epoch()
        loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)
        optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)

        do_uda_train(
            epoch,
            cfg,
            model,
            center_criterion,
            train_loader,
            val_loader,
            optimizer,
            optimizer_center,
            loss_func,
            num_query, args.local_rank
        )
    print(cfg.OUTPUT_DIR)

