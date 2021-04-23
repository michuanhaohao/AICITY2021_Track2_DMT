# encoding: utf-8

import glob
import re
import xml.dom.minidom as XD
import os.path as osp
from .bases import BaseImageDataset
import os

class SIM_VIEW(BaseImageDataset):
    sim_dataset_dir = 'AIC21/AIC21_Track2_ReID_Simulation'
    dataset_dir = 'AIC21/AIC21_Track2_ReID'


    def __init__(self, root='/home/michuan.lh/datasets', verbose=True, **kwargs):
        super(SIM_VIEW, self).__init__()
        self.sim_dataset_dir = osp.join(root, self.sim_dataset_dir)
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.test_track_path = osp.join(self.dataset_dir, 'test_track.txt')
        self.query_dir = osp.join(self.dataset_dir, 'image_query')
        self.gallery_dir = osp.join(self.dataset_dir, 'image_test')


        self.sim_train_dir = osp.join(self.sim_dataset_dir, 'sys_image_train')
        sim_train = self._process_sim(self.sim_train_dir, begin_id= 0, relabel=True)
        self.train = sim_train
        self.query = self._process_dir_test(self.query_dir, relabel=False)
        self.gallery = self._process_dir_test(self.gallery_dir, relabel=False, query=False)
        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)

        if verbose:
            print("=> SIM_VIEW loaded")
            self.print_dataset_statistics(self.train, self.query, self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))


    def _process_sim(self,dir_path,begin_id, relabel=False):
        img_path = os.listdir(dir_path)
        train = []
        label_path = osp.join(self.sim_dataset_dir, 'train_label.xml')
        info = XD.parse(label_path).documentElement.getElementsByTagName('Item')
        view_ids = set()
        for element in range(len(info)):
            image_name = str(info[element].getAttribute('imageName'))
            camid = int(info[element].getAttribute('cameraID')[1:])
            orientation = max(float(info[element].getAttribute('orientation')) - 0.1, 0.0)
            view_id = int(orientation/10.0)
            if view_id > 13: view_id = view_id - 5
            train.append((osp.join(dir_path, image_name), view_id, camid,-1))
            view_ids.add(view_id)
        return train

    def _process_dir_test(self, dir_path, relabel=False, query=True):
        _, _, frame2trackID = self._process_track(path=self.test_track_path)
        if query:
            xml_dir = osp.join(self.dataset_dir, 'query_label.xml')
        else:
            xml_dir = osp.join(self.dataset_dir, 'test_label.xml')
        info = XD.parse(xml_dir).documentElement.getElementsByTagName('Item')
        dataset = []
        for element in range(len(info)):
            camid = int(info[element].getAttribute('cameraID')[1:])
            image_name = str(info[element].getAttribute('imageName'))
            if query:
                trackid = -1
            else:
                trackid = frame2trackID[image_name]
            dataset.append((osp.join(dir_path, image_name), -1, camid,trackid))
        return dataset

    def _process_track(self,path): #### Revised

        file = open(path)
        tracklet = dict()
        frame2trackID = dict()
        nums = []
        for track_id, line in enumerate(file.readlines()):
            curLine = line.strip().split(" ")
            nums.append(len(curLine))
            #  curLine = list(map(eval, curLine))
            tracklet[track_id] =  curLine
            for frame in curLine:
                frame2trackID[frame] = track_id
        return tracklet, nums, frame2trackID

if __name__ == '__main__':
    aic = AIC(root='/home/michuan.lh/datasets')
