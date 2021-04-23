# encoding: utf-8

import glob
import re
import xml.dom.minidom as XD
import os.path as osp
from .bases import BaseImageDataset
import os

class AIC_SIM_SPGAN(BaseImageDataset):
    dataset_dir = 'AIC21/AIC21_Track2_ReID'
    sim_dataset_dir = 'AIC21/AIC21_Track2_ReID_Simulation'

    def __init__(self, root='../data', verbose=True,crop_test=False, **kwargs):
        super(AIC_SIM_SPGAN, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_track_path = osp.join(self.dataset_dir, 'train_track.txt')
        self.test_track_path = osp.join(self.dataset_dir, 'test_track.txt')
        self.sim_dataset_dir = osp.join(root, self.sim_dataset_dir)
        self.crop_test = crop_test

        self.train_dir = osp.join(self.dataset_dir, 'image_train')
        self.query_dir = osp.join(self.dataset_dir, 'image_query')
        self.gallery_dir = osp.join(self.dataset_dir, 'image_test')
        self.crop_train = osp.join(self.dataset_dir, 'training_part_seg/cropped_patch/image_train')

        self._check_before_run()

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir_test(self.query_dir, relabel=False)
        gallery = self._process_dir_test(self.gallery_dir, relabel=False, query=False)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)
        self.sim_train_dir = osp.join(self.sim_dataset_dir, 'sys_image_train_tr')
        sim_train = self._process_sim(self.sim_train_dir, begin_id= self.num_train_pids, relabel=True)
        self.train = self.train + sim_train
        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        

        if verbose:
            print("=> AIC_SIM_SPGAN loaded")
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

    def _process_dir(self, dir_path, relabel=False, if_track=False):
        xml_dir = osp.join(self.dataset_dir, 'train_label.xml')
        info = XD.parse(xml_dir).documentElement.getElementsByTagName('Item')

        pid_container = set()
        for element in range(len(info)):
            pid = int(info[element].getAttribute('vehicleID'))
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        _, _, frame2trackID = self._process_track(path=self.train_track_path) #### Revised

        for element in range(len(info)):
            pid, camid = map(int, [info[element].getAttribute('vehicleID'), info[element].getAttribute('cameraID')[1:]])
            image_name = str(info[element].getAttribute('imageName'))
            if pid == -1: continue  # junk images are just ignored
            if relabel: pid = pid2label[pid]
            trackid = frame2trackID[image_name]
            dataset.append((osp.join(dir_path, image_name), pid, camid,trackid))
            dataset.append((osp.join(self.crop_train, image_name), pid, camid,trackid))
        return dataset

    def _process_dir_test(self, dir_path, relabel=False, query=True):
        _, _, frame2trackID = self._process_track(path=self.test_track_path)
        if query:
            xml_dir = osp.join(self.dataset_dir, 'query_label.xml')
            crop_dir = dir_path.replace('image_query', 'cropped_aic_test/image_query')
        else:
            xml_dir = osp.join(self.dataset_dir, 'test_label.xml')
            crop_dir = dir_path.replace('image_gallery', 'cropped_aic_test/image_gallery')
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
            if self.crop_test:
                dataset.append((osp.join(crop_dir, image_name), -1, camid,trackid))
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

    def _process_sim(self,dir_path,begin_id, relabel=False):
        img_path = os.listdir(dir_path)
        train = []
        pid_container = set()
        for img in img_path:
            pid_container.add(int(img[:5]))

        pid2label = {pid: (label+begin_id) for label, pid in enumerate(pid_container)}
        for img in sorted(img_path):
            camid = int(img[7:10])
            pid = int(img[:5])
            if relabel: pid = pid2label[pid]
            train.append((osp.join(dir_path, img), pid, camid,-1))
        return train

if __name__ == '__main__':
    aic = AIC(root='/home/michuan.lh/datasets')
