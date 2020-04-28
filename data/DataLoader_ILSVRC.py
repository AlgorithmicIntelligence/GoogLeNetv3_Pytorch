#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 15:02:16 2020

@author: lds
"""
import torch.utils.data as data
import os, cv2
import numpy as np


class ILSVRC2012(data.Dataset):
    def __init__(self, ILSVRC_dir, classname_file, num_classes=1000, random_crop_times=1):
        self.random_crop_times = random_crop_times
        self.dirname_to_classnum = dict()
        self.classnum_to_classname = dict()
        with open(classname_file, 'r') as f:
            lines = f.read().splitlines()
            for i, line in enumerate(lines):
                self.dirname_to_classnum[line[:9]] = i
                self.classnum_to_classname[i] = line[10:]
        
        self.img_paths = list()
        self.labels = list()
        self.num_classes = num_classes
        for root, _, names in os.walk(ILSVRC_dir):
            for name in names:
                if name.endswith("JPEG"):
                    label = self.dirname_to_classnum[root.split('/')[-1]]
                    if label < self.num_classes:
                        self.img_paths.append(os.path.join(root, name))
                        self.labels.append(self.dirname_to_classnum[root.split('/')[-1]])
        this_dir = os.path.dirname(__file__)
        self.img_means = np.load(os.path.join(this_dir, 'img_means.npy'))
        self.img_means = self.img_means[16:16+224, 16:240]
        
        # for color jitter
        self.img_std = np.load(os.path.join(this_dir, 'img_std.npy'))
        self.eig_values = np.load(os.path.join(this_dir, 'eig_values.npy'))
        self.eig_vectors = np.load(os.path.join(this_dir, 'eig_vectors.npy'))
    
    def __getitem__(self, index):
        img = cv2.imread(self.img_paths[index])
        # img = self.color_jitter(img)
        h, w = img.shape[:2]
        min_ratio = np.max(320/np.array([h, w]))
        img_resize = cv2.resize(img, (int(min_ratio*w+0.5), int(min_ratio*h+0.5)))
        img_center = img_resize[round(img_resize.shape[0]/2-160):round(img_resize.shape[0]/2+160), 
                                round(img_resize.shape[1]/2-160):round(img_resize.shape[1]/2+160)]
            
        if self.random_crop_times == 0:
            x_start = 11
            y_start = 11
        else:
            x_start = np.random.randint(0, 21)
            y_start = np.random.randint(0, 21)
    
        img_crop = img_center[y_start:y_start+299, x_start:x_start+299]# - self.img_means#[y_start:y_start+224, x_start:x_start+224]

        #Flip Augmentation
        if np.random.randint(0,2):
            img_crop = img_crop[:, ::-1, :]
            
        img_result= img_crop / 255.
            
        return img_result, self.labels[index]
                
    def __len__(self):
        return len(self.labels) * max(1, self.random_crop_times)

    def PCA(self, img):
        img_avg = np.average(img, axis=(0, 1))
        img_std = np.std(img, axis=(0, 1))
        img_norm = (img - img_avg) / img_std
        img_cov = np.zeros((3, 3))
        for img_c in img_norm.reshape(-1, 3):
            img_cov += img_c.reshape(3, 1) * img_c.reshape(1, 3)
        img_cov /= len(img_norm.reshape(-1, 3))
        
        eig_values, eig_vectors = np.linalg.eig(img_cov)
        alphas = np.random.normal(0, 0.1, 3)
        img_reconstruct_norm = img_norm + np.sum((eig_values + alphas) * eig_vectors, axis=1)
        img_reconstruct = img_reconstruct_norm * img_std + img_avg
        img_reboundary = np.maximum(np.minimum(img_reconstruct, 255), 0).astype(np.uint8)
        return img_reboundary
    
    def color_jitter(self, img):
        alphas = np.random.normal(0, 0.1, 3)
        img_color_jitter = img + np.sum((self.eig_values + alphas) * self.eig_vectors, axis=1) * self.img_std
        img_color_jitter = np.maximum(np.minimum(img_color_jitter , 255), 0).astype(np.uint8)
        return img_color_jitter
                                                                                  

if __name__ == '__main__':
    def imshow(img):
        cv2.imshow('', img)
        cv2.waitKey()
        cv2.destroyAllWindows()
    batch_size_train = 128
    
    trainingSet = ILSVRC2012('/media/nickwang/StorageDisk/Dataset/ILSVRC2012/ILSVRC2012_img_train', 'dirname_to_classname')
    img, label = trainingSet.__getitem__(1000000)
    img_color_jitter = trainingSet.color_jitter(img*255)
    imshow(np.concatenate([img, img_color_jitter/255.], axis=1))
    imshow(img)
    imshow(img_color_jitter)
#    train_data_index = np.arange(len(trainingSet))  
#    np.random.shuffle(train_data_index)
#    i=0
#    
#    for i in range(len(trainingSet) // batch_size_train):
#        time_s = time.time()
#        a,b = trainingSet.__getitem__(range(i *batch_size_train, (i+1)*batch_size_train))
##        a,b = trainingSet.__getitem__(train_data_index[i *batch_size_train: (i+1)*batch_size_train])
#        time_e = time.time()
#        print(time_e - time_s)
