#!/usr/bin/python3 
# -*- coding: utf-8 -*-

import numpy as np
import caffe

# to-do: use serveral caffe net using different GPUs respectively

batch_size = 1
caffe.set_device(0)
caffe.set_mode_gpu()

net = caffe.Net('./my_deploy.prototxt', './snapshots/itracker_iter_92000.caffemodel', caffe.TEST)

mean_face = np.load('./mean_face_224.npy').reshape(1, 3, 224, 224)
mean_right = np.load('./mean_right_224.npy').reshape(1, 3, 224, 224)
mean_left = np.load('./mean_left_224.npy').reshape(1, 3, 224, 224)

