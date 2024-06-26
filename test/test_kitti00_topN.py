#!/usr/bin/env python3
# Developed by Junyi Ma, Xieyuanli Chen, and Jun Zhang
# This file is covered by the LICENSE file in the root of the project OverlapTransformer:
# https://github.com/haomo-ai/OverlapTransformer/
# Brief: calculate Recall@N using the prediction files generated by test_kitti00_prepare.py


import os
import sys
p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)
    
import copy
import matplotlib.pyplot as plt
import numpy as np
import yaml
from sklearn import metrics



def cal_topN(prediction_file_name, ground_truth_file_name, topn):
    precisions = []
    recalls = []


    # loading overlap predictions
    des_dists = np.load(prediction_file_name)['arr_0']
    des_dists = np.asarray(des_dists, dtype='float32')
    des_dists = des_dists.reshape((len(des_dists), 3))


    # loading ground truth in terms of distance
    ground_truth = np.load(ground_truth_file_name, allow_pickle='True')['arr_0']


    all_have_gt = 0
    tps = 0


    for idx in range(0,len(ground_truth)-1):
        gt_idxes = ground_truth[int(idx)]

        if not gt_idxes.any():
            continue

        all_have_gt += 1
        for t in range(topn):
            if des_dists[des_dists[:,0]==int(idx),:][t, 1] in gt_idxes:
                tps += 1
                break

    recall_topN = tps/all_have_gt
    print(recall_topN)


    return recall_topN




def test_with_topN(topn, ground_truth_file_name):

    prediction_file_name = dir_name + "predicted_des_L2_dis.npz"
    recall_topN = cal_topN(prediction_file_name, ground_truth_file_name, topn)
    return recall_topN



if __name__ == "__main__":
    # load config ================================================================
    config_filename = '../config/config.yml'
    config = yaml.safe_load(open(config_filename))
    ground_truth_file_name = config["test_config"]["gt_file"]
    # ============================================================================
    dir_name = "./test_/"
    # topn = 46  # for KITTI 00 top1%
    topn = 38  # for KITTI 00 top1%
    recall_list = []
    for i in range(1, topn):
        print("top"+str(i)+": ")
        rec = test_with_topN(i, ground_truth_file_name)
        recall_list.append(rec)
    print(recall_list)
    np.save(dir_name + "recall_list", np.array(recall_list))


