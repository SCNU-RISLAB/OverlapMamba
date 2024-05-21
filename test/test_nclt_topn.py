


import os
import sys

p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)

import numpy as np
import yaml
import time

rubish = []


def load_files(folder):
    file_paths = [os.path.join(dp, f) for dp, dn, fn in os.walk(
        os.path.expanduser(folder)) for f in fn]
    file_paths.sort()
    return file_paths


def cal_pr_curve(prediction_file_name, ground_truth_file_name, topn):
    des_dists = np.load(prediction_file_name)['arr_0']
    des_dists = np.asarray(des_dists, dtype='float32')
    des_dists = des_dists.reshape((len(des_dists), 3))

    ground_truth = np.load(ground_truth_file_name, allow_pickle='True')

    gt_num = 0
    all_num = 0
    check_out = 0

    img_paths_query = load_files(query_scan)
    print(len(img_paths_query))
    sum1 = 0
    for idx in range(0, len(img_paths_query), 5):

        gt_idxes = np.array(ground_truth[int(gt_num)])
        if gt_idxes.any():
            all_num += 1
        else:
            gt_num += 1
            continue

        gt_num += 1

        dist_lists_cur = des_dists[des_dists[:, 0] == idx, :]
        idx_sorted = np.argsort(dist_lists_cur[:, -1], axis=-1)

        for i in range(topn):
            if int(dist_lists_cur[idx_sorted[i], 1]) in gt_idxes:
                check_out += 1
                break

    print("top" + str(topn) + " recall: ", check_out / all_num)
    return check_out / all_num


def main(topn, ground_truth_file_name, dir_name):
    prediction_file_name = dir_name + "/predicted_des_L2_dis_bet_traj_forward.npz"

    topn_recall = cal_pr_curve(prediction_file_name, ground_truth_file_name, topn)

    return topn_recall


if __name__ == "__main__":
    # load config ================================================================
    config_filename = '../config/config_nclt.yml'
    config = yaml.safe_load(open(config_filename))
    ground_truth_file_name = config["file_root"]["gt_file"]
    query_scan = config["file_root"]["data_root_folder_test"]
    # ============================================================================
    topn = 20
    recall_sum = 0
    recall_list = []

    dir_name = ("./nclt_o1shift2/12.6.15/5_15dis/")

    for i in range(1, topn + 1):
        rec = main(i, ground_truth_file_name, dir_name)
        recall_sum += rec
        if i == 1:
            print("AR@1 = ", recall_sum / i)
        if i == 5:
            print("AR@5 = ", recall_sum / i)
        if i == 20:
            print("AR@20 = ", recall_sum / i)
        recall_list.append(rec)

    print(recall_list)

    np.save(dir_name + "/recall_list", np.array(recall_list))
