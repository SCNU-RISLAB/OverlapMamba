


import os
import sys

p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)
sys.path.append('../tools/')
sys.path.append('../modules/')

import matplotlib.pyplot as plt
import torch
import yaml
import numpy as np

from modules.overlap_transformer import featureExtracter
from tools.read_samples_haomo import read_one_need_from_seq
from tools.read_samples_haomo import read_one_need_from_seq_test

np.set_printoptions(threshold=sys.maxsize)
from tqdm import tqdm
import faiss
from tools.utils.utils import *


def shift(tensor, dim, index):
    length = tensor.size(dim)
    shifted_tensor = torch.cat((tensor.narrow(dim, index, length - index),
                                tensor.narrow(dim, 0, index)), dim=dim)
    return shifted_tensor

def unshift(tensor, dim, index):
    length = tensor.size(dim)
    unshifted_tensor = torch.cat((tensor.narrow(dim, length - index, index),
                                  tensor.narrow(dim, 0, length - index)), dim=dim)
    return unshifted_tensor

class testHandler():
    def __init__(self, height=32, width=900, channels=1, norm_layer=None, use_transformer=False,
                 data_root_folder=None, data_root_folder_test=None, test_weights=None):
        super(testHandler, self).__init__()

        self.height = height
        self.width = width
        self.channels = channels
        self.norm_layer = norm_layer
        self.use_transformer = use_transformer
        self.data_root_folder = data_root_folder
        self.data_root_folder_test = data_root_folder_test

        self.amodel = featureExtracter(channels=self.channels, use_transformer=self.use_transformer)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.amodel.to(self.device)
        self.parameters = self.amodel.parameters()

        self.test_weights = test_weights

    def eval(self):

        print("Resuming From ", self.test_weights)
        checkpoint = torch.load(self.test_weights)
        self.amodel.load_state_dict(checkpoint['state_dict'])

        range_image_paths_database = load_files(self.data_root_folder)
        print("scan number of database: ", len(range_image_paths_database))

        des_list = np.zeros((len(range_image_paths_database), 512))  # for forward driving
        for j in tqdm(range(0, len(range_image_paths_database))):
            f1_index = str(j).zfill(6)
            current_batch = read_one_need_from_seq(self.data_root_folder, f1_index)
            # current_batch_double = torch.cat((current_batch, current_batch), dim=-1)
            # current_batch_inv = current_batch_double[:, :, :, 450:1350]
            # print(current_batch.shape)
            # current_batch = torch.cat((current_batch, current_batch_inv), dim=0)




            # print(current_batch.shape)
            self.amodel.eval()
            #current_batch_des = self.amodel(current_batch)
            index222 = int(torch.rand(1).item() * 900)
            input_batch_shift = shift(current_batch, 3, index222)

            global_des = self.amodel(current_batch)
            global_des_shift = self.amodel(input_batch_shift)
            global_des_add = torch.cat((global_des, global_des_shift), dim=1)
            des_list[(j), :] = global_des_add[0, :].cpu().detach().numpy()

        des_list = des_list.astype('float32')

        nlist = 1
        k = 50
        d = 512
        quantizer = faiss.IndexFlatL2(d)

        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
        assert not index.is_trained

        index.train(des_list)
        assert index.is_trained
        index.add(des_list)
        row_list = []

        range_image_paths_query = load_files(self.data_root_folder_test)
        print("scan number of query: ", len(range_image_paths_query))

        for i in range(0, len(range_image_paths_query), 5):

            i_index = str(i).zfill(6)
            current_batch = read_one_need_from_seq_test(self.data_root_folder_test, i_index)  # compute 1 descriptors
            # current_batch_double = torch.cat((current_batch, current_batch), dim=-1)
            # current_batch_inv = current_batch_double[:, :, :, 450:1350]
            # current_batch = torch.cat((current_batch, current_batch_inv), dim=0)
            self.amodel.eval()
            index111 = int(torch.rand(1).item() * 900)
            input_batch_shift = shift(current_batch, 3, index111)

            global_des = self.amodel(current_batch)
            global_des_shift = self.amodel(input_batch_shift)
            global_des_add = torch.cat((global_des, global_des_shift), dim=1)
            #current_batch_des = self.amodel(current_batch)  # torch.Size([(1+pos_num+neg_num)), 256])
            print()
            des_list_current = global_des_add[0, :].cpu().detach().numpy()

            D, I = index.search(des_list_current.reshape(1, -1), k)  # actual search

            for j in range(D.shape[1]):
                one_row = np.zeros((1, 3))
                one_row[:, 0] = i
                one_row[:, 1] = I[:, j]
                one_row[:, 2] = D[:, j]
                row_list.append(one_row)
                print("query:" + str(i) + "---->" + "database:" + str(I[:, j]) + "  " + str(D[:, j]))

        row_list_arr = np.array(row_list)
        dir_name = "./nclt_cvtnet/12.2.5/5_15dis/"
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        np.savez_compressed(dir_name + "predicted_des_L2_dis_bet_traj_forward", row_list_arr)


if __name__ == '__main__':
    # data
    # load config ================================================================
    config_filename = '../config/config_haomo.yml'
    config = yaml.safe_load(open(config_filename))
    data_root_folder = config["file_root"]["data_root_folder"]
    data_root_folder_test = config["file_root"]["data_root_folder_test1"]
    test_weights = config["file_root"]["test_weights"]
    # ============================================================================

    test_handler = testHandler(height=32, width=900, channels=1, norm_layer=None, use_transformer=False,
                               data_root_folder=data_root_folder, data_root_folder_test=data_root_folder_test,
                               test_weights=test_weights)

    test_handler.eval()
