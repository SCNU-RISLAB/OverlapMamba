


import os
import sys
p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)
sys.path.append('../tools/')
sys.path.append('../modules/')
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


import numpy as np
from tensorboardX import SummaryWriter
from tools.read_all_sets import overlap_orientation_npz_file2string_string_nparray
from modules.overlap_transformer import featureExtracter
from tools.read_samples import read_one_batch_pos_neg
from tools.read_samples import read_one_need_from_seq
np.set_printoptions(threshold=sys.maxsize)
import modules.loss as PNV_loss
from tools.utils.utils import *
from valid.valid_seq import validate_seq_faiss
import yaml
import argparse

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

class trainHandler():
    def __init__(self, rank, world_size, height=64, width=900, channels=5, norm_layer=None, use_transformer=True, use_mamba=False, lr = 0.001,
                 data_root_folder = None, train_set=None, training_seqs=None):
        super(trainHandler, self).__init__()

        self.rank = rank
        self.world_size = world_size
        setup(rank, world_size)

        self.height = height
        self.width = width
        self.channels = channels
        self.norm_layer = norm_layer
        self.use_transformer = use_transformer
        self.use_mamba = use_mamba
        self.learning_rate = lr
        self.data_root_folder = data_root_folder
        self.train_set = train_set
        self.training_seqs = training_seqs

        self.amodel = featureExtracter(channels=self.channels, use_transformer=self.use_transformer, use_mamba=self.use_mamba)
        self.device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
        self.amodel.to(self.device)
        self.amodel = DDP(self.amodel, device_ids=[rank])
        self.parameters  = self.amodel.parameters()
        self.optimizer = torch.optim.Adam(self.parameters, self.learning_rate)

        # self.optimizer = torch.optim.SGD(self.parameters, lr=self.learning_rate, momentum=0.9)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.9)
        # self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.98)

        self.traindata_npzfiles = train_set

        (self.train_imgf1, self.train_imgf2, self.train_dir1, self.train_dir2, self.train_overlap) = \
            overlap_orientation_npz_file2string_string_nparray(self.traindata_npzfiles)

        """change the args for resuming training process"""
        self.resume = True
        self.save_name = "../weights/train/trained_overlap_transformer24.pth.tar"

        """overlap threshold follows OverlapNet"""
        self.overlap_thresh = 0.3

    def train(self):

        epochs = 100

        """resume from the saved model"""
        if self.resume:
            resume_filename = self.save_name
            print("Resuming from ", resume_filename)
            checkpoint = torch.load(resume_filename, map_location=self.device)
            starting_epoch = checkpoint['epoch']
            self.amodel.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print("Training From Scratch ..." )
            starting_epoch = 0

        writer1 = SummaryWriter(comment=f"LR_0.xxxx_{self.rank}")

        for i in range(starting_epoch+1, epochs):
            if self.rank == 0:
                (self.train_imgf1, self.train_imgf2, self.train_dir1, self.train_dir2, self.train_overlap) = \
                    overlap_orientation_npz_file2string_string_nparray(self.traindata_npzfiles, shuffle=True)
                data = [self.train_imgf1, self.train_imgf2, self.train_dir1, self.train_dir2, self.train_overlap]
                for r in range(1, self.world_size):
                    dist.send_object(data, dst=r)
            else:
                data = dist.recv_object(src=0)
                self.train_imgf1, self.train_imgf2, self.train_dir1, self.train_dir2, self.train_overlap = data

            dist.barrier()

            if self.rank == 0:
                print("=======================================================================\n\n\n")
                print("training with seq: ", np.unique(np.array(self.train_dir1)))
                print("total pairs: ", len(self.train_imgf1))
                print("\n\n\n=======================================================================")

            loss_each_epoch = 0
            used_num = 0

            used_list_f1 = []
            used_list_dir1 = []

            for j in range(len(self.train_imgf1)):
                """
                    check whether the query is used to train before (continue_flag==True/False).
                    TODO: More efficient method
                """
                f1_index = self.train_imgf1[j]
                dir1_index = self.train_dir1[j]
                continue_flag = False
                for iddd in range(len(used_list_f1)):
                    if f1_index==used_list_f1[iddd] and dir1_index==used_list_dir1[iddd]:
                        continue_flag = True
                else:
                    used_list_f1.append(f1_index)
                    used_list_dir1.append(dir1_index)

                if continue_flag:
                    continue

                """read one query range image from KITTI sequences"""
                current_batch = read_one_need_from_seq(self.data_root_folder, f1_index, dir1_index)

                """
                    read several reference range images from KITTI sequences 
                    to consist of positive samples and negative samples
                """
                sample_batch, sample_truth, pos_num, neg_num = read_one_batch_pos_neg \
                    (self.data_root_folder,f1_index, dir1_index,
                     self.train_imgf1, self.train_imgf2, self.train_dir1, self.train_dir2, self.train_overlap,
                     self.overlap_thresh)

                """
                    the balance of positive samples and negative samples.
                    TODO: Update for better training results
                """
                use_pos_num = 18
                use_neg_num = 18
                if pos_num >= use_pos_num and neg_num>=use_neg_num:
                    sample_batch = torch.cat((sample_batch[0:use_pos_num, :, :, :], sample_batch[pos_num:pos_num+use_neg_num, :, :, :]), dim=0)
                    sample_truth = torch.cat((sample_truth[0:use_pos_num, :], sample_truth[pos_num:pos_num+use_neg_num, :]), dim=0)
                    pos_num = use_pos_num
                    neg_num = use_neg_num
                elif pos_num >= use_pos_num:
                    sample_batch = torch.cat((sample_batch[0:use_pos_num, :, :, :], sample_batch[pos_num:, :, :, :]), dim=0)
                    sample_truth = torch.cat((sample_truth[0:use_pos_num, :], sample_truth[pos_num:, :]), dim=0)
                    pos_num = use_pos_num
                elif neg_num >= use_neg_num:
                    sample_batch = sample_batch[0:pos_num+use_neg_num,:,:,:]
                    sample_truth = sample_truth[0:pos_num+use_neg_num, :]
                    neg_num = use_neg_num

                if neg_num == 0:
                    continue

                input_batch = torch.cat((current_batch, sample_batch), dim=0)

                input_batch.requires_grad_(True)
                self.amodel.train()
                self.optimizer.zero_grad()

                global_des = self.amodel(input_batch)
                o1, o2, o3 = torch.split(
                    global_des, [1, pos_num, neg_num], dim=0)
                MARGIN_1 = 0.5
                """
                    triplet_loss: Lazy for pos
                    triplet_loss_inv: Lazy for neg
                """
                loss = PNV_loss.triplet_loss(o1, o2, o3, MARGIN_1, lazy=False)
                # loss = PNV_loss.triplet_loss_inv(o1, o2, o3, MARGIN_1, lazy=False, use_min=True)
                loss.backward()
                self.optimizer.step()

                if self.rank == 0:
                    print(str(used_num), loss)

                if torch.isnan(loss):
                    if self.rank == 0:
                        print("Something error ...")
                        print(pos_num)
                        print(neg_num)

                loss_each_epoch = loss_each_epoch + loss.item()
                used_num = used_num + 1

            if self.rank == 0:
                print("epoch {} loss {}".format(i, loss_each_epoch/used_num))
                print("saving weights ...")
                self.scheduler.step()

                """save trained weights and optimizer states"""
                self.save_name = "../weights/train/trained_overlap_transformer"+str(i)+".pth.tar"

                torch.save({
                    'epoch': i,
                    'state_dict': self.amodel.module.state_dict(),
                    'optimizer': self.optimizer.state_dict()
                },
                    self.save_name)

                print("Model Saved As " + f'trained_overlap_transformer{self.rank}_' + str(i) + '.pth.tar')

                writer1.add_scalar("loss", loss_each_epoch / used_num, global_step=i)

                print("validating ......")
                with torch.no_grad():
                    top1_rate = validate_seq_faiss(self.amodel.module, "02")
                    writer1.add_scalar("top1_rate", top1_rate, global_step=i)

        cleanup()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    args = parser.parse_args()

    # load config ================================================================
    config_filename = '../config/config.yml'
    config = yaml.safe_load(open(config_filename))
    data_root_folder = config["data_root"]["data_root_folder"]
    training_seqs = config["training_config"]["training_seqs"]
    # ============================================================================

    traindata_npzfiles = [os.path.join(data_root_folder, seq, 'overlaps/train_set.npz') for seq in training_seqs]

    torch.cuda.set_device(args.local_rank)

    train_handler = trainHandler(rank=args.local_rank, world_size=torch.cuda.device_count(), height=64, width=900, channels=1, norm_layer=None, use_transformer=False, use_mamba=True, lr=0.000005,
                                 data_root_folder=data_root_folder, train_set=traindata_npzfiles, training_seqs = training_seqs)

    train_handler.train()
