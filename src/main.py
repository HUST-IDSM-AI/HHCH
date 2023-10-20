import datetime
import random

import os


from Loss import contrastive_loss, parse_proto, contrastive_proto, \
    hierarchical_contrastive_loss
from dataLoader.DataSet_loader import getDataLoader
from models.clustering import hierarchical_clustering_K_Means


os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
import torchvision
from tqdm import tqdm
import numpy as np

from models.model import MainModel
from options import parser

from utils import util
from utils.logger import Logger
import pickle
import torch.nn.functional as F
from utils.measure_utils import get_precision_recall_by_Hamming_Radius_optimized, mean_average_precision
from utils.util import adjust_learning_rate

"""
official implementation of hierarchical hyperbolic contrastive hashing (HHCH)
"""


def set_seed(seed):
    # np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


set_seed(8888)


def compute_features(option, train_loader, model, epoch):
    Logger.info("computing feature")
    model.train()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # features = torch.zeros(len(train_loader.dataset), option.hash_bit).cuda()
    if option.hyper_c != 0:
        features_all = torch.zeros(len(train_loader.dataset), 128).to(device)
    else:
        features_all = torch.zeros(len(train_loader.dataset), option.hash_bit).to(device)
    for i, ((feature_1, feature_2, feature_origin), target, index) in enumerate(tqdm(train_loader)):
        with torch.no_grad():
            feature_1 = feature_1.to(device)
            feature_2 = feature_2.to(device)
            target = target.to(device)
            feature_origin = feature_origin.to(device)
            feat, _, _, _, _, _ = model((feature_1, feature_2, feature_origin), Train=True)
            features_all[index] = feat
            if i == 0:
                target_all = torch.zeros(len(train_loader.dataset), target.shape[1]).to(device)
                target_all[index] = target.float()
            else:
                target_all[index] = target.float()
    return features_all.detach()



def main(option, state):
    train_loader, test_loader, database_loader = getDataLoader(option)
    model = MainModel(option)

    optimizer = torch.optim.Adam(model.getParams(), lr=option.lr)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # criterion = criterion.cuda()

    #########################Epoch#################
    state['best_MAP'] = 0.0
    state['best_epoch'] = 0
    state['final_result'] = None
    state['filename_previous_best'] = None
    state['iter'] = 0
    for epoch in range(option.epochs):
        ############################
        lr = adjust_learning_rate(option, optimizer, epoch)
        # lr = option.lr
        state['epoch'] = epoch
        ############################
        Logger.divider("Epoch[{}]-lr{}".format(str(epoch), str(lr)))

        if option.IC:
            cluster_results = None
        else:

            all_feat = compute_features(option, train_loader=train_loader, model=model,
                                        epoch=epoch).detach().cpu().numpy().astype(
                'float64')
            # run hierarchical k means to construct hierarchical structure
            cluster_results = hierarchical_clustering_K_Means(option, all_feat, option.cluster_num, option.hyper_c)

        torch.cuda.empty_cache()
        loss_epoch = train_step(model, optimizer, train_loader, cluster_results, epoch, state, option)
        Logger.info("epoch: {} Loss: {}".format(epoch, loss_epoch))
        if epoch % option.eval_epochs == 0 and epoch >= option.start_eval:
            (Precision_TH, Recall_TH, MAP_TH), (MAP_Rank, Recall_Rank, P_Rank) = test_step(option, state, model,
                                                                                           test_loader,
                                                                                           database_loader, epoch)
            Logger.info(
                "epoch {0} Result：{1}\n".format(epoch,
                                                ((Precision_TH, Recall_TH, MAP_TH), (MAP_Rank, Recall_Rank, P_Rank))))

            util.saveStatus(option, state, epoch, MAP_Rank,
                            ((Precision_TH, Recall_TH, MAP_TH), (MAP_Rank, Recall_Rank, P_Rank)))

            is_best = MAP_Rank >= state['best_MAP']
            Logger.info("MAP epoch {}\tMAP_best {}\tIs_best {}\tBest epoch {}".format(MAP_Rank, state['best_MAP'],
                                                                                      MAP_Rank >= state[
                                                                                          'best_MAP'],
                                                                                      state['best_epoch']))
            #####################
            model_dict = {
                'epoch': epoch,
                'model_dict': model.state_dict() if option.use_gpu and torch.cuda.is_available() else model.state_dict(),
                'optimizer_hash_dict': optimizer.state_dict(),
                'best_MAP': state['best_MAP']
            }
            util.save_checkpoint(option, state, model_dict, is_best)
        #####################
    Logger.info("<======[Final Result]=====>")
    Logger.info(
        "Hash Pool Radius :{}\nMAP1 :{:.4f}\t Recall1 {:.4f}\tPrecision1 {:.4f}\t MAP2 {:.4f} \t Recall2 {:.4f} "
        "\t Precision2 {:.4f} ".format(
            option.R, state['final_result'][1][0], state['final_result'][1][1],
            state['final_result'][1][2],
            state['final_result'][0][2],
            state['final_result'][0][1], state['final_result'][0][0]
        ))


def train_step(model, optimizer, train_loader, cluster_results, epoch, state, option=None):
    model.train()
    loss_epoch = list()

    num_cluster = option.cluster_num
    if not option.IC:
        proto_corresponding, proto_index = parse_proto(cluster_results, num_cluster)
    train_loader = tqdm(train_loader, desc="epoch[" + str(epoch) + "]==>training")
    negatives_num = []
    for i, ((feature_1, feature_2, feature_origin), target, index) in enumerate(train_loader):
        optimizer.zero_grad()
        state['iter'] += 1

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        feature_1 = feature_1.to(device)
        feature_2 = feature_2.to(device)
        feature_origin = feature_origin.to(device)
        _, _, p_1, p_2, h_1, h_2 = model((feature_1, feature_2, feature_origin), Train=True)
        #############quantization loss###########
        quantization_loss = torch.mean((torch.abs(h_1) - torch.tensor(1.0).cuda()) ** 2) + torch.mean(
            (torch.abs(h_2) - torch.tensor(1.0).cuda()) ** 2)
        quantization_loss = quantization_loss / 2
        ###########contrastive loss###############
        if option.HIC:
            instance_contrastive_loss = (hierarchical_contrastive_loss(p_1, p_2, center_index=proto_index[index],
                                                                       tau=option.tau,
                                                                       hyper_c=option.hyper_c) +
                                         hierarchical_contrastive_loss(
                                             p_2, p_1, center_index=proto_index[index],
                                             tau=option.tau, hyper_c=option.hyper_c)) / 2

        elif option.IC:

            instance_contrastive_loss = (contrastive_loss(p_1, p_2, tau=option.tau, hyper_c=option.hyper_c)[0] +
                                         contrastive_loss(p_2, p_1, tau=option.tau, hyper_c=option.hyper_c)[0]) / 2
        else:
            instance_contrastive_loss = 0.
        if option.HPC:
            proto_corresponding_batch, proto_index_batch = proto_corresponding[index], proto_index[index]
            proto_contrastive_loss = contrastive_proto(p_1, p_2, center_corresponding=proto_corresponding_batch,
                                                       center_index=proto_index_batch, results=cluster_results,
                                                       tau=option.tau,
                                                       hyper_c=option.hyper_c)
        else:
            proto_contrastive_loss = 0.


        loss = 1. * instance_contrastive_loss + option.lambda_q * quantization_loss + proto_contrastive_loss
        loss_epoch.append(loss.item())
        # optimization
        loss.backward()
        optimizer.step()
    return np.mean(loss_epoch)


def predict_hash_code(option, state, model, data_loader, epoch, database_type: str):
    model.eval()
    data_loader = tqdm(data_loader, desc="epoch[" + str(epoch) + "]==>" + database_type + "==>Testing:")
    for i, (input, target) in enumerate(data_loader):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        images = input.to(device)
        target = target.to(device)
        hash_code = model(images, False)
        if i == 0:
            all_codes = hash_code
            all_label = target
        else:
            all_codes = torch.cat((all_codes, hash_code), dim=0)
            all_label = torch.cat((all_label, target), dim=0)
    if option.hyper_c == 0:
        return F.normalize(all_codes, dim=1), all_label
    else:
        return all_codes, all_label


def test_step(option, state, model, test_loader, database_loader, epoch):
    model.eval()
    rB, rL = predict_hash_code(option, state, model, database_loader, epoch, database_type="database")
    qB, qL = predict_hash_code(option, state, model, test_loader, epoch, database_type='testbase')

    Logger.info("===> start calculate MAP!\n")
    database_hashcode_numpy = rB.detach().cpu().numpy().astype('float32')

    del rB
    database_labels_numpy = rL.detach().cpu().numpy().astype('int8')

    del rL
    testbase_hashcode_numpy = qB.detach().cpu().numpy().astype('float32')

    del qB
    testbase_labels_numpy = qL.detach().cpu().numpy().astype('int8')

    del qL
    Logger.info("===> start calculate MAP!\n")

    ######################### PR record #################
    # calculatePrecision_R(database_hashcode_numpy, database_labels_numpy, testbase_hashcode_numpy, testbase_labels_numpy,
    #                      option.hyper_c)
    Precision_TH, Recall_TH, MAP_TH = get_precision_recall_by_Hamming_Radius_optimized(
        database_hashcode_numpy,
        database_labels_numpy,
        testbase_hashcode_numpy,
        testbase_labels_numpy, fine_sign=True)
    MAP_Rank, Recall_Rank, P_Rank = mean_average_precision(database_hashcode_numpy,
                                                           testbase_hashcode_numpy,
                                                           database_labels_numpy,
                                                           testbase_labels_numpy, option)

    del database_hashcode_numpy, testbase_hashcode_numpy, database_labels_numpy, testbase_labels_numpy
    return (Precision_TH, Recall_TH, MAP_TH), (MAP_Rank, Recall_Rank, P_Rank)


if __name__ == '__main__':

    start_time = datetime.datetime.now()
    Logger.info("\t\tstart program\t\t")
    option = parser.parse_args()
    Logger.divider("print option")
    for k, v in vars(option).items():
        Logger.info('\t{}: {}'.format(k, v))
    state = {'start_time': start_time}
    option.cluster_num = option.cluster_num.split(',')
    option.cluster_num = [int(x) for x in option.cluster_num]
    main(option, state)
    end_time = datetime.datetime.now()
    Logger.divider("END {}".format(Logger.getTimeStr(end_time)))
    pass
