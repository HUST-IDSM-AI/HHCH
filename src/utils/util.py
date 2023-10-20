import math, os, sys
import pickle
import shutil
from urllib.request import urlretrieve
import torch
from PIL import Image
from tqdm import tqdm
import numpy as np
import random
import platform
# DEBUG switch
from utils.logger import Logger

DEBUG_UTIL = False


def getDatabaseHashPoolPath(option, state):
    time = Logger.getTimeStr(state['start_time'])

    path = "../data/" + option.data_name + "/" + option.data_name + "_" + str(option.hash_bit) + "bit_" + str(
        state['epoch']) + "e_" + time + "_database.pkl"

    return path


def getTrainbaseHashPoolPath(option, state):
    time = Logger.getTimeStr(state['start_time'])
    path = "../data/" + option.data_name + "/" + option.data_name + "_" + str(option.hash_bit) + "bit_" + str(
        state['epoch']) + "e_" + time + "_trainbase.pkl"
    return path


def adjust_learning_rate(option, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every  epochs"""
    # for param_group in optimizer.param_groups:
    # param_group['lr'] = lr
    if epoch < 30:
        lr = option.lr
    else:
        lr = option.lr * (0.9 ** (epoch // 20))
        optimizer.param_groups[0]['lr'] = lr
        optimizer.param_groups[1]['lr'] = lr
    # optimizer.param_groups[2]['lr'] = lr
    # optimizer.param_groups[2]['lr'] = lr

    return lr


def saveStatus(option, state, epoch, MAP, result_all=None):
    # save hash center
    # print("!!!!!!!  type {}".format(hashCenter_pre.))
    # np.save('../data/' + self.option.data_name + '/centers.npy', hashCenter_pre.detach().cpu().numpy())
    if MAP >= state['best_MAP']:
        state['best_MAP'] = MAP
        state['best_epoch'] = epoch
        state['final_result'] = result_all
        # np.save(utils.getWeightBestPath(self.option, self.state), centerWeight_train)
    elif epoch >= option.epochs - 1:
        # np.save('../data/' + self.option.data_name + '/finalweight.npy', centerWeight_train)
        pass
    else:
        pass


def calc_ham_dist_2(outputs1, outputs2, option):
    ip = torch.mm(outputs1, outputs2.t())
    mod = torch.mm((outputs1 ** 2).sum(dim=1).reshape(-1, 1), (outputs2 ** 2).sum(dim=1).reshape(1, -1))
    cos = ip / mod.sqrt()
    hash_bit = outputs1.shape[1]
    dist_ham = hash_bit / 2.0 * (1.0 - cos)

    # dist_ham = torch.where(dist_ham < 0., dist_ham + 0.01, dist_ham)

    return dist_ham


def intra_distance(hash_code, label, option):
    """
    :param hash_code: numpy array
    :param label: numpy array
    :return:
    """
    mean_all_class = []
    for i in range(label.shape[1]):
        index = np.argwhere(label[:, i] > 0.5).flatten()
        all_code = hash_code[index, :]
        intra_num = all_code.shape[0]
        center = np.mean(all_code, axis=0).reshape((1, all_code.shape[1]))
        center = center.repeat(intra_num, axis=0)
        dist = calc_ham_dist_2(torch.tensor(all_code).float(), torch.tensor(center).float(), option)[:, 0]
        # dist = dist - torch.triu(dist)
        # mean = dist.sum() / (intra_num * (intra_num - 1) / 2)
        # mean_all_class.append(mean)
        mean = torch.mean(dist)
        mean_all_class.append(mean)
    return np.mean(mean_all_class)


def inter_distance(hash_code, label, option):
    """
    :param hash_code: numpy array
    :param label: numpy array
    :return:
    """
    all_centers = []
    for i in range(label.shape[1]):
        index = np.argwhere(label[:, i] > 0.5).flatten()
        all_code = hash_code[index, :]
        # intra_num = all_code.shape[0]
        center = np.mean(all_code, axis=0)
        all_centers.append(center)

    all_centers = np.array(all_centers)
    inter_num = all_centers.shape[0]
    dist = calc_ham_dist_2(torch.tensor(all_centers).float(), torch.tensor(all_centers).float(), option)
    dist = dist - torch.triu(dist)
    mean = dist.sum() / (inter_num * (inter_num - 1) / 2)

    return mean


def save_checkpoint(option, state, model_dict, is_best, filename='checkpoint.pth.tar'):
    save_model_path = '../data/' + option.data_name + '/models'
    if option.data_name is not None:
        filename_ = filename
        filename = os.path.join(save_model_path, filename_)
        if not os.path.exists(save_model_path):
            os.makedirs(save_model_path)
    Logger.info('save models {filename}\n'.format(filename=filename))
    torch.save(model_dict, filename)
    if is_best:
        filename_best = 'model_best.pth.tar'
        if save_model_path is not None:
            filename_best = os.path.join(save_model_path, filename_best)
        shutil.copyfile(filename, filename_best)
        if save_model_path is not None:
            if state['filename_previous_best'] is not None and os.path.exists(state['filename_previous_best']):
                os.remove(state['filename_previous_best'])
            filename_best = os.path.join(save_model_path,
                                         'model_best_{score:.4f}.pth.tar'.format(score=model_dict['best_MAP']))
            shutil.copyfile(filename, filename_best)
            state['filename_previous_best'] = filename_best


def loadHashPool(path, type='testbase'):
    # getDatabaseHashPoolPath()
    file = open(path, 'rb')
    start = True
    if type == 'testbase':
        while True:
            try:
                data = pickle.load(file)
                hashcode_batch = data['output'].cpu()
                hashcode_batch.require_grad = False
                label_batch = data['target'].cpu()
                label_batch.require_grad = False
                if start:
                    hash_pool = hashcode_batch
                    labels = label_batch
                    start = False
                else:
                    hash_pool = torch.cat((hash_pool, hashcode_batch), dim=0)
                    labels = torch.cat((labels, label_batch), dim=0)
            except Exception:
                break
        return hash_pool, labels
    elif type == 'database':
        while True:
            try:
                data = pickle.load(file)
                hashcode_batch = data['output'].cpu()
                label_batch = data['target'].cpu()
                hashcode_batch.require_grad = False
                label_batch.require_grad = False
                # centers = data['center'].cpu()
                if start:
                    hash_pool = hashcode_batch
                    labels = label_batch
                    # centers_all = centers
                    start = False
                else:
                    hash_pool = torch.cat((hash_pool, hashcode_batch), dim=0)
                    labels = torch.cat((labels, label_batch), dim=0)
                    # centers_all = torch.cat((centers_all, centers), dim=0)
            except Exception:
                break
        return hash_pool, labels


def getTestbaseHashPoolPath(option, state):
    time = Logger.getTimeStr(state['start_time'])

    path = "../data/" + option.data_name + "/" + option.data_name + "_" + str(option.hash_bit) + "bit_" + str(
        state['epoch']) + "e_" + time + "_testbase.pkl"
    return path


if __name__ == "__main__":
    option = {}
    state = {}
    code, labels = loadHashPool(option, state,
                                "D:\python\CSQ_NEW\data\\voc\\voc_64bit_21e_[0818-15_53_42]_testbase.pkl")
    code = code.numpy()
    labels = labels.numpy()
    print()

    pass
    #     # voc_adj.pkl path
    #     dir_voc_adj = "./data/voc/voc_adj.pkl"
    #     y = gen_A(20, 0.4, str(dir_voc_adj))
    #     #print(y)
