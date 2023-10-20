import argparse

parser = argparse.ArgumentParser(description='CHV')
# models
parser.add_argument('--model', type=str, default='vgg16', help='vgg16,resnet,vit')
parser.add_argument('--feature_dim', type=int, default=4096, help='dim of vgg output(4096,2048,768)')

parser.add_argument('--data_name', type=str, default='cifar10', help='cifar10 or coco or nuswide')
parser.add_argument('--data_path', type=str, default='../../../datasets/data/cifar10', help='dataset path...')

parser.add_argument('--tau', default=0.3, type=float,
                    help='softmax temperature')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')

parser.add_argument('--epochs', type=int, default=80, help='training epoch')

parser.add_argument('--use_gpu', type=bool, default=True, help="use gpu ?")
# parser.add_argument('--gpu_ids', nargs='+', type=int, default=None, help='gpu devices ids')

parser.add_argument('--batch_size', type=int, default=64,
                    help='the batch size for training')  # batch_size most be even in this project
parser.add_argument('--eval_epochs', type=int, default=2)
parser.add_argument('--start_eval', type=int, default=0, help="the epoch when start to test")
parser.add_argument('--gamma', type=float, default=2.0, help='gamma for Cauchy distribution')
parser.add_argument('--lambda_q', type=float, default=0.01, help='lambda to balance the quantization loss')
parser.add_argument('--workers', type=int, default=8, help='number of data loader workers.')
# Hashing
parser.add_argument('--hash_bit', type=int, default=64, help='hash bit,it can be 8, 16, 32, 64, 128...')

parser.add_argument('--R', type=int, default=1000, help='MAP@R')
parser.add_argument('--T', type=float, default=0, help='Threshold for binary')
# Loss


# hyperbolic
parser.add_argument('--hyper_c', type=float, default=0.01, help='balance between hyperbolic space and Euclidean space')
parser.add_argument('--clip_r', type=float, default=2.3, help='feature clip radius')
parser.add_argument('--hyper_dim', type=int, default=128, help='dimension of hyperbolic embeddings')
parser.add_argument('--WOHP', action='store_true', help='disable prototypical contrastive learning')
parser.add_argument('--WOIC', action='store_true', help='disable instance contrastive learning')

parser.add_argument('--IC', action='store_true', help=' instance-wise contrastive learning without hierarchies')


parser.add_argument('--HIC', action='store_true', help=' hierarchical instance-wise contrastive learning')
parser.add_argument('--HPC', action='store_true', help=' prototypical contrastive learning')

parser.add_argument('--cluster_num', default='150,120,80', type=str,
                    help='number of clusters')
