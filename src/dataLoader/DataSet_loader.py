import torch
from torchvision.transforms import transforms
# from common.utils import MultiScaleCrop, Warp
from dataLoader.cifar10 import MyTestDataset, MyTrainDataset, get_cifar_2
# from dataLoader.coco.coco import COCO2014
from dataLoader.data_list import ImageList
from dataLoader.flickr25k import Flickr25k
from utils.gaussian_blur import GaussianBlur



def getDataLoader(option):
    '''
    :param option:
    :return: train_loader,test_loader
    '''
    color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
    train_transforms = transforms.Compose([transforms.RandomResizedCrop(size=224, scale=(0.5, 1.0)),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.RandomApply([color_jitter], p=0.7),
                                           transforms.RandomGrayscale(p=0.2),
                                           GaussianBlur(3),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                           ])
    test_transforms = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.5, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_cifar10_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if option.data_name == 'cifar10':
        X_train, Y_train, X_val, Y_val, X_test, Y_test, X_database, Y_database = get_cifar_2(option.data_path)

        train_dataset = MyTrainDataset(X_train, Y_train, train_transforms, test_cifar10_transforms)
        test_dataset = MyTestDataset(X_val, Y_val, test_cifar10_transforms)
        database = MyTestDataset(X_database, Y_database, test_cifar10_transforms)
    elif option.data_name == "flickr25k":
        Flickr25k.init(option.data_path, 1000, 10000)
        train_dataset = Flickr25k(option.data_path, 'train', train_transform=train_transforms,
                                  test_transform=test_transforms, Train=True)
        test_dataset = Flickr25k(option.data_path, 'query', train_transform=train_transforms,
                                 test_transform=test_transforms)
        database = Flickr25k(option.data_path, 'retrieval', train_transform=train_transforms,
                             test_transform=test_transforms)
    else:
        database_list = '../data/' + option.data_name + '/database.txt'
        test_list = '../data/' + option.data_name + '/test.txt'
        train_list = '../data/' + option.data_name + '/train.txt'
        train_dataset = ImageList(option, open(train_list).readlines(),
                                  train_transform=train_transforms, test_transform=test_transforms, Train=True)
        database = ImageList(option, open(database_list).readlines(),
                             test_transform=test_transforms)

        test_dataset = ImageList(option, open(test_list).readlines(),
                                 test_transform=test_transforms)

    database_loader = torch.utils.data.DataLoader(database, batch_size=option.batch_size, shuffle=False,
                                                  num_workers=option.workers)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=option.batch_size, shuffle=False,
                                              num_workers=option.workers)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=option.batch_size,
                                               shuffle=True, num_workers=option.workers)

    return train_loader, test_loader, database_loader

    pass
