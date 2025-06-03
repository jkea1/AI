import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np 

def dataloader(datatype='cifar10', batch_size=128, mode='train'):

    if datatype == 'cifar10':
        transform_train = transforms.Compose([
                                                transforms.RandomCrop(32, padding=4),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                            ])

        transform_test = transforms.Compose([
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                            ])
    else:
        print("in preparation")

    if mode == 'train':
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
   
    else:
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        dataloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    return dataloader