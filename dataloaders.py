import torch
from torchvision import datasets, transforms


def mnist(args):
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size*args.num_users, shuffle=True,)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.test_batch_size, shuffle=True,)
    return train_loader, test_loader


def cifar10(args):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
        transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),  # R,G,B每层的归一化用到的均值和方差
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    trainset = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)  # 训练数据集
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size*args.num_users, shuffle=True, num_workers=2)  # 生成一个个batch进行批训练，组成batch的时候顺序打乱取

    testset = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        testset, args.test_batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader


def cifar100(args):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
        transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),  # R,G,B每层的归一化用到的均值和方差
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    trainset = datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform_train)  # 训练数据集
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size*args.num_users, shuffle=True, num_workers=2)  # 生成一个个batch进行批训练，组成batch的时候顺序打乱取

    testset = datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        testset, args.test_batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader


def stl10(args):
    transform_train = transforms.Compose([
        transforms.RandomCrop(96, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
        transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),  # R,G,B每层的归一化用到的均值和方差
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    trainset = datasets.STL10(
        root='./data', split='train', download=True, transform=transform_train)  # 训练数据集
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size*args.num_users, shuffle=True, num_workers=2)  # 生成一个个batch进行批训练，组成batch的时候顺序打乱取

    testset = datasets.STL10(
        root='./data', split='test', download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        testset, args.test_batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader



def svhn(args):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
        transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),  # R,G,B每层的归一化用到的均值和方差
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    trainset = datasets.SVHN(
        root='./data', split='train', download=True, transform=transform_train)  # 训练数据集
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size*args.num_users, shuffle=True, num_workers=2)  # 生成一个个batch进行批训练，组成batch的时候顺序打乱取

    testset = datasets.SVHN(
        root='./data', split='test', download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        testset, args.test_batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader


def tinyimgnet(args):

    traindir = './tinyimgnet/train'
    testdir = './tinyimgnet/val'
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    )
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size*args.num_users, shuffle=(train_sampler is None),
        num_workers=8, pin_memory=True, sampler=train_sampler)

    test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(testdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.test_batch_size, shuffle=False,
        num_workers=8, pin_memory=True)

    return train_loader, test_loader

