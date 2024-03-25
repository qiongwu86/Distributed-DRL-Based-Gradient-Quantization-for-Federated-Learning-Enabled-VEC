import argparse # 引入模块

def args_parser():
    parser = argparse.ArgumentParser(
        description='Gradient Quantization Samples')
    parser.add_argument('--network', type=str, default='fcn')
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--num-classes', type=int, default=10)
    parser.add_argument('--quantizer', type=str, default='qsgd')
    parser.add_argument('--mode', type=str, default='ps')
    parser.add_argument('--scale', type=str, default="exp")

    parser.add_argument('--c-dim', type=int, default=128)
    parser.add_argument('--k-bit', type=int, default=8)
    parser.add_argument('--n-bit', type=int, default=2)
    parser.add_argument('--cr', type=int, default=256)
    parser.add_argument('--random', type=int, default=True)

    parser.add_argument('--num-users', type=int, default=4, metavar='N',
                        help='num of users for training (default: 8)')
    parser.add_argument('--logdir', type=str, default='logs_qsgd',
                        help='For Saving the logs')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=350, metavar='N',
                        help='number of epochs to train (default: 350)')

    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4, metavar='M',
                        help='weight decay momentum (default: 5e-4)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--ef', action='store_true', default=False,
                        help='enable error feedback')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-epoch', type=int, default=1, metavar='N',
                        help='logging training status at each epoch')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--two-phase', action='store_true', default=False,
                        help='For Compression two phases')
    args = parser.parse_args() # 属性给予args实例
    return args
