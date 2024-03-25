import math
import torch
from identical_compressor import IdenticalCompressor


class Quantizer():
    def __init__(self, Compressor, parameters, args, level):
        self.parameters = list(parameters) # 模型权重
        self.num_layers = len(self.parameters) # 网络层数
        self.compressors = list() # 初始化量化器
        self.compressed_gradients = [list() for _ in range(self.num_layers)] # 压缩后的梯度
        self.args = args # 参数
        self.error_feedback = args.ef # False
        self.two_phase = self.args.two_phase # False
        for param in self.parameters: # 对每层的权重
            param_size = param.flatten().shape[0] # 权重降维后的尺寸
            self.compressors.append(
                Compressor(param_size, param.shape, args, level) if param_size > 1000 # 添加量化器
                else IdenticalCompressor()
            )
            if self.error_feedback:
                param.error = [torch.zeros_like(param)
                               for _ in range(args.num_users)]
            if self.error_feedback and self.two_phase:
                param.server_error = torch.zeros_like(param)

    def  record(self, user, epoch):
        if self.args.scale == 'exp':
            scale = (2 / (math.exp(-epoch) + 1) - 1)
        else:
            scale = float(self.args.scale)

        for i, param in enumerate(self.parameters):
            if self.error_feedback:
                param.grad.data.add_(scale * param.error[user])
                decompressed_g = self.compressors[i].decompress(
                    self.compressors[i].compress(param.grad.data)
                )
                param.error[user].data = param.grad.data - decompressed_g
            else:
                decompressed_g = self.compressors[i].decompress(
                    self.compressors[i].compress(param.grad.data)
                )
            self.compressed_gradients[i].append(decompressed_g)

    def apply(self):
        for i, param in enumerate(self.parameters):
            g = torch.stack(self.compressed_gradients[i], dim=0).mean(dim=0)

            # if compress gradient on two phase, i.e.,
            # compress the sum of decompressed gradient
            if self.two_phase:
                if self.error_feedback:
                    g.add_(param.server_error)
                    decompressed_g = self.compressors[i].decompress(
                        self.compressors[i].compress(g))
                    param.server_error = g - decompressed_g
                    g = decompressed_g
                else:
                    g = self.compressors[i].decompress(
                        self.compressors[i].compress(g))

            param.grad.data = g
        for compressed in self.compressed_gradients:
            compressed.clear()
