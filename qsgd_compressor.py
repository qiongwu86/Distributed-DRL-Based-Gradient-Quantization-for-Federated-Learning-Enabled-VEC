import torch


class QSGDCompressor(object):
    def __init__(self, size, shape, args, q_level):
        self.random = args.random
        self.bit = q_level # 量化电平(动作)
        c_dim = args.c_dim # 桶?
        assert self.bit > 0 # 量化电平大于零

        self.cuda = not args.no_cuda # False
        self.s = 2 ** self.bit
        self.size = size # 权重的尺寸
        self.shape = shape # 权重的形状

        if c_dim == 0 or self.size < args.c_dim:
            self.dim = self.size
        else:
            self.dim = c_dim # 桶大小
            for i in range(0, 10):
                if size % self.dim != 0:
                    self.dim = self.dim // 2 * 3

        if c_dim != self.dim:
            print("alternate dimension form"
                  " {} to {}, size {} shape {}"
                  .format(c_dim, self.dim, size, shape))

        assert self.dim != 0, \
            "0 sub dimension size {}  " \
            "c_dim {} self.dim {}"\
                .format(size, c_dim, self.dim)
        assert size % self.dim == 0, \
            "not divisible size {} " \
            " c_dim {} self.dim {}"\
                .format(size, c_dim, self.dim)

        self.M = size // self.dim
        self.code_dtype = torch.int32


    def compress(self, vec):
        """
        :param vec: torch tensor
        :return: norm, signs, quantized_intervals
        """
        vec = vec.view(-1, self.dim)
        # norm = torch.norm(vec, dim=1, keepdim=True)
        norm = torch.max(torch.abs(vec), dim=1, keepdim=True)[0]
        normalized_vec = vec / norm

        scaled_vec = torch.abs(normalized_vec) * self.s
        l = torch.clamp(scaled_vec, 0, self.s-1).type(self.code_dtype)

        if self.random:
            # l[i] <- l[i] + 1 with probability |v_i| / ||v|| * s - l
            probabilities = scaled_vec - l.type(torch.float32)
            r = torch.rand(l.size())
            if self.cuda:
                r = r.cuda()
            l[:] += (probabilities > r).type(self.code_dtype)

        signs = torch.sign(vec) > 0
        return [norm, signs.view(self.shape), l.view(self.shape)]

    def decompress(self, signature):
        [norm, signs, l] = signature
        assert l.shape == signs.shape
        scaled_vec = l.type(torch.float32) * (2 * signs.type(torch.float32) - 1)
        compressed = (scaled_vec.view((-1, self.dim))) * norm / self.s
        return compressed.view(self.shape)
