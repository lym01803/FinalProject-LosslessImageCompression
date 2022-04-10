import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import random

class InvertibleModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def build(self):
        pass

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def backward(self, *args, **kwargs):
        raise NotImplementedError

    def inverse(self, *args, **kwargs):
        pass


class Permute(InvertibleModule):
    def __init__(self, dim):
        '''
        params:
            dim: dimension, input_tensor.shape[-1]
        '''
        super().__init__()
        p = torch.zeros((dim, dim))
        ids = [i for i in range(dim)]
        random.shuffle(ids)
        p[torch.tensor([i for i in range(dim)]), torch.tensor(ids)] = 1
        self.P = Parameter(p, requires_grad=False)
        self.inv_P = Parameter(p.t(), requires_grad=False)
    
    def forward(self, x, logv):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = F.linear(x, self.P)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x, logv

    def backward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = F.linear(x, self.inv_P)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class InvertibleModuleList(InvertibleModule, nn.ModuleList):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def inverse(self):
        for i in range(self.__len__()):
            m = self.__getitem__(i)
            if isinstance(m, InvertibleModule):
                m.inverse()
    
    def forward(self, x, logv, *args, **kwargs):
        for i in range(self.__len__()):
            m = self.__getitem__(i)
            x, logv = m.forward(x, logv, *args, **kwargs)
        return x, logv

    def backward(self, x, *args, **kwargs):
        for i in range(self.__len__()):
            m = self.__getitem__(self.__len__() - 1 - i)
            x = m.backward(x, *args, **kwargs)
        return x
