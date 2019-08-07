from torch import nn

#################
# 为了支持onnx
################
class MyAdaptiveAvgPool2d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()

    def forward(self, x):
        inp_size = x.size()
        return nn.functional.avg_pool2d(input=x,kernel_size=(inp_size[2], inp_size[3]), ceil_mode=False)




class SELayer(nn.Module):
    def __init__(self, channel, reduction=1):
        super(SELayer, self).__init__()
        
        #  self.avg_pool = nn.AdaptiveAvgPool2d(1)
        #  import ipdb;ipdb.set_trace()
        self.avg_pool = MyAdaptiveAvgPool2d()
        
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
