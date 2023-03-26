import torch.nn.functional as F
from torch import nn
import torch
from torchinfo import summary

# In[]
class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=3)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        self.fc = nn.Linear(2880, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(in_size, -1)
        x = self.fc(x)
        for batch in range(in_size):
            for index in range(x.size(1)):
                if x[batch, index] >= 0:
                    x[batch, index] = 1
                else:
                    x[batch, index] = 0
        x = self.fc2(x)
        return x


ts_net = net()
data = torch.zeros([1, 3, 16, 16])
out = ts_net(data)

print('1')
summary(model=ts_net, input_size=(1, 3, 16, 16))
print('2')
torch.onnx.export(ts_net,
                  data,
                  f='test.onnx')