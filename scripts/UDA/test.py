# %%
import torch

def torch_device_one():
    return torch.tensor(1.).to(_get_device())

def _get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

device_one = torch_device_one()
print(device_one)

# %%
a = torch.ones((8, 10))
b = torch.ones((8, 1))
print(a)
print(b)
print('a * b', a * b)
print('sum b', torch.sum(b))
x = torch.sum(a * b, dim=-1) / torch.max(torch.sum(b, dim=-1), torch_device_one())
print('x', x)
print('sum a * b', torch.sum(a * b, dim=-1))

# %%
a = {
    'test': [[1, 2]]
}
b = {
    'test': [[2, 3]]
}

a['test'].extend(b['test'])

# %%
'''Temporary configuration'''
from argparse import Namespace

cfg = Namespace(
    tsa = False,
)

print(cfg.tsa)

# %%
import torch
import torch.nn as nn
from torch.nn import Linear
import random
from torch.autograd import Variable

# %%
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear = Linear(2, 3)
    
    def forward(self, x):
        z = self.linear(x)
        return z

# %%
train_x = Variable(torch.tensor([[1.0, 1.0], [1.0, 0.3], [0.8, 0.8], [5.0, 5.0], [6.0, 5.6]]))
train_y = Variable(torch.tensor([0, 0, 0, 1, 0]))

# %%
x_input = torch.randn(3, requires_grad=True)
print(x_input)
x_target = torch.empty(3).random_(2)
print(x_target)

# %%
model = Net()

# %%
criterion = nn.CrossEntropyLoss(reduction='none')
optim = torch.optim.SGD(model.parameters(), lr=0.1)

# %%
print('Size:', train_y.shape)
pred_y = model(train_x)
print('Y pred:', pred_y)
print('Y pred shape:', pred_y.shape)
print(torch.mean(criterion(pred_y, train_y)))

# %%
# for epoch in range(1000):
#     pred_y = model(train_x)
#     print('Pred y:', pred_y)

#     loss = criterion(pred_y, train_y)

#     optim.zero_grad()
#     loss.backward()
#     optim.step()

# %%
# y_pred = model(train_x)
# print(nn.Softmax()(y_pred))

# %%
# class LSTMNet(nn.Module):
#     def __init__(input_size, hidden_size, num_classes):
#         self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
# rnn = nn.LSTM(5, 5, 2) # (input_size, hidden_size, num_layer)
# input = torch.randn(5, 1, 5) # (seq_len, batch_size, input_size)
# h0 = torch.randn(2, 1, 5) # (num_layer * num_direction, batch_size, hidden_size)
# c0 = torch.randn(2, 1, 5) # (num_layer * num_direction, batch_size, hidden_size)

# print(rnn(input))

# %%
