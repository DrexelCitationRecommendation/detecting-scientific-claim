# %%
import torch
import torch.nn as nn

a = torch.Tensor([1.3, 1.8, 2.2])
b = nn.Softmax()
print('Before temperature control', b(a))
c = a / 0.85
print('After temperature control ', b(c))

# %%
import torch

loss = 0.0
for i in range(3):
    loss += torch.Tensor([1.0])
print(loss/3)

# %%
import torch
import torch.nn as nn

# criterion = nn.CrossEntropyLoss(reduction='none', weight=torch.tensor([1.0, 1.0]))

# Loss function
def multiple_target_CrossEntropyLoss(logits, labels):
    output_loss = []
    for i in range(logits.shape[0]): # batch_size
        smaller_loss = nn.CrossEntropyLoss(weight=torch.tensor([1.0,1.0]), reduction='none')(logits[i, :, :], labels[i, :])
        smaller_loss = smaller_loss.unsqueeze(0)
        output_loss.append(smaller_loss)
        # loss = loss + nn.CrossEntropyLoss(weight=torch.tensor([1.0,1.0]).cuda())(logits[i, :, :], labels[i, :])
    return torch.cat(output_loss, dim=0)

# logits = torch.zeros((3, 2, 2), dtype=torch.float32)
logits = torch.Tensor([[[0.3, 0.7], [0.3, 0.7]], [[0.3, 0.7], [0.3, 0.7]], [[0.7, 0.3], [0.7, 0.3]]]).type(torch.float32)
labels = torch.zeros((3, 2), dtype=torch.long)
res = multiple_target_CrossEntropyLoss(logits, labels)
print(res.shape)

# %%
for i in range(3):
    pass
print(i)

# %%
import torch
import torch.nn as nn

criterion = nn.KLDivLoss(reduction='none')
a = torch.tensor([0.4724, 0.9769])
b = torch.tensor([0.4974, 0.9368])
print(criterion(a, b))

a = torch.tensor([0.4724, 0.9769])
b = torch.tensor([0.4974, 0.9368])
print(criterion(a, b))

# %%
a = [1, 2, 3]
print(*a)

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
        self.linear = Linear(2, 2)
    
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
print(criterion(torch.tensor([[0.9469, 0.5334]]), torch.tensor([0])))
print(criterion(pred_y, train_y))

# %%
for epoch in range(3):
    pred_y = model(train_x)
    print('Pred y:', pred_y)

    loss = criterion(pred_y, train_y)
    print('Loss: ', loss)
    loss = torch.mean(loss)
    print('Mean Loss: ', loss)

    optim.zero_grad()
    loss.backward()
    optim.step()

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
