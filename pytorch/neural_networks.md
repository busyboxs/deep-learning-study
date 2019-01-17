# Neural Networks

![](https://pytorch.org/tutorials/_images/mnist.png)

## 神经网络的训练过程

* 定义拥有可学习参数的神经网络
* 在输入数据集上进行迭代
* 通过网络处理输入
* 计算损失
* 反向传播梯度到每一个参数
* 更新网络的权重（`weight = weight - learning_rate * gradient` ）

## 在定义网络时，只需要定义初始化函数和前向传播函数即可

* 初始化函数 `def init(self)`
* 前向传播函数 `def forward(self,...)`

```python
# example
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    # init function
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    # forward function    
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    # calculate dimension of flat
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
print(net)
```

```python
# output
Net(
  (conv1): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1))
  (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
  (fc1): Linear(in_features=400, out_features=120, bias=True)
  (fc2): Linear(in_features=120, out_features=84, bias=True)
  (fc3): Linear(in_features=84, out_features=10, bias=True)
)
```

从输出中可以看到全连接层 `Linear` 具有的属性包括 `in_feature`, `out_feature` 和 `bias`。`in_feature` 代表全连接层的输入维度，`out_feature` 代表全连接层的输出维度。这个属性比较重要，尤其是在迁移学习中，可以通过修改最后一层全连接层的输出维度来进行 finetune。

## 模型的参数

模型的可学习参数可以通过 `net.parameters()` 得到，得到的是一个 generator object。例如，获取上面模型的参数，并打印参数的 shape。

```python
# example
params = list(net.parameters())
print(len(params))
for p in params:
    print(p.shape)
```

```python
# output
10
torch.Size([6, 1, 5, 5])
torch.Size([6])
torch.Size([16, 6, 5, 5])
torch.Size([16])
torch.Size([120, 400])
torch.Size([120])
torch.Size([84, 120])
torch.Size([84])
torch.Size([10, 84])
torch.Size([10])
```

从输出结果中可以看到，可学习的参数有 10 个，分别为 3 个卷积层和 2 个全连接层的权重和偏执。

## 进行一次前向计算和反向传播

给定一个随机的输入数据，将该数据输入到网络中并得到输出。然后将梯度设置为0，并进行一次反向传播。

```python
# example
input_data = torch.randn(1, 1, 32, 32)
out = net(input_data)
print(out)
net.zero_grad()
out.backward(torch.randn(1, 10))
```

```python
# output
tensor([[-0.0610, -0.0345, -0.0832, -0.0523, -0.0140, -0.0822,  0.0919,  0.0462,
         -0.0568,  0.0270]], grad_fn=<AddmmBackward>)
```

> **NOTE**:`torch.nn` 只支持mini-batches。即输入输出均为 4D Tensor of (nSamples, nChannels, Height, width)。如果是单个 sample，可以使用 input_data.unsqueeze() 添加假的 batch 维度。

## 损失函数

损失函数将网络的输出和标签作为输入，然后计算它们之间的差距。`nn` 包下有许多类型的损失函数，下面代码展示的是一个 `nn.MSELoss` 用于计算均方误差。

```python
# example
output = net(input_data)
target = torch.randn(10)
target = target.view(1, -1)
print(output)
print(target)
criterion = nn.MSELoss()
loss = criterion(output, target)
print(loss)
```

```python
# output
tensor([[-0.0610, -0.0345, -0.0832, -0.0523, -0.0140, -0.0822,  0.0919,  0.0462,
         -0.0568,  0.0270]], grad_fn=<AddmmBackward>)
tensor([[ 1.7878, -0.5403,  0.2431, -0.9502, -0.3379, -0.0433, -1.4311,  1.1591,
         -0.1547, -2.3632]])
tensor(1.3974, grad_fn=<MseLossBackward>)
```

## 获取反向传播路径（待添加，这部分还不是很会）

```
input -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d
      -> view -> linear -> relu -> linear -> relu -> linear
      -> MSELoss
      -> loss
```

```python
# example
print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU
```

```python
#output
<MseLossBackward object at 0x000001924499C710>
<AddmmBackward object at 0x000001924499C7B8>
<AccumulateGrad object at 0x00000192445BC0F0>
```

## 执行反向传播

执行反向传播，我们只需要调用 `loss.backward()` 就可以。当我们调用 `loss.backward()` 时，整个图就会自动求导，图中的损失和 `requires_grad=True` 的 Tensor 的梯度累计会存放在对应的 `.grad`中。

在执行反向传播之前，需要清除已经存在的梯度，否则梯度会累积到已经存在的梯度中。

```python
# example
net.zero_grad()  # zeros the gradient buffers of all parameters
print('conv1.bias.grad before backword')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)
```

```python
# output
conv1.bias.grad before backword
tensor([0., 0., 0., 0., 0., 0.])
conv1.bias.grad after backward
tensor([-0.0085, -0.0076,  0.0146,  0.0054, -0.0078,  0.0056])
```

## 更新权重

梯度下降权重更新规则：

```
wright = weight - learning_rate * gradient
```

* 使用 python 来实现以上公式

```python
# example
learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)
```

* 使用 pytorch 提供的函数实现

```python
# example
import torch.optim as optim

# create optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

# train loop
optimizer.zero_grad()  # zeros gradient buffers
output = net(input_data)
loss = criterion(output, target)
loss.backward()
optimizer.step()  # Does the update
```

在训练循环中，需要零初始梯度，反向传播，优化。

