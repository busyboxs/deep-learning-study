# Pytorch Basic

## 1. Inplace operation

任何带有 `_`(下划线)的操作都是执行的 in-place 操作，会改变原来 tensor 的值。例如: `x.copy_(y)`, `x.t_()`。

```python
# example
x = torch.ones(2, 4)
y = torch.ones(2, 4)
x.add_(y)
print(x)
print(y)
```

```python
# output
tensor([[2., 2., 2., 2.],
        [2., 2., 2., 2.]])
tensor([[1., 1., 1., 1.],
        [1., 1., 1., 1.]])

```

## Numpy and Tensor

Numpy array can convert to torch tensor and torch tensor can convert to numpy array.

* `tensor.numpy()`: convert tensor to numpy
* `torch.from_numpy()`: convert numpy to tensor, share memory

### 1. Converting a Torch Tensor to a NumPy Array

```python
# example
# 1. Converting a Torch Tensor to a NumPy Array
x_tensor = torch.ones(2, 4)
print(x_tensor)
x_np = x_tensor.numpy()
print(x_np)
```

```python
# output
tensor([[1., 1., 1., 1.],
        [1., 1., 1., 1.]])
[[1. 1. 1. 1.]
 [1. 1. 1. 1.]]
```

### 2. Converting NumPy Array to Torch Tensor

```python
# 2. Converting NumPy Array to Torch Tensor
import numpy as np
x_np = np.random.rand(2, 4)
print(x_np)
x_tensor = torch.from_numpy(x_np)
print(x_tensor)
# from_numpy make numpy array and tensor share memory
np.add(x_np, 1, out=x_np)
print(x_np)
print(x_tensor)
```

```python
# output
[[0.7850417  0.07124835 0.79930567 0.33572965]
 [0.66205956 0.19922021 0.54025427 0.40338249]]
tensor([[0.7850, 0.0712, 0.7993, 0.3357],
        [0.6621, 0.1992, 0.5403, 0.4034]], dtype=torch.float64)
[[1.7850417  1.07124835 1.79930567 1.33572965]
 [1.66205956 1.19922021 1.54025427 1.40338249]]
tensor([[1.7850, 1.0712, 1.7993, 1.3357],
        [1.6621, 1.1992, 1.5403, 1.4034]], dtype=torch.float64)
```

从输出结果的后两个可以看到，在改变 numpy array 后，通过 `from_numpy` 得到的 tensor 的数据也会发生改变，是因为通过 `from_numpy` 得到的 tensor 和 numpy array 是共享内存的。

