# Pytorch autograd

## Tensor

* 通过设置 tensor 的属性 `.requires_grad` 为 `True`，让 Tensor 自动进行微分计算。

```python
# example
x_tensor = torch.ones(2, 4, requires_grad=True)
print(x_tensor)
```

```python
# output
tensor([[1., 1., 1., 1.],
        [1., 1., 1., 1.]], requires_grad=True)
```

* `.requires_grad_(...)` 以 in-place 的方式 改变已有 tensor 的 `requires_grad` 属性，其默认输入为 `False`。

```python
# example
x_tensor = torch.randn(2, 4)
print(x_tensor)
print(x_tensor.requires_grad)

x_tensor.requires_grad_(True)
print(x_tensor)
print(x_tensor.requires_grad)
```

```python
# output
tensor([[ 0.9718, -0.4612, -0.0507, -0.0315],
        [ 0.4063,  0.2852, -0.2386,  0.7216]])
False
tensor([[ 0.9718, -0.4612, -0.0507, -0.0315],
        [ 0.4063,  0.2852, -0.2386,  0.7216]], requires_grad=True)
True
```

* 如果不想让某个 Tensor 自动计算梯度，可以使用 `.detach()`。

```python
# example
# first define a tensor with requires_grad=True
x_tensor = torch.ones(2, 4, requires_grad=True)
print(x_tensor)
# use detach to stop a tensor from tracking history
x_tensor = x_tensor.detach()
print(x_tensor)
print(x_tensor.requires_grad)
```

* 也可以使用 `with torch.no_grad():` 代码块来阻止梯度计算，这在测试时非常有用。

```python
# example
x_tensor = torch.rand(2, 4, requires_grad=True)
print(x_tensor)
with torch.no_grad():
    x_d_tensor = x_tensor * 2
    print(x_d_tensor)
    print(x_d_tensor.requires_grad)
```

```python
# output
tensor([[0.8591, 0.4493, 0.6926, 0.0125],
        [0.2930, 0.1093, 0.4500, 0.7154]], requires_grad=True)
tensor([[1.7182, 0.8985, 1.3851, 0.0251],
        [0.5861, 0.2187, 0.8999, 1.4308]])
False
```

* Tensor 进过一些操作或者运算后，`grad_fn` 属性会被赋上对应的值。

```python
# example
# define a tensor with requires_grad=True
x = torch.ones(2, 2, requires_grad=True)
# add operate
y = x + 2
# multiply operate
z = y * y * 3
print(x)
print(y)
print(z)
```

```python
#ouput
tensor([[1., 1.],
        [1., 1.]], requires_grad=True)
tensor([[3., 3.],
        [3., 3.]], grad_fn=<AddBackward>)
tensor([[27., 27.],
        [27., 27.]], grad_fn=<MulBackward>)
```

* 在某个 Tensor 上调用`.backward()` 来计算梯度（导数）。

```python
# example
x = torch.ones(2, 2, requires_grad=True)
y = x + 2
z = y * y * 3
out = z.mean()
out.backward()
# Print gradients d(out)/dx
print(x.grad)
```

```python
# output
tensor([[4.5000, 4.5000],
        [4.5000, 4.5000]])
```

根据链式法则便可以得到上面的结果。因为 $out=\frac{1}{4}\sum_{i}z_i$, $z_i = 3(x_i+2)^2$, $z_i|_{x_i=1}=27$, 则有 $\frac{\partial o}{\partial x_i}=\frac{3}{2}(x_i+2)$，则有 $\frac{\partial o}{\partial x_i}|_{x_i=1}=\frac{9}{2}=4.5$。

