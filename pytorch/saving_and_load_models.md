# SAVING AND LOADING MODELS

## API

* **`torch.gather(input, dim, index, out=None) → Tensor`**

Gathers values along an axis specified by dim.

For a 3-D tensor the output is specified by:

```
out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2
```

If input is an n-dimensional tensor with size $(x_0, x_1..., x_{i-1}, x_i, x_{i+1}, ..., x_{n-1})$ and `dim = i`, then index must be an $n$-dimensional tensor with size $(x_0, x_1, ..., x_{i-1}, y, x_{i+1}, ..., x_{n-1})$ where $y \geq 1$ and out will have the same size as index.

**Parameters:**
* input (Tensor) – the source tensor
* dim (int) – the axis along which to index
* index (LongTensor) – the indices of elements to gather
* out (Tensor, optional) – the destination tensor


**Example:**

```bash
>>> t = torch.tensor([[1,2],[3,4]])
>>> torch.gather(t, 1, torch.tensor([[0,0],[1,0]]))
tensor([[ 1,  1],
        [ 4,  3]])
```

首先`torch.gather`返回一个和 index 具有相同尺寸的 tensor ，具体值为修改对应的 dim 的值为index中的值即可。例如上面例子就是修改 dim=1，首先 out 是一个2x2的tensor，让那后结果为[[ t[0][**0**], t[0][**0**] ], [ t[1][**1**], t[1][**0**] ]，其中粗体字为 index 替换原来的值。