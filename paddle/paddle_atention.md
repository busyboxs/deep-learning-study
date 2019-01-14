# 使用 paddlepaddle Fluid 时的一些注意事项

## 1. Executor 的 fetch_list

* 被 fetch 的 Variable 必须是 persistable(持久化) 的
*  `fetch_list` 可以传入 Variable 的列表，也可以传入 Variable 的 name 列表

```python
# example
result = exe.run(..., fetch_list=[loss])  # Variable list
result = exe.run(..., fetch_list=[loss.name])  # Variable name list
```

## 2. 多 GPU 上的参数初始化

* 在单卡上初始化直接执行 `exe.run(program=fluid.default_startup_program())`
* 使用多 GPU 训练，参数需要先在 GPU0 上初始化，再经由`fluid.ParallelExecutor` 分发到多张显卡上

## 3. 模型训练与测试之间的区别

* 模型测试时不进行反向传播，不优化更新参数
* 有些 op 在训练阶段和测试阶段是不同的，例如：BatchNorm，dropout
* 通过 `Program.clone(for_test=True)` 复制得到用于测试的 Program

## 4. 模型变量

* 模型变量包括模型参数(`fluid.framework.Parameter`)、长期变量(`fluid.Variable(persistable=True)`)、临时变量
* 所有的模型参数都是长期变量，但并非所有的长期变量都是模型参数

## 5. 载入模型变量

* 载入模型时的 prog 必须和调用 `fluid.io.save_params` 时所用的 prog 中的前向部分完全一致，且不能包含任何参数更新的操作
* 运行 `fluid.default_startup_program()` 必须在调用 `fluid.io.load_params` 之前

## 6. 