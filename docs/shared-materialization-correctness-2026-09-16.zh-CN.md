# 三项共性物化正确性修复

这次修改不改变候选搜索策略，也不增加预测模型。修复均根据代码依赖或类型规则
处理，不按算子名称分支。

1. **重排时补齐嵌套区域的依赖。** 内层循环可能引用外层定义的值。移动内层循环
   时，调度器必须查看循环内部的操作，不能只看循环操作自己的参数列表。遇到
   不属于当前变换的有副作用操作仍然拒绝跨越，不放松等价性要求。
2. **向量打包时清理旧维度属性。** 多个一维张量打包后可能变成二维或更高维。
   原来的连续性、整除性、常量区间属性描述的是旧维度，不能原样复制；由后续
   分析重新推导，而不是猜一个新值。
3. **整数分组归约保留已有累加值。** 本组贡献的和不等于整个循环的累加值。
   先证明完整加法依赖链，再生成“进入本组时的值＋本组贡献之和”。

实现位于 `lib/Dialect/Triton/Transforms/HBVLoop.cpp`。
对应直接测试分别为 `test/Triton/hbv-loop-bridge-nested-capture.mlir`、
`hbv-loop-packed-axis-hints.mlir` 和 `hbv-loop-integer-carry-materialize.mlir`。
三份测试中的计划是固定序列化输入，不依赖预测器。

## 验证范围

本仓库自身构建的编译器上，16项HBV编译测试、5项候选控制测试通过。
`python/test/unit/l_lite/test_materialization_numeric.py` 的3项GPU测试也通过：
比较实际输出，检查输入未改动和输出保护区未越界。这是RTX 4090上的定向
正确性回归，不是性能结果，也不代表所有类型、循环或架构均已验证。

构建使用 `cmake --build build/l-lite-public --target triton triton-opt --parallel 2`。
GPU测试需要显式设置 `L_LITE_RUN_NUMERIC=1`，并使用本仓库Python包和二进制；
测试会检查两者路径。直接CMake构建若未安装CUDA工具，可设置
`TRITON_PTXAS_PATH` 指向本机ptxas，并设置 `C_INCLUDE_PATH` 指向CUDA头文件目录。
不能用另一个Triton工作树的编译器代替本仓库验证。

本次不宣称已经提供从普通JIT函数到全部候选的完整公开编译入口，也不改变
架构设计书中对这一产品接口缺口的说明。
