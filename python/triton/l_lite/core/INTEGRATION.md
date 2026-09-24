# Python 完整候选编译入口（同步中，尚未发布验收）

`lite_entry.autotune_kernel` 接受原生 `@triton.jit` 函数与实参，使用原生参数
绑定、统一候选准备和物化，再交给原生 Autotuner 编译、测量与选择。
本目录不包含预测选择器、回归系数或时间模型，也不依赖主L的历史执行器。

本分支已同步新版C++核心、原生传播支持和Python绑定，并独立重建通过
12项SM89 GPU端到端测试。H20接入尚未验证，两个库的批量测试适配尚待完成；
交接步骤见docs/H20-SOL-HANDOFF.zh-CN.md，不得借用旧extension冒充新版。

后续验收必须包括完整原生前后缀、Python 实参、不同 shape/factor、三路线、
正确性与原生 autotune 的候选实际执行；固定 TTIR 数值测试不能代替这些检查。
