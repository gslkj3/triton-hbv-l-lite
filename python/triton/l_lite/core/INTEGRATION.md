# Python 候选编译入口与原生流程接线

`lite_entry.autotune_kernel` 接受原生 `@triton.jit` 函数与实参，使用原生参数
绑定、统一候选准备和物化，再交给原生 Autotuner 编译、测量与选择。
本目录不包含预测选择器、回归系数或时间模型，也不依赖主L的历史执行器。

本分支已同步新版C++核心、原生传播支持和Python绑定，并独立重建通过
15项SM89 GPU端到端测试。H20接入尚未验证，两个库的批量测试适配尚待完成；
交接步骤见docs/H20-SOL-HANDOFF.zh-CN.md，不得借用旧extension冒充新版。

后续验收必须包括完整原生前后缀、Python 实参、不同 shape/factor、三路线、
正确性与原生 autotune 的候选实际执行；固定 TTIR 数值测试不能代替这些检查。

## 2026-09-24：物化属于编译流程，预测仅提供决策

原生 `CUDABackend.make_ttir` 保留流程编排，不在阶段开头跳到另一套编译器。
`TRITON_L_LITE_MODE=default` 的执行顺序为：

1. 原生函数内联、指针整理、表达式简化、公共表达式消除等前置 Pass。
2. 调用 `bind_default_decision`：在分析点提取事实，使用 Bridge factor=1
   和原生默认 stage 选择合法流水配置；没有合适循环则保留普通编译路径。
3. 项目决策 Pass → 原生 LoopUnroll → 项目物化与后置验证 Pass。
4. 返回原生编译器，继续 TTGIR、LLVM、PTX 与二进制生成。

`prepare_at_analysis_point` 不重复执行原生前置 Pass；`prepare_module` 是
供显式候选准备使用的完整前缀适配器。前者接收整理后的 TTIR，后者接收刚由
前端生成的 TTIR，不能混用。软件流水的真正调度仍在后续原生 TTGIR 阶段。

本轮15项正确性测试通过（4.29秒），不是性能收益证明，也不是任意算子覆盖证明。
主L的 `--range_predict` 外接分析器仍在接线；公共L-lite不包含预测模型，
不把该开关解释成 autotune。L-lite搜索入口仍是 `autotune_kernel`。
