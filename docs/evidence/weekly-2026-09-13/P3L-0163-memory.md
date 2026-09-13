# P3L-0163 memory

P162 完成了七模板 O/B/C、单一共享 Pipeline head 和运行时零计时选择，但独立 Q 仅在
Conj 选择一个不同 CUBIN：点预测约 `1.05213x`，两个相同 flat extent 的真实点为
`0.999724x` 和 `1.001999x`。经验单侧半径约 `1.05244x`，会消除所有 Q 物质候选；
因此没有打开 sealed E，也没有用 Q 标签重拟合。

P162-015 定位到更早的总体选择偏差：冻结 L-lite manifest 中有 32 个角色包含有限
实测的 `Bridge>1 + Pipeline` 候选，而 P162 按“Pipeline 为全局 winner”只选了 7 个。
遗漏的 25 个角色包括大量同域负对照，也包括 5 个有物质 Pipeline 候选、但全局最佳
属于其他 route 的角色。P163 回到这个 owner，不改变因果模型或物化规则。

起始成功率：32 角色编译事实采集与精确 join 约 80%（主要风险是 900 秒超时和多 kernel
角色身份连接）；完整总体后 full-D 零 harmful 且非零物质恢复约 75%；grouped 外推
从零提升为非零约 55%--65%；新 Q/E 安全非零发布约 55%--65%；长期目标综合约
60%--65%。

P163-002 完成了全部 25 个补充角色：22 个完整通过，3 个以 typed test failure 结束，
没有超时。已完成角色中逐候选 B-PTX/C-SASS 身份检查保持精确；`softmax_backward`
约 600 秒、SVD 约 676 秒也在 900 秒上限内完成。三个失败不属于回归误差：

- `softmax` 与 `renorm` 的组合计划解析错误地把多个已有循环压成一个匿名运行时主体；
  最早 owner 是 route-specific Provider subject binding。
- `top_k_per_row_prefill` 的静态分区递归候选已经成功物化，但 Cut-B 特征层没有把
  `static_consecutive_virtual_program_partition_recurrence` 识别为 Bridge 构造主体，
  因而错误回查空的 Provider loop census。

两项都按通用规则修复，没有 operator/template 分支、性能标签或新模型参数。53 个
相关单测通过。修复 population 以 P163-002b 冻结，P163-002c 只重跑三个角色；完整
32 角色精确连接仍保留 P163-003 编号。当前物化闭环成功率提高为约 85%--90%，完整
目标仍约 60%--65%，主要不确定性已转移到完整总体 grouped 泛化和后续独立 Q/E。

P163-002c 的真实重跑结果进一步区分了功能与执行粒度：

- `top_k_per_row_prefill` 在通用修复后完整通过，8 个 request 精确覆盖；其中 4 个
  request 各有 12 个有限 Pipeline 候选，全部具有 C artifact 与 O/B/C feature，合计
  48/48。
- `renorm` 不再复现主体绑定异常，但完整节点达到 900 秒硬上限；这是采集角色过大，
  不是新的模型或物化反例。
- `softmax` 同样运行至 watchdog 边界，并被 SIGTERM 终止在原子输出提交之前；原始
  证据诚实保留为 `ROLE_OUTPUT_MISSING`，不能冒充 timeout-complete 或通过。

因此 P163-002e 冻结唯一的执行粒度修复：只沿两个 benchmark 原有的 dtype 迭代拆为
FP16/FP32/BF16 六个角色，每片仍限 900 秒，入口、shape、代码、candidate population
和冻结 L-lite 标签均不变。P163-002g 只有在每个原角色的 slice request 集不重叠且并集
严格等于冻结 L-lite request 集时才允许合并。该拆分不是回归分域，也不增加模型 head。
