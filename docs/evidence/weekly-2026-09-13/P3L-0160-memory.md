# P3L-0160 memory

P159 已证明 softmax backward 上 75/75 请求可与 L-lite 精确哈希连接，189/189
个 Pipeline stage 2/3/4 候选最终 SASS 相同并可安全复用 L-lite 开发计时。189
条 D 记录中 86 条超过 1.0175×；stage 2/3/4 几何均值为 0.922×、1.129×、
1.220×。

P159-007 将五个 FlagGems 节点放进一个 3600 秒角色。运行 1710.18 秒时仍未完成
第一个 pytest 节点，CPU 保持约 80%，并持续出现 ptxas/cuobjdump，因此不是死锁；
但零个测试和零个 kernel 请求写入结果。用户要求以 900 秒为逐算子硬上限，慢
算子只记录、后续集中处理，不能阻塞可疑快速反馈节点。该批角色被安全 SIGTERM，
子进程组已清零，GPU 无遗留 compute PID。

P160 因执行协议变更而升版。科学模型、开发总体、标签来源、特征阶梯和验收标准
不变；仅将仓库级批角色改为逐算子原子角色，并按冻结请求数升序执行，任何超时
都继续下一项。

## 2026-09-07/08 — 逐算子结算与外层回退

- 六个角色全部获得终态账本：softmax forward、pack_seq 和 conv_transpose2d
  正常完成；FlagGems 与 FlagGems-vllm 的两个 MHC 角色分别在 900.56 秒和
  900.52 秒被进程组超时门终止，均为零完整请求。
- 最后的 chunk_gated_delta 角色在外层控制会话消失后成为孤儿；其 pytest 最终
  写出一个包含 82 请求的完整 observation，但完成时间相对上一原子角色至少晚
  2256.06 秒，且没有原子角色结果和环境后置证据。P160-002a 将 82 请求全部判为
  不可用于科学证据，并把后续生命周期 owner 定为父进程死亡联动加独立外层
  watchdog。
- P160-003 对三个正常完成角色与冻结 L-lite 原始计时作精确请求连接。21 个物质
  收益 Pipeline winner 中，7 个已有 Main-L 特征，9 个因 Cut-B 显式排除
  `bridge_factor > 1` 的 Bridge×route 组合而缺失，5 个因 Cut-B 只接受顶层
  Pipeline subject 而遗漏已由 P99 证明可执行的唯一嵌套 Pipeline subject。
- 这两项均是规则通用的 Cut-B 服务覆盖回归，不是算子、名称或 shape 特例。
  P99 已有唯一嵌套 Pipeline 的真实物化证据，P115 已有 O/B/C 三臂因果分解的
  方法证据。下一版必须从 `CutBRuleGenericPlanAndThreeArmProjection` 恢复这些
  能力，再重复受影响的快速角色；超时算子继续留待集中处理。
