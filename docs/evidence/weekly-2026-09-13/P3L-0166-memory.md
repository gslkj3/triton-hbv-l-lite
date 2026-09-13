# P3L-0166 Memory

## 2026-09-08 起点

- P165 请求内正式 Q 完成 16 对 rank-1 测量，但没有一对达到 1.0175x，因此 E 未开封。
- 同入口 L-lite `mean` 完整 oracle 在 876 个候选中找到两个物质请求：f16
  `[Bridge=2, Pipeline, factor=2]` 为 1.113043x，另一请求为 phase-major 1.019230x；
  `prod` 没有物质请求。
- f16 Pipeline 机会不是 Strong、Pass 或物化失败。P164 authority 仅因
  `log1p_output_bytes_per_logical_program=log(3)` 小于 D 下界 `log(5)` 而拒绝；因此
  最早 owner 是规则式开发支持覆盖。
- P164 被选中的 T1 模型共有 26 个坐标：2 个 intervention dose、16 个原始
  Provider/launch 数值事实、8 个 body-type 指示量。它没有消费 O→B、B→C PTX、
  allocator、候选时间或身份字段。
- 这证明排序在科学上可以前移到 Candidate C 编译之前。当前 Cut-B 仍先编译所有
  plans/Bridge probes 再评分，是执行接线位置造成的编译开销，不是模型需要。
- S 快速论文基线/H20/S5000 审计只确认可迁移的方法论：预编译状态和独立后编译事实
  分离、明确 support/refusal、角色不可逆；没有迁移任何 S 数值模型。

## 当前边界

- P164/P165 Q 标签不可用于 P166 拟合。
- P164-010 的旧 E 保持 sealed。
- P165 L-lite oracle 是已公开 development oracle，可用于恢复目标与对照，不是 P166
  独立验证。
- 当前尚未宣称 Candidate 编译下降或总墙钟低于 L-lite。

## 2026-09-08 预编译排序闭环

- `p166-009-precompile-mean-smoke-evidence-v1.json` 已证明在公开 development
  `mean` 入口上，pre-C 排序与旧 full-C 排序选择同一坐标；该证据只验证执行接线，
  不承担独立性能发布角色。
- `p166-011-old-authority-precompile-mean-smoke-v2.json` 的可审计计数为：旧路径编译
  540 个 Candidate C 和 60 个 Bridge B，pre-C 路径只编译 9 个 C 和 9 个 B；
  Candidate C 数下降 98.333%，B 数下降 85%，记录的纯编译 acquisition 从
  426.703 秒降到 10.174 秒。环境有噪声，因此 506.537→36.609 秒墙钟仅作
  development 执行证据，不作性能发布声明。
- pre-C 特征可导出审计 `p166-005-pre-C-derivability-audit-v1.json` 覆盖 2016 行、
  201 个请求，0 个字段违规。26 个坐标均来自 Original/Provider、既定 launch 与
  候选干预 dose；Candidate C、候选 PTX、候选计时和身份字段均不参与排序。

## 2026-09-08 完整头部开发模型与运行时权威

- 完整头部 D `p166-010-augmented-typed-pipeline-D-v1.json` 含 2172 行、214 个物理
  请求、36 个模板组，并显式纳入公开 L-lite `mean` 头部。
- 最小参数选择 `p166-012-augmented-minimum-parameter-model-v1.json` 按预注册顺序先
  比较 leave-template-out MAE，在 MAE+0.005 的等精度集合里选择拟合参数更少者，
  最后才观察开发效用；结果为单个 T1 GradientBoosting-Huber，26 个输入坐标、224
  个拟合参数，LOTO MAE 0.173154。它恢复 31 个开发物质请求中的 19 个，并恢复
  `mean` f16 的 Bridge=2/Pipeline factor=2，实测 1.113043x。
- 自然 reduction census `p166-014-natural-reduction-census-evidence-v2.json` 闭环：
  624 个 kernel 请求、16696 个有限 Pipeline plans；旧 authority 只编译/计时 228 个
  rank-1 请求，396 个请求类型化拒绝。原始行中 5 个达到 1.0175x；合并物理重复并
  只保留外围正确性通过的观察后，120 个可用 partial-action 行中 1 个物理物质请求
  (`_std_map_kernel`, 1.0572x)。6 对处于最终 OOM 失败测试内，不能作为性能标签。
- 首次 census 和 partial-D 构建器分别暴露了分母检查错误，失败件
  `p166-013-*`、`p166-015-*` 保持不可变；只修正验证器的 v2 件为 `p166-014-*`、
  `p166-016-*`，未重写采集数据。
- 将 120 个由旧 ranker 选择的 partial actions 直接并入 action ranker 会造成
  missing-not-at-random：它们没有同请求完整反事实头部。`p166-017-*` 虽把全局
  LOTO MAE 降到 0.166786，却把物质恢复从 19/31 降到 16/31 并丢失 `mean`；
  `p166-018-partial-action-outer-diagnosis-v1.json` 因而把最早 owner 定位为
  `DevelopmentActionAssignmentIdentifiability`。partial actions 只能用于性能、校准、
  离散度证据或挑选下一批 full-head D，不能冒充可识别的候选排序训练样本。
- `p166-019-precompile-runtime-authority-v1.json` 冻结完整头部 D 上的 26 坐标单模型，
  保持 runtime 既有 schema 以复用稳定解析接口，但 generation 明确为 P3L-0166、
  状态明确为 development-only。`p166-021-runtime-precompile-equivalence-v2.json`
  证明 2172/2172 个分数与同一 full-D sklearn fit 数值一致（最大绝对差
  4.44e-16），214/214 个 rank-1 完全一致；同时通过数值 support、类别 support、
  target mismatch 和 machine-noop 的类型化拒绝测试。失败的 `p166-020-*` 是把
  full-D deployment fit 错与 LOTO out-of-fold 预测比较，保留为 comparator-role
  错误账本。

## 2026-09-08 独立 Q/E 冻结与执行

- `p166-023-independent-qe-protocol-v2.json` 在任何本轮计时前冻结 24 个此前未被
  Main-L 使用的真实 FlagGems 入口：reduction、scan、normalization、
  elementwise-extrema 各 6 个，SHA 排序后交替分到 12 Q / 12 sealed E，每入口 3 次。
  v1 中 `index_reduce_amax` 被发现已有历史 Main-L 提及，故在未计时前以同层的
  `scatter_reduce_two_amax` 替换；失败件 `p166-022-*` 保持不可变。
- Q 的预注册统计单位是物理请求，而不是 pytest 调用行：先在每次入口执行内合并重复
  调用，再对三次执行的 request-local log speedup 取中位数；请求离散度为各次值到
  中位数的最大绝对偏差，冻结半径为 pair-complete Q 请求离散度的最大值。物质收益
  下界必须达到 `log(1.0175)`；Q 标签禁止回写 ranker/support。Q 未通过则 E 永不打开。
- `p166-025-qe-library-execution-protocol-v2.json` 绑定 19 个源文件的内容 hash；v1
  只因 added/inherited 多节点文件重复计数而失败，v2 只修验证器并保持角色及源文件
  不变。
- `run_p166_q_campaign_v1.py` 以单 writer、单原子角色、最多 4 个编译进程、每角色
  900 秒上限在 GPU0 顺序执行 Q；每个角色记录前后 GPU telemetry 与 compute PID。
  它只绑定并读取 `p166-019` authority，状态固定 `E_started=false`，不会接触 E。
