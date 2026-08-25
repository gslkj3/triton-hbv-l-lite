# L-lite 项目架构设计书：原生 Triton 对照、设计与具体实现

> 评审版本：2026-08-25
> L-lite 公开提交：`25bca1f4ec7404878ba4b52e19d4a5919a1e41ce`
> 原生 Triton 基线：`7c56a5e40f7fd928dfd5c72902d5def0097db73a`（Triton 3.6.0）
> 本文范围：提交 `25bca1f4e` 相对上述原生基线的全部 27 个文件级改动；本文是其后
> 新增的评审交付物，不计入该 27 文件实现快照
> 不包含：L 主项目的预测器源码、拟合参数和私有实验账本；本文第二部分补充其架构、
> 算法责任与当前完成状态，但不把私有实现复制进 L-lite

## 0. 评审摘要

### 0.1 一句话定义

L-lite 是一个建立在 Triton 3.6.0 上的循环联合优化实验对照：它把 **Loop
Bridge** 与一条 **route** 作为两个顺序独立的编译干预，枚举
`bridge_factor × route × route_factor`，对合法候选执行真实 TTIR 物化与后置
验证，再把已经构建好的候选交给 Triton 原生 autotune 实测选择。

### 0.2 相对原生 Triton，L-lite 增加了什么

原生 Triton 已有通用 `scf.for` 展开、GPU 软件流水和 autotune，但没有 L-lite
这套“发现对象—证明依赖—显式选择 route—保留两次干预 lineage—验证目标确实
物化”的控制面。L-lite 增加了：

1. 将 program-id 维度上的重复工作构造成显式循环的 Loop Bridge；
2. 三条类型化 route：原生软件流水、完全展开+重排、完全展开+向量化；
3. Bridge factor 与 route factor 的独立语义本体和组合合法性；
4. 六个 TTIR Pass，用于发现、事实提取、决策闭合、Bridge 物化、route 物化和
   最终后置验证；
5. 对原生 `LoopUnroll` 的最小扩展，用来保存 factor main/tail 与操作 lineage；
6. Python 侧候选图、不可变 identity、物化 attestation 和原生 autotune 适配层；
7. 13 个公开 MLIR 编译器测试、4 个 Python 控制层测试及两份已有说明文档。

全部变更统计为 27 个文件、15,490 行新增。原生后端软件流水实现、原生 benchmark
算法和 winner 选择算法没有被 L-lite 重写。

### 0.3 评审时必须区分的三个结论

| 结论 | 当前状态 | 含义 |
|---|---|---|
| 编译机制不是手写 kernel 特例 | 已实现 | Bridge 和 route 是注册到 Triton 的真实 C++/MLIR Pass；准入不读取 kernel、算子或 benchmark 名称 |
| 公开仓具备编译器能力面 | 已实现但公开测试覆盖不完整 | 物化器、Plan parser 和 validator 均在源码中；公开测试主要覆盖 Bridge/事实层 |
| 公开仓开箱即用地端到端 autotune | **尚未闭合** | 公开代码没有候选 PlanBundle/Kernel binding 工厂，也没有把六个 Pass 插入 Triton 默认 pipeline；调用者目前必须提供预构建 bindings |

第三项不是 autotune 算法问题，而是公开控制层与编译器层之间缺少接线。评审不能把
“Pass 已注册”误读成“默认编译流程会自动执行这些 Pass”。

## 1. 原生 Triton 基线

### 1.1 原生已有能力

原生 Triton 基线已经具备：

- `tt.loop_unroll_factor` 驱动的通用 `scf.for` 展开；
- TritonGPU 后端的软件流水分析、调度和 lowering；
- `triton.runtime.autotuner.Autotuner` 的配置遍历、实测、winner 选择和 key cache；
- 常规 TTIR/TTGIR 优化与 NVIDIA/AMD codegen。

原生 `LoopUnroll` 只读取 factor、调用 MLIR 的 `loopUnrollByFactor`，并把 epilogue
的 `tt.num_stages` 设为 1；它不保存 L-lite 所需的 route subject、main/tail
归属或操作组 lineage。L-lite 对这一点的差异见
[`LoopUnroll.cpp`](../lib/Dialect/Triton/Transforms/LoopUnroll.cpp#L24-L195)。

### 1.2 原生能力为何不足以表达本项目

本项目需要回答的不是“某个局部 pass 能否运行”，而是：

```text
同一个原始 kernel
  → 是否先用 Bridge 改变物理 program/显式循环结构
  → 在 Bridge 后的确切 subject 上选择哪条 route 和 factor
  → 两次干预是否都真正发生
  → 最终候选是否仍与原程序等价
  → 穷举实测哪个最终 kernel 最快
```

原生各 pass 分别工作，不提供上述跨 pass 的候选 identity、顺序组合证书和最终
postcondition。因此 L-lite 增加的是一层 **编译干预控制与证明架构**，不是替换
Triton 的全部编译器或 GPU 后端。

## 2. 目标、非目标与不可变原则

### 2.1 目标

- 对每个可观察循环对象，以规则而非 workload 名称判断三条 route 的能力；
- 把没有原生 `scf.for`、但 program 之间独立的工作通过 Bridge 暴露成循环；
- 保持 Bridge 与 route 两个 pass 的独立归因，同时对最终组合结果联合选优；
- 确认候选真的改变了目标 IR，而不是 pass 返回成功后回退为 Original；
- 为主 L 项目提供同一套合法性/物化能力的 exhaustive-autotune 对照。

### 2.2 非目标

- 不包含主 L 的 HBV 性能预测、剪枝、残差、误差半径或发布门；
- 不建立新的 autotune winner 算法；
- 不完全白盒复刻 NVIDIA/AMD 后端；
- 不保证每个观察到的循环都适合三条 route；不能证明时应类型化拒绝；
- 不以测试集、kernel 名、算子名或历史性能作为合法性输入。

### 2.3 五条不可变原则

1. **Bridge 和 route 是顺序独立的 pass。** Bridge 只产生或保留 subject，route
   才决定如何优化该 subject。
2. **factor 只在所属机制中解释。** `bridge_factor` 是 program 合并规模；软件流水
   factor 是 stage 数；重排/向量化 factor 是组宽。
3. **合法不等于物化。** 静态证明、真实改写、正确性和性能是四个不同责任层。
4. **拒绝不能变成 Original。** 不支持的候选要失败或获得无穷大计时，不能静默
   回退后冒充优化候选。
5. **准入必须去定制。** 只使用 IR 结构、SSA 依赖、内存效应、factor、显式目标
   能力和可复核 Provider 证书。

## 3. 总体架构

### 3.1 分层视图

```text
┌──────────────────────────────────────────────────────────────┐
│  Python 控制层                                               │
│  factor ontology → composition legality → Cartesian graph    │
│  → candidate identity → native Autotuner facade              │
└───────────────────────────┬──────────────────────────────────┘
                            │ PlanBundle + per-candidate binding
                            │ （公开仓当前缺少 binding 工厂）
┌───────────────────────────▼──────────────────────────────────┐
│  TTIR 编译控制层                                             │
│  BridgeDiscover → BridgeProgramCoarsening → LoopFacts        │
│  → LoopDecision → native LoopUnroll → LoopMaterialize        │
│  → ValidateLoopPlan                                          │
└───────────────────────────┬──────────────────────────────────┘
                            │ normal Triton lowering
┌───────────────────────────▼──────────────────────────────────┐
│  原生 Triton GPU 后端                                        │
│  TTGIR → native software pipeline/codegen → PTX/cubin        │
└───────────────────────────┬──────────────────────────────────┘
                            │ prebuilt KernelInterface bindings
┌───────────────────────────▼──────────────────────────────────┐
│  原生 autotune                                               │
│  全候选实测 → 最小耗时 winner → native key cache             │
└──────────────────────────────────────────────────────────────┘
```

### 3.2 候选状态机

```text
requested
  ├─ subject absent / schema invalid / dependence unproved → typed rejected
  └─ statically legal
       ├─ Bridge materialization failed                    → failed
       └─ Bridge terminal
            ├─ route materialization failed                → failed
            └─ route terminal
                 ├─ postcondition/lineage mismatch         → failed
                 ├─ correctness mismatch                   → failed
                 └─ executable candidate                   → native autotune
```

性能只决定最后一条路径中哪个 executable candidate 获胜，不能修复前面的任何失败。

### 3.3 建议的 TTIR Pass 顺序

当前实现所表达的顺序为：

```text
triton-loop-bridge-discover
triton-loop-bridge-program-coarsening
triton-hbv-loop-facts
triton-hbv-loop-decision
triton-loop-unroll                    # 原生 Pass，L-lite 扩展 lineage
triton-hbv-loop-materialize
triton-hbv-validate-loop-plan
```

Pass 名与依赖方言定义在
[`Passes.td`](../include/triton/Dialect/Triton/Transforms/Passes.td#L93-L125)，Python
Pass manager wrapper 在
[`passes.cc`](../python/src/passes.cc#L39-L59)。

**当前公开边界：** 这些 wrapper 使调用者可以显式组装 pipeline，但公开分支没有
修改原生默认 TTIR pipeline。`rg` 全仓只会在 `passes.cc` 找到 `add_hbv_*` 和
`add_loop_bridge_*` 注册，不会找到默认 pipeline 注入。

## 4. 因果域与 factor 本体

### 4.1 一个前置构造域、三条 route

| 域 | 参数含义 | 对 subject 的终态 | 实现归属 |
|---|---|---|---|
| Loop Bridge | 合并的逻辑 program 数，或多轴 divisor 乘积 | 新增或保留 route 可消费循环 | L-lite 新增 |
| 软件流水 | pipeline stage 数 | 保留 live loop，跨迭代重叠 load/compute | L-lite 决策；原生 Triton 物理流水化 |
| 完全展开+重排 | phase reorder grouping width | factor 主组展开并按 operation phase 重排；可能保留 tail | L-lite 新增 |
| 完全展开+向量化 | logical vector grouping width | factor 主组展开并合并等价操作；可能保留 tail | L-lite 新增 |

正式 route 常量和 factor 语义见
[`LoopRouteFactorMeaningV1`](../python/triton/l_lite/factor_ontology.py#L54-L101)。

### 4.2 候选空间

对一个 subject，请求空间为：

```text
{Original}
∪ { Bridge(b) → Pipeline(stage) }
∪ { Bridge(b) → FullUnrollReorder(group) }
∪ { Bridge(b) → FullUnrollVectorize(group) }
```

基本候选 identity 为 `(bridge_factor, route_ref, route_factor)`。Bridge factor 和
route factor 不合并；某些 Bridge 构造循环当前要求完整展开 factor 等于 Bridge
trip count，只是当前 materializer 能力限制，类型化原因为
`current_full_unroll_requires_route_factor_equal_trip_count`，不是两个 factor 的语义
相同。规则在
[`certify_loop_bridge_route_composition_v1`](../python/triton/l_lite/contract.py#L79-L125)
和 V2 的
[`certify_loop_bridge_route_composition_v2`](../python/triton/l_lite/contract.py#L173-L241)。

### 4.3 subject 本体

[`LoopRouteSubjectV1`](../python/triton/l_lite/factor_ontology.py#L291-L352) 将 subject
区分为：

- `absent`：没有可用循环；
- `existing_loop` + exact trip count：原 IR 中的静态循环；
- `existing_loop` + runtime main-tail certificate：有动态边界证书的循环；
- `bridge_constructed`：Bridge 构造、trip count 等于 bridge factor 的循环。

[`route_subject_after_bridge_v1`](../python/triton/l_lite/factor_ontology.py#L414-L428)
只完成 subject 转移，不读取 route。这个函数是“Bridge 与三条 route 不混域”的最小
代码证据。

### 4.4 route factor admission

[`decide_loop_route_factor_admission_v1`](../python/triton/l_lite/factor_ontology.py#L493-L586)
按 route 独立计算：

- 软件流水：factor 是 stage 数，不消耗 source trip；
- 重排/向量化：计算 `main_iteration_extent`、`tail_iteration_extent`、
  `main_group_count` 和是否完整消除 source loop；
- 动态 subject：必须携带 main-tail certificate；
- factor 超过精确 trip count时类型化拒绝。

### 4.5 嵌套循环

嵌套设计不是 `outer_route × inner_route`。一个候选只选择一条 route 和一个 factor，
另用结构 scope 表达它穿过的维度。相关对象为
[`LoopNestDimensionV1`](../python/triton/l_lite/factor_ontology.py#L588-L618)、
[`LoopNestedRouteSubjectV1`](../python/triton/l_lite/factor_ontology.py#L619-L664) 和
[`LoopNestStructuralScopeV1`](../python/triton/l_lite/factor_ontology.py#L665-L725)。

完整展开上限可来自结构 mask：只展开外层、只展开内层或展开内外乘积，但仍属于同一
route 的一个结构选择，避免引入互相独立的 outer/inner route 造成无法归因的指数空间。

### 4.6 exact-prefix 的身份

exact-prefix 是完全展开+向量化的动态 recurrence 子型，不是第四条因果 route。
它以独立 artifact route 保留物化身份，但
`mechanism_route_ref` 始终映射到 logical vector route。定义见
[`LoopLogicalVectorSubtypeMeaningV1`](../python/triton/l_lite/factor_ontology.py#L104-L158)
和
[`decide_loop_logical_exact_prefix_admission_v1`](../python/triton/l_lite/factor_ontology.py#L258-L287)。

## 5. Python 控制层设计

### 5.1 `factor_ontology.py`：只定义语义

该文件负责 factor、subject、exact-prefix subtype、嵌套 scope 和 admission 的不可变
含义，不决定性能，也不读取后端计时。对象通过规范化 JSON 的 SHA-256 前缀生成稳定
引用，见 [`_ref`](../python/triton/l_lite/factor_ontology.py#L43-L47)。

设计作用是防止：

- 把 stage count 当成 trip count；
- 把 Bridge 构造规模当成 route 组宽；
- 把 exact-prefix 误加成第四条 route；
- 用函数名或列表下标代替 subject identity。

### 5.2 `contract.py`：两次干预的组合合法性

V1 负责简单 Bridge-by-route 组合；V2 先构造/保留 route subject，再调用 route factor
admission，并携带 exact trip 或 runtime certificate。两个版本都只判断当前组合是否
有明确 subject 和 factor 语义，不判断性能。

核心对象：

- [`LoopBridgeRouteCompositionLegalityV1`](../python/triton/l_lite/contract.py#L53-L77)；
- [`LoopBridgeRouteCompositionLegalityV2`](../python/triton/l_lite/contract.py#L128-L171)。

### 5.3 `composition.py`：请求网格与 intervention lineage

[`build_loop_intervention_cartesian_graph_v1`](../python/triton/l_lite/composition.py#L158-L235)
和 V2 版本
[`build_loop_intervention_cartesian_graph_v2`](../python/triton/l_lite/composition.py#L380-L472)
先枚举全部请求单元，再为每个单元保存合法/拒绝状态。Original 永远是唯一的基线 arm。

[`attest_loop_bridge_route_materialization_v1`](../python/triton/l_lite/composition.py#L584-L702)
从真实 artifact 报告中分别验证：

- Bridge 是否按所选 factor 发生；
- route 是否按所选 factor 发生；
- intervention 顺序是否严格为 `("bridge", "route")`；
- 两个 factor 的 lineage 是否都保留；
- 没有 silent fallback。

### 5.4 `autotune.py`：一笔带过的原生适配

L-lite 不改变 Triton 的 winner 算法。它把每个 candidate ref 放进一个原生
`triton.Config`，通过 mux 分派到对应的预构建 kernel binding，再调用原生
`Autotuner`。`prune_configs_by=None`，不安装 performance model 或 early stop，见
[`LoopNativeAutotuneControlV1`](../python/triton/l_lite/autotune.py#L487-L592)。

公开接口还能读取完整候选计时、总 benchmark 时间和 winner identity。具体算法不是
本设计评审重点。

### 5.5 当前 Python 接线缺口

这是公开架构当前最重要的 P0 问题：

1. `LoopNativeAutotuneControlV1` 要求调用者提供
   `Mapping[candidate_ref, KernelInterface]`；仓库没有把 composition arm 序列化成
   `tt.hbv.plan_bundle`、组装上述七段 pipeline 并编译成 binding 的公开工厂；
2. exhaustive domain 构造器
   [`build_loop_exhaustive_autotune_domain_v1`](../python/triton/l_lite/autotune.py#L371-L405)
   只接收 `LoopInterventionCartesianGraphV1`，尚未接通已经实现的 V2 图；
3. `project_loop_autotune_domain_v1` 可以接收外部 qualification 结果，但 qualification
   生产者不在公开仓中。

因此当前公开 API 能证明“候选域和原生 autotune 适配层的控制逻辑”，但不能只靠公开
API 从普通 `@triton.jit` 函数自动生成全部 L-lite candidate bindings。

## 6. 编译器 Pass 设计

六个新增 Pass 都实现在
[`HBVLoop.cpp`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L1-L10922)，通过
[`CMakeLists.txt`](../lib/Dialect/Triton/Transforms/CMakeLists.txt#L5-L29) 编入
`TritonTransforms`。

### 6.1 `TritonLoopBridgeDiscover`

入口：[`LoopBridgeDiscoverPass`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L4993)。

职责：

- 找到唯一 public entry；
- 解析编译时绑定的 runtime scalar 与 grid extent；
- 证明 program-id 到地址的仿射/混合基数关系；
- 证明跨 program store 不重叠、读写不发生未证 alias；
- 检查 atomic、volatile、未知副作用、call closure 和提前 return；
- 只写入 `tt.loop_bridge.discovery` JSON 事实，不做性能判断。

核心 program 独立性证明在
[`certifyBridgeProgramIndependence`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L2152-L2270)。
它从 pointer root、局部地址跨度、pid stride 和内存 effect 推导，不包含 workload
allowlist。

### 6.2 `TritonLoopBridgeProgramCoarsening`

入口：[`LoopBridgeProgramCoarseningPass`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L6191)。

职责：

- 读取 Bridge factor/axis divisor 和 discovery certificate；
- 将多个逻辑 program 映射到一个物理 program；
- 生成显式 `scf.for`，在循环迭代中恢复逻辑 program-id；
- 对多轴 Bridge 计算 divisor 向量和笛卡尔 program 坐标；
- 克隆或内联可证明纯的 helper；
- 对单个提前 void return 做共享 CFG 谓词化；
- 写入 Bridge origin、factor、subject、ordinal 和 dependence lineage。

Bridge factor=1 时不构造循环。factor>1 时 launcher 仍接收原始逻辑 grid，物理 grid
缩减属于 Bridge binding 的责任，不能由调用者再次手工预除。

### 6.3 `TritonHBVLoopFacts`

入口：[`HBVLoopFactsPass`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L5115)。

它是只读 Provider/事实面，输出 `hbv.loop.static-facts.v12`。主要事实包括：

- load/store/atomic/reduce/loop 数量和类型体积；
- 每个可见循环的静态/动态边界、trip count、carried value；
- 原生 software-pipeline 能力与失败原因；
- 重排、向量化、nested、runtime main-tail 能力；
- program affine/mixed-radix access；
- runtime mask scalar 及其完整性；
- exact-prefix、state-axis sibling graph 和 inner reduction 信息；
- Bridge 的 construction/partition certificate。

逐循环能力 census 的写入位于
[`HBVLoopFactsPass`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L5620-L5699)。
planning-cut locator 只是传输 identity，不是准入特征，相关约束在
[`HBVLoopFactsPass`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L5242-L5253)。

### 6.4 `TritonHBVLoopDecision`

入口：[`HBVLoopDecisionPass`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L7210)。

职责：

- 严格解析模块上的 `tt.hbv.plan_bundle`；
- 检查 pinned compiler commit、route、adapter version、subject policy 和全部字段闭合；
- 按 Provider locator 重新推导 subject，防止 planning cut 到 mutation 之间事实漂移；
- 对 software pipeline 写 `tt.num_stages`；
- 对完整展开 route 写 `tt.loop_unroll_factor`；
- 给操作写临时 role、role_subject、role_index、operation_group lineage；
- 把 PlanBundle ownership 从 module 收束到唯一 public function。

Plan parser 从
[`parseLoopPlan`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L2969) 开始；它使用 exact
key 集合拒绝未知语义，而不是忽略额外字段。决策阶段对 pipeline subject set 的重新
验证见
[`HBVLoopDecisionPass`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L7653-L7756)，
对重排/向量化 existing subject 的规则式选择见
[`HBVLoopDecisionPass`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L7759-L8000)。

### 6.5 原生 `TritonLoopUnroll` 的扩展

L-lite 没有复制原生 unroller，而是在原文件中增加 lineage 保留：

- 在展开前收集 source exact trip、role subject 和每个 role 的数量；
- 调用原生 `loopUnrollByFactor`；
- 给 surviving main/tail loop 或被直接内联的 operation clone 标记分区；
- 静态完整展开、静态有余数和动态 main-tail 都能被后续 materializer 区分；
- 保留原生“epilogue 不再流水化”的行为。

完整差异集中在
[`LoopUnrollPass::runOnOperation`](../lib/Dialect/Triton/Transforms/LoopUnroll.cpp#L45-L188)。

### 6.6 `TritonHBVLoopMaterialize`

入口：[`HBVLoopMaterializePass`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L10108)。

职责按 route 分派：

- Bridge-only：只保留 Bridge subject，禁止 route-local factor/stage；
- software pipeline：确认 exact subject set 和 stage request 留存，物理 pipeline
  lowering 继续由 TritonGPU 原生后端完成；
- phase-major：在 unroll clone lineage 上做 operation-neutral phase 重排；
- logical-group：对 load、store 和一般 elementwise group 分别做 exact packing；
- affine runtime：选择 guarded lanes 或有证书的 main-tail；
- exact-prefix：在目标显式预算内生成 predicated vector container；
- state-axis：把同构 sibling operation graph 沿状态轴打包。

任何 materializer 返回失败都会设置 `tt.hbv.materialization_failure` 并使 Pass 失败，
不会回到 Original。route 分派与失败传播见
[`HBVLoopMaterializePass`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L10387-L10438)。

### 6.7 `TritonHBVValidateLoopPlan`

入口：[`HBVValidateLoopPlanPass`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L10496)。

这是 TTIR 级 fail-closed validator，检查：

- module 级 PlanBundle 已被消费；
- 所有临时 role 和 main/tail clone lineage 已清理；
- route、mechanism route、subtype、artifact route 和 subject 对应；
- Bridge/route factor 与多轴 divisor identity 未丢失；
- dependence certificate 仍匹配；
- Bridge-only 没有偷带 route 参数；
- software pipeline 的 live subject/stage set 完整；
- full-unroll route 没有无解释地保留 focal loop；
- phase/logical/exact-prefix/state-axis 各自的目标 postcondition 成立。

需要注意：该 Pass 验证的是 TTIR/编译计划终态，不等于公开仓已经提供 cubin/SASS 级
artifact identity 检查。最终机器码差异和数值正确性仍需编译/运行 harness 补充。

## 7. Loop Bridge 因果域

### 7.1 核心变化

Bridge 将 program 级并行重复：

```text
program p 执行 work(p)
```

改写为：

```text
physical program q:
  for lane in [0, bridge_factor):
    logical p = q * bridge_factor + lane
    if p < original_grid:
      work(p)
```

其正确性依赖的是 program 间 ownership/footprint 独立，而不是 body 必须出现某种
固定 load/store 配方。只有 store、只有 load 或一般纯计算都不因操作类别本身被拒绝；
关键是副作用、地址和依赖是否可证明。

### 7.2 规则式子域

| 子域 | 准入事实 | 物化方式 | 典型拒绝 |
|---|---|---|---|
| identity | `bridge_factor=1` | 不构造循环 | 原 IR 无 route subject |
| 单轴 coarsening | 唯一 axis、仿射 footprint、写入不重叠 | 一维 `scf.for` 恢复 logical pid | pid stride/局部跨度导致重叠 |
| 多轴 Bridge | 至少两个 divisor>1，乘积闭合 | 混合基数恢复 x/y/z pid | divisor/extent/方向不闭合 |
| 提前 return | 三块 CFG、唯一立即 void return、continuation 可谓词化 | active predicate 包围 continuation | 未证除数、不可谓词化 effect |
| pure helper call | one-level call graph prospectively pure | inline/clone 后构造 | 递归、未知 call、helper 有 effect |
| recursive effect container | region 内 effect 可递归证明 | 保留结构并克隆 | atomic、volatile、alias 未证 |
| 连续分区递推 | 静态连续 virtual program partition | 显式 partition recurrence | 非连续或跨分区 carried dependency |

提前 return 的通用 predication 规则从
[`continuationOperationIsPredicatable`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L410-L429)
开始；它递归检查 reduction combiner，不通过算子名适配。

### 7.3 地址与副作用证明

Bridge 对每个 load/store 追踪唯一 pointer root 和 pid footprint。对于 store，要求跨
logical program 的区间不重叠；对于共享 root 的 read/write，必须能排除 alias。
atomic、volatile load 和未知 region effect 默认拒绝。

仿射局部跨度不会被错误当作零：`localMin/localMax` 一起参与区间比较。动态局部跨度
无法证明时拒绝，而不是乐观准入。相关数据结构从
[`AffinePidFootprint`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L297-L301) 开始。

### 7.4 共享结构能力

Bridge 构造的 loop body 在进入 route 前需要成为普通、可审计的结构 subject：

- helper call 必须已内联或有纯性闭包；
- 提前 return 必须被一次性正规化；
- structured conditional 的谓词化是共享能力，不应分别在三个 route 中复制；
- Bridge 的 factor、origin 和 dependence lineage 必须保留到最终 validator。

这保证动态/结构化循环支持属于公共物化能力，而不是某一 route 的 workload adapter。

## 8. 三条 route 的设计

### 8.1 原生软件流水 route

#### 是什么

设置合法 loop subject 的 `tt.num_stages`，让 TritonGPU 原生 pipeliner 对跨迭代
load/compute 建立重叠。循环不会因 L-lite 本身消失。

#### 准入

[`certifyNativeDynamicPipelineSubject`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L4667)
要求原生可见的 asynchronous load service、闭合的 payload width、可接受依赖和 live
loop。Bridge 构造出循环不代表它天然可流水化；如果 body 没有后端可识别的 load
service，应该类型化拒绝。

#### 物化边界

L-lite 的 materializer 验证 subject set 和 stage request 未丢失；真正的 async copy、
commit/wait 或后端 schedule 由原生 Triton 后续 pipeline 产生。因此“TTIR stage request
成立”和“最终机器码出现软件流水事实”需要两个不同测试层。

### 8.2 完全展开 + phase-major 重排

#### 是什么

原生 unroller 先产生 factor-wide clone。L-lite 再按 source operation group 建立依赖闭包，
以稳定拓扑顺序把同一 phase 的操作聚合，同时保持跨 phase 的 SSA/effect 约束。

#### 去定制设计

当前实现使用 operation-neutral group，而不是要求固定“一个 load + 一个 compute + 一个
store”。入口为
[`materializeOperationNeutralPhase`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L8124)。
它按 lineage 和 dependency closure 重排；没有 load 就不会产生 load phase，但不因此
拒绝整个候选。

#### main/tail

当 factor 小于 trip 或不能整除 trip 时，factor 主组重排，ordered tail 保留。只有
factor 完整覆盖 exact trip 时 focal loop 才完全消失。validator 接受 surviving loop
marker 或 operation-owned partition lineage 两类证据。

### 8.3 完全展开 + logical-group 向量化

#### 是什么

把展开后的同构操作组按 logical lane 打包，而不是仅改变源码写法。目标是让后端看到
更宽的 tensor/向量数据流。

#### exact packing

当前实现把三类规则分开：

- load group：指针、mask、other、边界与缓存语义精确对应；
- store group：指针、value、mask 和副作用顺序精确对应；
- general elementwise group：操作名、类型、属性和 operand graph 同构。

通用 exact operation packing 的入口为
[`materializeExactOperationVectorization`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L9413)。
不会因为 body 不符合某个算子配方而隐性拒绝。

#### Bridge logical

Bridge 构造的 lane 可通过 `tt.join` 合并；invariant hoisting、tensor lane fusion 和 exact
split elision 是可独立记录的物化能力。入口为
[`materializeBridgeLogical`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L9114)。

#### exact-prefix 子型

动态 active extent 被放入静态 power-of-two container，以 predicate 保证 inactive lane
不参与语义，再执行向量 reduction/操作。它需要 Provider 给出 exact integer prefix
证书和显式 element budget。实现门位于
[`HBVLoopMaterializePass`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L10171-L10203)。

#### sibling state-axis SLP

若连续多个对象拥有同构 operation graph，可沿状态轴打包；路数不是固定 4，而由
`stateCardinality` 决定，可 pad 到合法容器宽度。准入比较完整 operation graph、reduction
combiner 和 cross-state consumer，不以 `r0/r1/r2/r3` 名称识别。核心 region 描述在
[`StateAxisNormalizationRegion`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L205-L218)。

## 9. Plan、属性与证书协议

### 9.1 PlanBundle

编译决策通过模块属性 `tt.hbv.plan_bundle` 输入。parser 要求：

- `schema_version=1`；
- `project_kind="loop"`；
- route、decision、subject locator 和 candidate parameters 闭合；
- minimal provenance 固定 compiler commit；
- dynamic binding、guard、feedback、fallback 的字段数和语义精确匹配 adapter。

未知字段不是向前兼容地忽略，而是拒绝，避免新旧 schema 在同一次编译中混用。

### 9.2 关键属性分组

| 分组 | 代表属性 | 用途 |
|---|---|---|
| Plan/subject | `tt.hbv.plan_bundle`, `tt.hbv.l.subject`, `tt.hbv.l.subject_ref` | 绑定唯一决策和对象 |
| route identity | `route`, `mechanism_route`, `route_subtype`, `artifact_route` | 区分因果机制、子型和物化器 |
| composition | `composition_schema`, `composition_bridge_factor`, `composition_route_factor`, axis divisors | 保留 Bridge→route 两次干预 |
| Bridge | `tt.loop_bridge.factor`, `origin`, `subject`, `discovery`, grid divisors | 发现和构造证据 |
| temporary lineage | `role`, `role_subject`, `role_index`, `operation_group_ref`, `unroll_partition_lineage` | 原生 unroll 前后关联 clone |
| terminal | `realized`, `postcondition`, `composition_postcondition`, `dependence_certificate` | 最终 fail-closed 验证 |
| failure | `tt.hbv.materialization_failure` | 让失败显式传播 |

常量全集位于
[`HBVLoop.cpp`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L44-L136)。

### 9.3 adapter version

C++ parser 仍能读取多代 schema，但将部分历史 adapter 明确标为“evidence-readable,
not production-executable”，见
[`parseLoopPlan`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L3123-L3139)。这避免旧证据
无法读取，同时防止旧行为重新进入生产。

代价是 `HBVLoop.cpp` 内保留了大量历史路径和版本分支。当前 active capability 可以
是规则式的，但源码文件并非已经物理删除所有历史 materializer。这是维护性问题，
不能表述为“源码中完全没有旧路径”。

## 10. 合法性、物化、正确性与性能的责任边界

| 层 | 证明的问题 | 当前公开实现 | 不能推出 |
|---|---|---|---|
| 可观察性 | IR 中是否有对象或 Bridge 可构造对象 | Discover/Facts | 候选合法 |
| 合法性 | 依赖、副作用、factor 和 subject 是否闭合 | ontology/contract/Decision | 改写一定发生 |
| 物化 | 目标 route 的 TTIR 后置事实是否存在 | Materialize/Validate | 数值一定正确、机器码一定不同 |
| artifact | 最终编译产物是否非 Original、目标后端事实是否出现 | 公开仓缺少统一 end-to-end harness | 性能更快 |
| 正确性 | Original、候选与独立 oracle 是否一致 | 需具体运行 harness | 性能更快 |
| 性能 | 哪个正确候选实测更快 | 原生 autotune | 因果解释自动成立 |

因此公开说明中的“final artifact”应严格解释：C++ validator 当前是 TTIR route/artifact
lineage 后置验证；cubin/SASS identity 和运行正确性要由公开端到端 harness 才能完全
复现。

## 11. 与原生 Triton 的逐文件差异

### 11.1 修改原生文件（4 个）

| 文件 | 改动 | 架构作用 |
|---|---|---|
| `include/triton/Dialect/Triton/Transforms/Passes.td` | 新增六个 Pass 定义 | 将 L-lite 能力注册为正式 MLIR Pass |
| `lib/Dialect/Triton/Transforms/CMakeLists.txt` | 编译 `HBVLoop.cpp` | 接入 `TritonTransforms` |
| `lib/Dialect/Triton/Transforms/LoopUnroll.cpp` | 增加 source trip、role 和 main/tail lineage | 让原生 unroll 可被 route materializer/validator 审计 |
| `python/src/passes.cc` | 导出六个 Python Pass wrapper | 允许显式组装 pipeline |

### 11.2 新增实现文件（6 个）

| 文件 | 内容 |
|---|---|
| `lib/Dialect/Triton/Transforms/HBVLoop.cpp` | 10,922 行的发现、证书、Plan parser、Bridge/route 物化和验证 |
| `python/triton/l_lite/factor_ontology.py` | factor/subject/subtype/nested scope 本体 |
| `python/triton/l_lite/contract.py` | Bridge→route V1/V2 组合合法性 |
| `python/triton/l_lite/composition.py` | 笛卡尔图、arm identity 和物化 attestation |
| `python/triton/l_lite/autotune.py` | 原生 exhaustive autotune facade |
| `python/triton/l_lite/__init__.py` | 公开 Python API |

### 11.3 新增测试（14 个文件）

- `python/test/unit/l_lite/test_control.py`：4 个控制层测试；
- `test/Triton/hbv-loop-*.mlir`：13 个 Bridge/事实层 MLIR 测试。

### 11.4 文档性改动（4 处，含 README 和本文）

- `README.md` 的 L-lite 发布说明；
- `docs/l-lite-understanding-guide.zh-CN.md` 的因果域理解手册；
- `docs/l-lite-capability-validation-2026-08-25.md` 的构建验证记录；
- 本文是新增的评审架构设计书，提交后文档总增量将成为 4 个文件。

## 12. 去定制审计

### 12.1 当前 active 规则没有读取的变量

- kernel/function 名；
- 算子类别名；
- 仓库、benchmark、测试集身份；
- 输入 shape 作为样本 identity；
- 温度、功率、时钟、utilization；
- 历史 winner、预测收益或计时标签。

### 12.2 当前允许读取的事实

- loop trip/bound/step、nest 和 carried values；
- SSA dependency 和 operation effect；
- load/store pointer root、mask、alignment、affine footprint；
- program-id axis、grid extent 和 mixed-radix mapping；
- dtype、tensor width、目标显式资源预算；
- Provider locator/certificate 和 route factor admission。

### 12.3 仍需评审的隐性定制风险

1. `HBVLoop.cpp` 中有多代 adapter 和少量窄结构 matcher。虽然 active V20 authority
   可将其排除，公开源码仍应逐个证明这些 matcher 是结构规则而非算子组合模板；
2. runtime top-k、exact-prefix 等历史命名可能让读者误认为按算子准入，应统一改为
   结构语义命名或在接口层隐藏；
3. 10,922 行单文件提高了共享 helper 与历史 adapter 交叉污染的风险。

## 13. 测试与已有证据

### 13.1 已执行验证

公开提交已经完成：

```text
TritonRelBuildWithAsserts: passed
libtriton.so + triton-opt: built
python/test/unit/l_lite/test_control.py: 4 passed
test/Triton/hbv-loop-*.mlir: 13 passed
maximum build parallelism: 4
```

独立验证记录见
[`l-lite-capability-validation-2026-08-25.md`](l-lite-capability-validation-2026-08-25.md)。

### 13.2 公开 MLIR 测试覆盖

| 测试 | 覆盖 |
|---|---|
| `hbv-loop-bridge-scan.mlir` | 基本 Bridge 构造与 Facts |
| `hbv-loop-bridge-early-return.mlir` | CFG predication 正反例 |
| `hbv-loop-bridge-affine-local-span*.mlir` | 局部 footprint 安全/重叠/动态和真实构造 |
| `hbv-loop-bridge-pure-call-closure.mlir` | pure/effect/nested call 边界 |
| `hbv-loop-bridge-recursive-effect-container*.mlir` | recursive region、alias、atomic、volatile |
| `hbv-loop-bridge-operation-neutral.mlir` | 不要求固定操作配方 |
| `hbv-loop-bridge-ungrouped-axis-translation*.mlir` | 未分组轴的地址平移 |
| `hbv-loop-mixed-radix-independence.mlir` | 多轴 mixed-radix 正反例 |
| `hbv-loop-runtime-mask-authority.mlir` | runtime scalar mask 的完整绑定 |
| `hbv-loop-body-semantics.mlir` | runtime scalar 改变 body 事实 |

### 13.3 公开测试缺口

公开 13 个 MLIR 测试没有系统执行完整
`Discover→Bridge→Facts→Decision→Unroll→Materialize→Validate`，也没有覆盖三条 route
的全部正反物化、最终 PTX/cubin identity、GPU 数值 oracle 或真实性能。Python 4 个
测试只覆盖 V1 图、factor equality 限制、原生全候选计时和 key cache。

因此当前公开可重复证据支持：

- Bridge 的若干通用合法性/构造规则；
- Python V1 候选控制和原生 autotuner facade；
- 整个源树能够编译。

它还不足以仅凭公开测试证明：

- 所有 C++ adapter 都闭环；
- V2 图可以端到端 autotune；
- 三条 route 在公开 harness 中都物化且正确；
- 存在任何性能收益。

## 14. 构建与运行边界

### 14.1 构建

L-lite 保持 Triton 原生构建体系，只新增 `HBVLoop.cpp` 到 `TritonTransforms`。正常可
编辑安装需要 NVIDIA/AMD backend symlink；CUDA 的 `ptxas`、`cuobjdump`、`nvdisasm`
是外部依赖，不提交到仓库。

### 14.2 当前可直接使用的内容

- 用 `triton-opt` 显式运行新增 Pass；
- 从 Python Pass manager 显式插入新增 Pass；
- 构造 V1/V2 composition graph；
- 对外部预构建 bindings 使用原生 autotune facade；
- 运行公开 CPU/MLIR 测试。

### 14.3 当前不能仅靠公开 API 自动完成的内容

- 从普通 `@triton.jit` kernel 自动发现全部 candidate domain；
- 为每个 candidate 生成对应 PlanBundle；
- 把 PlanBundle 注入编译并返回 KernelInterface binding；
- 自动运行完整 artifact/correctness qualification；
- 将 V2 graph 直接提交给 exhaustive autotune。

## 15. 架构优点

1. **因果边界清楚。** Bridge 与 route 分离，同时通过笛卡尔积实现联合优化；
2. **factor 不混义。** stage、program grouping 和 vector/reorder group 有独立本体；
3. **fail-closed。** Plan 字段、依赖证书和 postcondition 不匹配即编译失败；
4. **不靠 workload allowlist。** 主体准入从 IR/SSA/effect 推导；
5. **复用原生组件。** unroll、GPU pipeline 和 autotune 不重复实现；
6. **支持动态与嵌套的结构化扩展。** 通过证书/subtype/scope 扩展，而非增加算子特例；
7. **适合作为主 L 的对照。** 共享变换能力，选择端分别是 exhaustive measurement 和
   HBV prediction。

## 16. 架构风险与整改优先级

### P0：评审前必须明确，后续应闭环

1. **公开端到端 binding 工厂缺失。** 需要一个公开、可测试的
   `candidate → PlanBundle → pass pipeline → compiled binding` 路径；
2. **V2 图没有接入 autotune。** 应增加 V2 domain adapter，或删除“V2 已端到端”的任何
   暗示；
3. **公开 route 物化证据不足。** 应为三条 route 增加完整 pipeline 正反例、silent
   fallback 检查和 GPU correctness harness；
4. **artifact 层术语过强。** TTIR postcondition、PTX identity、cubin identity 和
   机器码机制事实应分别命名。

### P1：可维护性与可信度

1. 将 10,922 行 `HBVLoop.cpp` 按 Facts/Bridge/Plan/Phase/Logical/Validate 拆分；
2. 将生产 adapter registry 与历史 evidence reader 物理分层；
3. 给每个 active adapter 建立 schema→Provider proof→materializer→postcondition 的静态
   注册表，减少 parser、decision、materializer 三处平行分支漂移；
4. 将 runtime-topk 等窄历史名称改为纯结构语义；
5. 用一套生成式 schema 同时生成 Python contract 和 C++ parser，减少 V1/V2/V3
   手工不一致。

### P2：工程体验

1. 提供最小公开 demo kernel 与一条命令完成候选编译和 autotune；
2. 自动输出每个 observable loop 的三 route 覆盖矩阵；
3. 把 typed rejection 汇总为用户可读报告；
4. 将本文和理解手册纳入 docs index/CI link check。

## 17. 建议的端到端闭环设计

### 17.1 是否应该产品化

结论是 **应该**。理由不是为了增加功能层数，而是现有公开架构的科学对照责任要求它
必须独立可复现：

- 没有统一 CandidateCompiler，外部调用者可以用不同方式构造 bindings，L-lite 与
  主 L 未必实际比较同一候选；
- 没有公开 PlanBundle serializer，Python ontology 与 C++ parser 的一致性无法由
  用户复查；
- 没有完整编译/正确性 disposition，autotune 无法区分“不合法”“没有物化”“结果
  错误”和“性能较慢”；
- 没有统一 acquisition ledger，失败候选花费的编译时间可能从对照开销中消失。

产品化不等于修改 Triton 的全局默认行为。建议提供显式 opt-in API，例如
`triton.l_lite.compile_candidates(...)`；未调用该 API 时，原生 Triton pipeline、JIT
和 autotune 行为完全不变。

### 17.2 产品模块

为了把现有源码能力变成真正独立、可复现的 L-lite，建议补充以下公共模块：

```text
LoopCensus
  → enumerate Bridge/route factors
  → V2 composition graph
  → typed PlanBundleSerializer
  → CandidateCompiler
       ├─ inject runtime scalar/grid facts
       ├─ run frozen seven-stage TTIR pipeline
       ├─ reject on any pass/postcondition failure
       └─ return artifact identity + KernelInterface
  → CorrectnessQualifier
  → LoopNativeAutotuneControlV2
  → Report
```

其中 `CandidateCompiler` 必须是唯一 authority，不能让调用者在外面手工设置一组 TTIR
属性。每个 binding 需要携带：candidate ref、PlanBundle hash、TTIR/PTX/cubin hash、
route postcondition 和 correctness disposition。

建议的模块责任如下：

| 模块 | 输入 | 输出 | 不允许做的事 |
|---|---|---|---|
| `LoopCensus` | JIT function、specialization、launch facts | 稳定 subject/能力 census | 使用计时或 kernel allowlist |
| `CandidateEnumerator` | census、Bridge/route factor domain | V2 requested Cartesian graph | 根据预测收益删单元 |
| `PlanBundleSerializer` | V2 arm、Provider certificate | C++ parser 可接受的 exact schema | 手写未注册 adapter 字段 |
| `CandidateCompiler` | source kernel、PlanBundle、target | typed compile outcome、artifact hashes、binding | silent fallback 或吞掉失败 |
| `CorrectnessQualifier` | Original、candidate、oracle contract | correct/incorrect/ambiguous | 用性能结果决定容差 |
| `NativeAutotuneControlV2` | 全请求 disposition 和可运行 bindings | winner、逐候选 timings | perf model、early stop、预测剪枝 |
| `AcquisitionReport` | 所有 compile/qualification/autotune event | 完整总开销与 typed ledger | 隐藏失败候选成本 |

### 17.3 建议的公开 API

```python
product = triton.l_lite.compile_candidates(
    kernel,
    specialization=...,
    launch=...,
    bridge_factors=(1, 2, 4, 8),
    route_factors={
        SOFTWARE_PIPELINE_ROUTE: (2, 3, 4),
        FULL_UNROLL_REORDER_ROUTE: (2, 4, 8, 16),
        FULL_UNROLL_VECTOR_ROUTE: (2, 4, 8, 16),
    },
    correctness=oracle_contract,
)

winner = product.autotune(*args, grid=grid, key=(...))
report = product.report()
```

`product` 至少要公开：

- 完整 requested graph；
- 每个 cell 的 PlanBundle hash 和 typed disposition；
- compile wall time，包括失败 cell；
- TTIR/PTX/cubin identity 与 route postcondition；
- correctness 结果；
- 原生 autotune 的逐候选耗时、总 acquisition 和 winner。

### 17.4 失败候选如何保证公平

“完整候选产品”不能为了安全把非法程序送上 GPU，也不能为了表面公平把失败成本
删除。正确做法是：

1. 先固定并枚举完整 requested graph；
2. 对每个 cell 都执行相同的 candidate build attempt 并记录 wall time；
3. 静态拒绝、编译失败、物化失败或 correctness 失败都保留 typed disposition；
4. 给原生 Autotuner 保留该 candidate identity，通过 fail binding 返回失败/无穷大，
   但不执行错误 kernel；
5. 最终 acquisition 同时报告候选编译、正确性和原生 benchmark 总开销。

这样没有替 autotune 隐藏本应承担的候选探索成本，也不会牺牲程序安全。

### 17.5 实施顺序与成功判断

建议按以下顺序实现：

1. 生成式 PlanBundle serializer 与 C++ parser round-trip；
2. 单候选显式 pipeline compiler 和 artifact identity；
3. V2 graph 全候选 compiler、typed failure binding 和 acquisition ledger；
4. correctness qualifier；
5. NativeAutotuneControlV2；
6. 三条 route 的公开完整 pipeline 测试；
7. kernel/端到端 benchmark 与主 L 同候选域对照。

产品“完成”的最低标准不是 API 能返回 winner，而是：任一请求 cell 都有且只有一个
可复核 disposition；所有 executable cell 的 Plan、artifact、correctness 和 timing
identity 一致；请求域没有无解释消失；总 acquisition 包含失败成本。

## 18. 评审验收清单

### 18.1 架构一致性

- [ ] Bridge factor 与 route factor 在 schema、代码和报告中始终独立；
- [ ] exact-prefix 始终是 logical-vector subtype；
- [ ] nested loop 只使用一条 route、一个 factor 和一个结构 scope；
- [ ] software pipeline 的物理 lowering 明确归属原生 Triton；
- [ ] active adapter 不依赖 kernel/operator/benchmark identity。

### 18.2 编译正确性

- [ ] 每个 candidate 有唯一 PlanBundle 和唯一 binding；
- [ ] 默认 pipeline 或显式 compiler API 的执行顺序唯一；
- [ ] 每条 route 有正例、最早层拒绝反例和 silent-fallback 反例；
- [ ] validator 失败会阻止候选进入 autotune；
- [ ] Original 始终保留且不携带 L-lite 干预。

### 18.3 公开可复现性

- [ ] V2 graph 已接 native autotune；
- [ ] 公开 demo 可以从源码 kernel 生成所有 bindings；
- [ ] 公开测试覆盖完整七段 pipeline；
- [ ] GPU correctness 与 artifact identity 独立于性能测试；
- [ ] 性能报告明确测试环境、总 autotune acquisition 和 winner。

## 19. 最终评审结论

L-lite 的核心架构方向是正确的：它没有把 Bridge 和三条 route 混成一个无法归因的
mega-pass，也没有为了 exhaustive autotune 引入性能预测；它用真实编译 Pass、类型化
subject/factor 和 fail-closed postcondition 建立了一个可扩展的循环变换能力面。

但当前公开版本更准确的定位是：

> **“已编译通过的 L-lite 编译器能力面 + V1 exhaustive-autotune 控制 facade”，而不是
> “从普通 Triton kernel 到 exhaustive winner 的开箱即用完整产品”。**

明天评审最应聚焦的不是原生 autotune 细节，而是三个闭环问题：

1. 如何公开并固定 candidate→PlanBundle→binding 的唯一编译入口；
2. 如何让 V2 规则式 subject/factor 图真正成为 exhaustive control 的 active authority；
3. 如何用公开完整 pipeline、artifact 和 correctness 测试证明三条 route 无回退。

这三项闭环后，L-lite 才能作为主 L 项目的严格实验对照：两者共享完全相同的观察对象、
合法性和物化能力，只在“exhaustive autotune”与“HBV 预测发布”这一后端选择方式上不同。

## 附录 A：代码阅读顺序

1. [`factor_ontology.py`](../python/triton/l_lite/factor_ontology.py#L1)；
2. [`contract.py`](../python/triton/l_lite/contract.py#L1)；
3. [`composition.py`](../python/triton/l_lite/composition.py#L1)；
4. [`Passes.td`](../include/triton/Dialect/Triton/Transforms/Passes.td#L93-L125)；
5. [`HBVLoop.cpp` 常量与 Plan`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L44-L203)；
6. [`LoopBridgeDiscoverPass`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L4993)；
7. [`HBVLoopFactsPass`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L5115)；
8. [`LoopBridgeProgramCoarseningPass`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L6191)；
9. [`HBVLoopDecisionPass`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L7210)；
10. [`LoopUnroll.cpp`](../lib/Dialect/Triton/Transforms/LoopUnroll.cpp#L24-L195)；
11. [`HBVLoopMaterializePass`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L10108)；
12. [`HBVValidateLoopPlanPass`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L10496)；
13. 最后阅读 [`autotune.py`](../python/triton/l_lite/autotune.py#L453-L616)。

## 附录 B：变更审计命令

```bash
git diff --name-status 7c56a5e40..25bca1f4e
git diff --stat 7c56a5e40..25bca1f4e
git diff 7c56a5e40..25bca1f4e -- \
  include/triton/Dialect/Triton/Transforms/Passes.td \
  lib/Dialect/Triton/Transforms/CMakeLists.txt \
  lib/Dialect/Triton/Transforms/LoopUnroll.cpp \
  python/src/passes.cc
```

这组命令给出本文的原生 Triton 对照边界；`HBVLoop.cpp`、`python/triton/l_lite/`、
公开测试和文档均为该基线之后新增。

## 附录 C：`25bca1f4e` 的 27 个文件级差异清单

```text
M  README.md
A  docs/l-lite-capability-validation-2026-08-25.md
A  docs/l-lite-understanding-guide.zh-CN.md
M  include/triton/Dialect/Triton/Transforms/Passes.td
M  lib/Dialect/Triton/Transforms/CMakeLists.txt
A  lib/Dialect/Triton/Transforms/HBVLoop.cpp
M  lib/Dialect/Triton/Transforms/LoopUnroll.cpp
M  python/src/passes.cc
A  python/test/unit/l_lite/test_control.py
A  python/triton/l_lite/__init__.py
A  python/triton/l_lite/autotune.py
A  python/triton/l_lite/composition.py
A  python/triton/l_lite/contract.py
A  python/triton/l_lite/factor_ontology.py
A  test/Triton/hbv-loop-body-semantics.mlir
A  test/Triton/hbv-loop-bridge-affine-local-span-materialize.mlir
A  test/Triton/hbv-loop-bridge-affine-local-span.mlir
A  test/Triton/hbv-loop-bridge-early-return.mlir
A  test/Triton/hbv-loop-bridge-operation-neutral.mlir
A  test/Triton/hbv-loop-bridge-pure-call-closure.mlir
A  test/Triton/hbv-loop-bridge-recursive-effect-container-materialize.mlir
A  test/Triton/hbv-loop-bridge-recursive-effect-container.mlir
A  test/Triton/hbv-loop-bridge-scan.mlir
A  test/Triton/hbv-loop-bridge-ungrouped-axis-translation-materialize.mlir
A  test/Triton/hbv-loop-bridge-ungrouped-axis-translation.mlir
A  test/Triton/hbv-loop-mixed-radix-independence.mlir
A  test/Triton/hbv-loop-runtime-mask-authority.mlir
```

本架构设计书是第 28 个差异文件；它描述而不改变上述实现快照。

# 第二部分：L 主项目的预测、校准与安全发布体系

## 20. 为什么在 L-lite 之后介绍主 L

L-lite 与主 L 不是两套循环编译器。两者应共享完全相同的：

- 循环观察对象和 Provider 事实；
- Bridge、软件流水、完全展开+重排、完全展开+向量化的因果语义；
- factor/subject ontology；
- 合法性、物化、artifact、正确性和类型化拒绝；
- 完整 CandidateCompiler 与 acquisition 账本。

它们只在正确候选已经形成后分岔：

```text
                                ┌─ L-lite：全部实测 → native autotune winner
共享 CandidateCompiler ─────────┤
                                └─ 主 L：分域预测 → 安全下界 → hot-key 发布
```

因此主 L 的科学问题是：在不对每个候选做 exhaustive timing 的前提下，能否用可解释的
上游因果链与低容量后端统计代理，安全判断哪个候选值得发布。L-lite 则给出“如果愿意
支付全部 acquisition，最佳候选是什么”的实验对照。

## 21. 主 L 的完整证据链

主 L 目标链条如下：

```text
IR / Provider 事实
  → Pass 前强语义状态
  → Pass 专属因果转移
  → 可识别核心环境状态
  → 弱语义传播
       ├─ 传播受 Pass 改变的终态属性
       └─ 传播不受该 Pass 改变、但决定时间的环境属性
  → 跨代弱语义学习残差
  → 完整预编译终态
  → Candidate 编译与 typed PTX/live-shape
  → 分域后端统计代理
  → O/B/C 时间估计与相对加速比
  → 小幅时间残差
  → 小幅环境残差
  → 独立校准的机制区间
  → hot-key 生命周期发布
  → 后编译事实与真实执行时间更新下一代
```

这条链的约束是 **最早责任原则**：如果 Provider、强语义、弱语义或代理中任一层出现
系统性反例，必须回到该层重推；不能用下游残差、扩大区间或 exact lookup 抹平上游
错误。

## 22. 强语义、弱语义与环境状态

### 22.1 强语义状态

强语义只表达当前 Pass 直接造成的结构变化。三条 route 分域建模：

| 因果域 | 直接干预 | 强语义终态 |
|---|---|---|
| Bridge | program ownership 合并 | physical/logical program 关系、构造循环及 program-footprint 证书 |
| 软件流水 | stage request | live loop、异步 load service、stage 和跨迭代 service 结构 |
| 完全展开+重排 | factor grouping + phase reorder | main/tail、operation group、依赖保持与重排终态 |
| 完全展开+向量化 | factor grouping + exact packing | logical group、vector container、main/tail 与 subtype 终态 |

“后续所有发生变化的 IR 字段”不都属于强语义。Pass 直接变化是因，随后由该变化触发
的 instruction count、live range、资源压力等多为中间变量或后端响应。把所有终态字段
塞进强语义会把因果方向倒置。

### 22.2 弱语义传播

弱语义模型承担两种传播：

1. 把强语义中受 Pass 影响的核心状态传播到预编译终态；
2. 把强语义不直接改变、但时间代理需要的可识别环境状态传播到终态。

第二类例如 dtype、元素宽度、逻辑 program 数、输入/输出字节、算术量、trip/tail、
目标架构显式容量等。它们不能因为“不在当前 Pass 因果域”就统一丢给环境残差。

### 22.3 弱语义学习残差

弱语义的主体映射是有因果解释的传播模型。学习残差只修正同一因果分支在编译代际或
同类目标上的小幅系数漂移：

- 必须 generation-delayed，不能读取同一代真实时间再修正同一代决策；
- 代理定义或输入语义改变时必须清零，从头拟合；
- 容量必须低于主体代理，且只能在独立数据上证明“小而稳定”；
- 因果分支翻转不是残差，应进入独立架构迁移风险报告。

## 23. 为什么采用 O/B/C 顺序估计量

Bridge 和 route 是两个顺序独立 Pass。如果只比较 Original 与最终 Candidate，目标
`log(T_O/T_C)` 同时包含 Bridge 和 route 两种作用，某个 route 模型会被迫解释不属于
自己的 Bridge 收益。

主 L 因而对每个组合候选保留三个真实 artifact：

```text
O = Original
B = Bridge-only terminal
C = Bridge terminal + one route terminal
```

定义：

```text
L_bridge       = log(T_O / T_B)
L_route|bridge = log(T_B / T_C)
L_total        = log(T_O / T_C)
L_total        = L_bridge + L_route|bridge
```

最后一个式子是代数恒等式，不是需要拟合的第五个模型。Bridge factor=1 时，B 必须与
O artifact-identical，`L_bridge=0` 精确成立，不能用两次有噪声的重复 timing 估计。

这一分解带来四个独立预测组件：

1. 一个所有 route 共用的 Bridge `O→B` 组件；
2. 软件流水的 `B→C` 组件；
3. 完全展开+重排的 `B→C` 组件；
4. 完全展开+向量化的 `B→C` 组件。

route 组件可以读取完整、命名明确的 B 终态，但不能重新拥有 Bridge factor 的因果
解释。联合候选的预测由两个组件相加得到。

## 24. typed PTX/live-shape 后端统计代理

### 24.1 为什么需要统计代理

上游可以白盒解释 Pass 做了什么，但寄存器分配、spill、instruction scheduling、
issue overlap 等后端行为不能总从 TTIR 精确推导。主 L 不以回归模型替换上游因果链，
而是在上游终态闭合后，以较粗粒度的后端响应块预测时间。

这形成“上游因果闭环 + 后端统计代理闭环”：

- 上游字段必须有 Provider/强/弱语义 producer；
- 后端代理只接收 Candidate 编译前或 typed PTX 后、真实执行前可获得的字段；
- 代理内部不宣称每个系数都是独立物理因果；
- 代理整体必须在 family-disjoint 数据上形成小圈子闭环。

### 24.2 最小充分粗因果状态

代理状态分两层：

#### P0：机制核心

- Bridge/route factor 的正确语义；
- logical/physical program 数；
- input/output bytes 和 arithmetic elements；
- trip、main/tail 和 tail fraction；
- route-specific stage/group/subtype；
- 显式目标架构能力；
- B 终态中由 route 消费的命名服务状态。

#### P1：typed PTX/backend shape

- instruction count 和命令类别的粗分组；
- memory instruction count；
- peak typed live slots 与 live-area；
- O→B 或 B→C 的 typed PTX 轴向比值；
- 不携带 kernel/family identity 的 live-shape response。

P1 不能因为训练误差下降就自动进入模型。只有在 paired whole-family bootstrap 中，P1
相对 P0 在独立外层 fold 上显著降低发布风险，才允许加深代理。

### 24.3 禁止输入

当前代理不得读取：

- kernel/family identity、源码路径或 hash 作为特征；
- 温度、功率、时钟、utilization、测量顺序；
- 原始 pointer 值；
- 同代真实执行时间；
- ptxas 的最终 allocation/spill 结果、SASS 或 winner identity；
- 旧版本误差、残差或已经披露的 qualification/sealed 标签。

温度、功率和 utilization 可以用于判断测量环境是否有效，但不能进入因果预测链。

### 24.4 后端深度停止门

如果当前粗代理已经满足独立 family 的精度、方向、区间和 false-adoption 门，就停止
继续拆解后端。继续加深必须同时满足：

1. 新字段在 decision time 有独立、可解释的 producer；
2. 不是从最终时间反推出来的隐性 identity；
3. 在预注册的独立 holdout 上显著降低发布风险；
4. 没有显著增加跨架构重新学习成本。

因此目标不是复制整个 GPU backend，而是找到足以支撑安全发布的最浅统计闭环。

## 25. 代理模型族与最小参数原则

### 25.1 domain-local first

四个组件先独立选择自己的状态、交互和模型。允许复用相同的特征变换、归一化和低容量
算法，但不能为了强行统一而改变某个因果域的最小充分状态。只有各域先闭环后，才分析
能否建立有条件的部分共享模型。

### 25.2 当前低容量模型路线

当前预注册路线使用 label-free farthest-center RBF ridge：

1. 在任何 timing label 进入前，用 feature geometry 选择中心；
2. normalization 与中心选择都在每个训练 fold 内完成；
3. 用 whole physical family 做独立单位和 leave-one-family-out；
4. ridge 只拟合 RBF 权重，不选择 identity patch；
5. 每个 fold 的 effective degrees of freedom 不得超过训练 family 数的一半；
6. Bridge 允许比单 route 稍高但仍有界的中心候选；route 使用更小中心候选；
7. 同精度时选择字段更少、中心更少、有效自由度更低的模型。

RBF 是当前预注册候选，不是不可修改的永恒结论。如果系统性 subgroup 失败，外层循环
必须重新审查：

- 参数集合；
- 因果域划分；
- 交互图；
- 模型族；
- 参数可识别性和 family 支撑。

不能只在现有 RBF 上增加一个局部 patch。

### 25.3 最小参数组合验收

若两个模型的 family-disjoint 误差、方向覆盖、区间覆盖和 false-adoption 都在可接受
范围内，则认为精度等价，并选择：

```text
更少 causal fields
  → 更少 interaction blocks
  → 更少 centers / 更低 effective DoF
  → 更浅 backend depth
```

“最小”不是只比较参数数量，而是在发布后果等价的条件下比较模型容量。

## 26. 内外循环如何工作

### 26.1 内层循环

内层只修复已经确定的最早 owner，例如：

- schema/序列化实现错误；
- 同一证书公式实现与定义不一致；
- 一个已注册 proxy block 的数值实现错误；
- fold-local normalization 或 capacity gate 的 bug。

内层禁止加入 kernel 名、shape identity、温度、功率、测量顺序、same-generation outcome
或只服务一个失败样本的 indicator。

### 26.2 外层循环

任一注册 subgroup 出现系统性 signed bias，按最早责任回退：

```text
事实错             → Provider
Pass 变化错         → Strong
终态传播错          → Weak
对象放错域          → domain partition
字段/交互不足       → backend proxy state / interaction graph
同一状态仍不稳定    → model family / granularity
独立校准才出现尾部  → residual / interval
环境不合格          → measurement protocol，整角色无效
```

subgroup 至少包括因果域、Bridge factor、route factor/stage、trip 是否精确、tail fraction、
memory geometry 是否可识别和 typed PTX response sign pattern。family identity 是重采样
单位，不是模型特征。

## 27. 数据分层与防拟合设计

### 27.1 Development

用于选择 P0/P1、模型容量和外层修复。必须按 whole physical family 分 fold。development
中的正预测只表示方向诊断，不具有 release authority。

### 27.2 Qualification

在模型结构、字段、中心选择规则和容量冻结后才打开。qualification 可以判断模型是否
合格，但不能反向修改当代模型。

### 27.3 Calibration / untouched holdout

只用于校准残差和预测区间、检查方向/覆盖/false-adoption。任何系统性 subgroup 失败都
返回上游，而不是塞进区间。

### 27.4 Sealed hot keys

只在前面全部门通过后打开，用于最终生命周期价值与发布后果。sealed 结果永不调整
同一代模型、阈值或候选空间。

### 27.5 数据污染处理

无法证明是在要求的独占/稳定环境下采集的 timing，只能保留为工程诊断或失效证据，
不得进入模型、残差、区间、holdout 或发布。过去的“证伪”若依赖未证明独占的计时，
必须降级并在干净环境重测。

## 28. 残差、区间、查表与迁移风险的分工

| 对象 | 学习什么 | 数据时序 | 不能吸收什么 |
|---|---|---|---|
| 弱语义学习残差 | 完整弱语义/代理映射跨代的小漂移 | generation-delayed | 因果分支翻转、上游字段缺失 |
| domain time residual | 合格分域代理剩余的小时间偏差 | fresh calibration | 系统性 subgroup、错误域划分 |
| environment residual | 主要可观测环境因素传播后的小 common-mode 漂移 | 同环境独立校准 | 温度/功率特征、外部干扰、无效测量 |
| mechanism radius | in-domain 不可消除尾部的 q80/q95 覆盖 | family-calibrated | 架构迁移、错误 center、未建模机制 |
| exact measured lookup | 同一 hot key 已执行后的实测复用 | post-execution | 新 key 的因果证明、模型资格 |
| architecture migration risk | 跨架构后因果边可能翻转的模型有效性风险 | 独立迁移审计 | 当前架构区间或发布决策 |

这些对象不能相加后互相吸收。特别是：

- 当前架构的模型内不确定性进入残差/机制半径；
- 跨架构因果翻转只进入独立迁移风险报告；
- 迁移风险不能扩大当前发布区间；
- 无法解释的环境变量宁可使证书失败，也不能污染强语义→Pass→窄域时间链。

## 29. 预测区间与发布下界

模型先预测四个组件中心：

```text
L̂_total = L̂_bridge + L̂_route|bridge
```

残差和机制半径只有在组件中心通过独立 qualification 后才允许拟合。区间至少要报告
q80 和 q95，且以 whole family 校准。发布使用的是 total benefit 的安全下界，而不是
中心点：

```text
lower_bound(L_total) > 0
```

仍不足以直接发布，还要把 acquisition 和生命周期成本换算到 hot key 的预计复用次数。
如果覆盖失败、interval 过宽或出现 false adoption，发布回到 Original。

## 30. hot-key 生命周期价值

### 30.1 为什么只发布极热 kernel

一次性或低复用 kernel 无法摊销 Candidate 编译、代理推理、正确性和必要校准开销。
主 L 应优先识别长期重复的 hot key；冷 key 直接使用 Original，不做深度优化。

### 30.2 选择规则

对每个 hot key，在 Original 和所有合法、正确、区间合格的组合候选中，最多选择一个
安全收益下界最高的 route/factor。不是每个 route 各选一个，也不是在同一 kernel 内
同时发布多套互相冲突的全局计划。

### 30.3 价值公式

概念上的净生命周期价值为：

```text
gross_saved
  = expected_reuse_count ×
    (T_original - T_candidate - steady_runtime_overhead)

net_value
  = gross_saved
    - candidate_compile_acquisition
    - required_measurement_or_correctness_acquisition
    - proxy_inference_cost
    - cache/storage_cost
    - execution_interference_cost
```

只有安全下界下的 `net_value>0` 才允许发布。报告必须同时给出冷 key 排除机会、
false-adoption、break-even reuse count 和 steady overhead。

## 31. 生产运行与反馈

建议的生产流程为：

```text
第一次遇到 specialization/key
  → 共享 CandidateCompiler 生成合法且正确候选
  → 主 L 读取预执行 causal state + typed PTX/live-shape
  → 四组件代理预测中心和区间
  → heat/lifecycle gate
  → 选择 Original 或一个 Candidate
  → 执行
  → 保存 key、Plan/artifact identity、环境有效性和真实 T_execution
  → 只用于下一代 residual/lookup/update
```

保存实际运行 key 和结果本身开销很小，应保留；但同代执行结果不能回填同代发布证书。
exact lookup 可以让同一 key 后续直接复用真实数据，却不能证明未见 key 的泛化。

## 32. 环境 authority

高精度性能角色不应强绑定某个固定设备编号。正确设计是 device-selected：

1. 只读发现一张允许使用且空闲的物理 GPU；
2. 记录 index、UUID、架构和 driver/runtime identity；
3. 获取共享 request/advisory lock；
4. 连续观察无外部 context、SM idle 和允许的 slowdown；
5. 在一个不可拆分的 O/B/C 原子角色内持续监控；
6. 任一外部 PID、请求切换或 stationarity 失败使整个角色无效；
7. 无效角色不进入任何模型/残差/区间。

GPU index 只是本机枚举，不是证据 identity；UUID 和连续环境观察才是。编译/正确性
角色可以使用共享 GPU，但必须显式禁止 performance authority。

## 33. 架构迁移

迁移到新架构时把弱语义边分成三类：

| 情况 | 处理 |
|---|---|
| 结构与因果分支保持，系数小漂移 | generation-delayed learning residual 可调整 |
| 因果分支可能翻转 | 独立报告失效误差和发布风险；不自动扩大当前区间 |
| 数据不足，无法判断是否翻转 | 仍报告风险；保持 Original 或由使用者决定是否重新拟合 |

若重新拟合所需事实可在正常 Triton 编译/运行反馈中自动获得，可提供固定自动拟合；若
需要用户源码安装、手工调试或重新开发整个工程，则不能作为 wheel 用户的隐含前提，
只能报告风险并保守不发布。

## 34. 论文证书与 HBV 服务边界

### 34.1 正向闭环

论文可信的正向结论需要逐层证明：

- 每个 Provider 字段有独立 producer；
- 强语义只拥有 Pass 直接变化；
- Weak 正确传播受影响状态和核心环境状态；
- typed PTX/live-shape 代理使用最小充分状态；
- 四组件模型 family-disjoint 合格；
- 残差小、区间覆盖且无系统 subgroup；
- sealed hot-key 下界有正生命周期价值；
- 生产行为与证书后果一致。

### 34.2 合法失败出口

如果每个 in-scope 模块都完成责任，而剩余误差可证实来自不可观测后端分配、无法暴露
的架构状态或外部 measurement validity，则可以归入 HBV 服务边界之外。不能仅因问题
困难就提前退出。

### 34.3 变量剥离账本

最终必须单独记录从“工程成功”到“论文证书版”剥离的每个变量：

- 旧工程为何使用它；
- 它看似带来多少收益/精度；
- 为什么没有独立因果 producer；
- 剥离后哪些 key、候选或收益消失；
- 为什么不能归到学习残差、环境残差、time residual、机制半径或 exact lookup；
- 最终是修复上游、保守排除，还是归入 HBV 服务边界之外。

只有到最终合法出口冻结后，才生成只读的逐 key“工程成功版→论文归因版”收益变迁
报告；该报告是审计结果，不是内外循环的控制变量。

## 35. L-lite 与主 L 的接口契约

| 阶段 | L-lite | 主 L | 是否必须一致 |
|---|---|---|---|
| census/Provider | 使用 | 使用 | 是 |
| factor/subject ontology | 使用 | 使用 | 是 |
| PlanBundle/CandidateCompiler | 使用 | 使用 | 是 |
| 合法性/物化/artifact/correctness | 使用 | 使用 | 是 |
| 请求 candidate domain | exhaustive | 同一完整域后由发布门选择 | 是，不能少能力 |
| candidate selection | native timing | 四组件代理安全下界 | 否 |
| acquisition | 全部编译+全部实测 | 全部必要编译+少量/延迟实测 | 分别报告 |
| residual/interval | 无 | 有 | 不共享 |
| hot-key lifecycle gate | 对照报告 | 生产发布 authority | 不共享 |

共享 CandidateCompiler 是两者公平对比的关键：后续新增通用 subject 或 route
materializer 时，应一次实现并同步暴露给两条选择路径，而不是维护两份行为近似的编译器。

## 36. 当前完成状态

截至本文评审版本，可以确认：

- 当前主 L 与 L-lite 已同步一套规则式合法性/物化源码能力；
- 当前开发 population 的完整请求网格已完成 typed disposition、真实物化、TTIR
  postcondition 和正确性闭环；
- O/B/C 顺序估计量、四组件 typed-PTX 代理、低容量模型协议、外层回退规则和独立
  population 分层已经在性能标签前冻结；
- 旧 mixed `O→C` route 估计量和旧残差不再具有当前 authority。

尚不能确认：

- 新四组件代理的开发精度、方向和 subgroup 闭环；
- qualification/holdout 的独立覆盖；
- generation-delayed residual、环境 residual 和 q80/q95 区间；
- sealed hot-key 的正生命周期价值；
- 最终论文发布或生产 Go。

根因不是合法性/物化仍未闭环，而是顺序估计量重构后尚未采集一套满足严格、可切换
device authority 的新鲜 O/B/C 性能标签。当前 residual、radius 和 release 必须保持
关闭，不能引用旧总估计量标签提前宣布成功。

## 37. 主 L 预测体系的评审重点

1. O/B/C 三 artifact 是否由同一 CandidateCompiler 产生并在一个原子角色测量；
2. 四组件是否真的 domain-local，是否存在 route 模型偷偷读取 Bridge patch 字段；
3. P0/P1 promotion 是否使用 whole-family 独立证据；
4. 同精度下是否选择最小 causal field/interaction/center 组合；
5. 系统性 subgroup 是否回到最早 owner，而非进入残差；
6. 环境 telemetry 是否只做 validity，不进入预测；
7. residual、radius、migration risk 和 exact lookup 是否严格分离；
8. qualification、holdout、sealed 是否都不修改同代模型；
9. lifecycle 是否计入 Candidate 编译、推理、存储、干扰和失败成本；
10. 工程收益减少是否最终具有逐 key、逐原因、只读变迁账本。
