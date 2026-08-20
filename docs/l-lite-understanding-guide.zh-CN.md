# L-lite 中文理解手册：从因果域理解 Loop Bridge 与三条优化路线

## 1. L-lite 是什么

L-lite 是一个循环联合优化的穷举 autotune 对照组。它不使用 L 主项目的 HBV
性能预测器，也不根据预测收益提前剪枝。它保留两类能力：

1. 判断一个候选在语义上是否合法；
2. 把合法候选真正物化为不同的 IR/机器码，并验证没有悄悄回退到 Original。

随后，L-lite 把请求的候选空间交给 Triton 原生 autotune 实测，让原生 autotune
选择最快候选。因此它回答的是：如果愿意支付穷举编译和实测开销，当前这些循环
变换本身能获得什么结果。

L-lite 的核心关系不是“一个混合 pass”，而是两个顺序独立的决策：

```text
原始 IR
  │
  ├─ Bridge：是否把多个 program 合并成一个显式循环主体
  │           bridge_factor = 1 表示不构造
  │
  └─ Route：对 Bridge 之后已有或新构造的循环选择一种机制
              ├─ 原生软件流水
              ├─ 完全展开 + phase-major 重排
              └─ 完全展开 + logical-group 向量化
```

候选空间的基本形式是：

```text
bridge_factor × route × route_factor
```

Bridge factor 和 route factor 的含义不同，不能因为某些物化器当前要求两者相等，
就把它们解释成同一个参数。

## 2. “pass”“route”“因果域”和“子域”

- **Pass** 是编译器中的执行单元。Bridge discovery、Bridge materialization、
  route decision、route materialization 和最终验证分别由编译 pipeline 中的 pass
  承担。
- **Route** 是一种代码变换机制。L-lite 有三条 route。
- **因果域** 是一类具有相同“干预 → 状态变化 → 物化后果”的机制。两个机制即使
  共用一个 C++ pass 外壳，只要改变 IR 的原理不同，就仍然属于不同因果域。
- **子域（adapter）** 是同一机制面对不同结构化输入时的规则化入口和物化方式。
  子域不能由 kernel 名、算子名或测试集身份触发，只能由 IR 事实、依赖证书、
  factor 限制和目标能力触发。

按这个定义，L-lite 包含一个前置构造域和三个 route 因果域：

| 因果域 | 是否由 L-lite 新增物化 | factor 的含义 | 核心终态 |
|---|---:|---|---|
| Bridge 构造域 | 是 | 合并的逻辑 program 数或多轴 divisor 乘积 | 产生 route 可消费的循环主体 |
| 软件流水域 | 否，复用 Triton 原生实现 | pipeline stage 数 | 循环仍存在，但 load/compute 跨迭代重叠 |
| 完全展开 + 重排域 | 是 | 每组完全展开并进行 phase-major 排列的宽度 | 消除所选循环或留下主组/尾部结构 |
| 完全展开 + 向量化域 | 是 | logical group 的宽度；exact-prefix 子域有独立语义 | 多个标量/迭代对象合成为可向量执行的组 |

“L-lite 主要实现三个 pass”通常指 Bridge、完全展开+重排、完全展开+向量化。
软件流水仍是完整候选空间中的第三条 route，但其变换复用 Triton 原生 pass，
L-lite 只负责发现、绑定、合法性检查和最终物化事实验证。

**对应代码：** 三条 route 及各自 factor 的正式语义定义在
[`LoopRouteFactorMeaningV1`](../python/triton/l_lite/factor_ontology.py#L55-L101)；
六个编译阶段的 pass 名称在
[`Passes.td`](../include/triton/Dialect/Triton/Transforms/Passes.td#L93-L125)；
route 虽共享 `triton-hbv-loop-materialize` 外壳，但在
[`HBVLoopMaterializePass`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L8805-L9154)
中分派给不同 materializer。

## 3. 公共入口：事实、合法性、物化和后置验证

所有因果域都经过同一组责任边界：

```text
IR / launch facts
  → 找到结构化 subject
  → 生成因果域专属合法性证书
  → 绑定 Bridge factor 与 route factor
  → 执行 Bridge，再执行 route
  → 检查目标 route 的后置条件和 artifact identity
  → 正确性验证
  → 原生 autotune 计时与缓存
```

这里有三个不能混淆的结论：

1. **合法**只说明按当前证明可以安全尝试，不说明编译后一定产生目标机器码；
2. **物化成功**要求出现目标 route 的终态事实，发生回退不能算成功；
3. **性能更快**只能由 autotune 实测得出，不能反过来替代合法性或物化证明。

### 3.1 因果中立的准入原则

准入规则只能读取下列类型的事实：

- 循环边界、步长、静态 trip count 或有证书的动态主尾关系；
- SSA 数据依赖、循环携带依赖、内存读写与副作用；
- program-id 与地址的仿射关系、不同 program 的输出是否不相交；
- 元素类型、张量宽度、对齐、目标架构显式能力和物化预算；
- Provider 给出的稳定 loop locator、嵌套关系和能力证书。

kernel 名、仓库名、算子类别名、测试集身份、历史计时和预期收益都不能决定合法性。

### 3.2 类型化拒绝

不满足规则的候选必须在最早能确定的层次拒绝，例如：subject 不存在、factor
不是合法的幂次、动态循环缺少主尾证书、存在未证明的跨迭代依赖、目标能力不足，
或物化后的 route identity 不成立。拒绝不是把候选悄悄改成 Original。

在穷举对照模式中，请求的笛卡尔积不会因为性能预测而缩小；无法形成正确可执行
artifact 的单元会作为失败/无穷大结果处理，而不是作为一个错误程序参与比速。

**对应代码：** Bridge 后的 route subject 和 factor 准入由
[`certify_loop_bridge_route_composition_v2`](../python/triton/l_lite/contract.py#L173-L241)
闭合；完整请求网格由
[`build_loop_intervention_cartesian_graph_v1`](../python/triton/l_lite/composition.py#L158-L235)
枚举；公平对照域由
[`build_loop_exhaustive_autotune_domain_v1`](../python/triton/l_lite/autotune.py#L371-L406)
保留全部请求单元；Bridge 与 route 的物化后置条件分别由
[`attest_loop_bridge_route_materialization_v1`](../python/triton/l_lite/composition.py#L584-L702)
核对。离线测试直接检查了
[`完整网格`](../python/test/unit/l_lite/test_control.py#L49-L58)、
[`类型化 factor 限制`](../python/test/unit/l_lite/test_control.py#L60-L72)和
[`原生 autotune 不剪枝`](../python/test/unit/l_lite/test_control.py#L75-L105)。

## 4. 因果域 A：Loop Bridge 构造域

### 4.1 是什么

Bridge 把原来由多个 Triton program instance 分别执行的工作，变成一个物理
program 内的显式循环。`bridge_factor = 1` 是身份变换；大于 1 时表示每个物理
program 顺序服务多个逻辑 program。

### 4.2 为什么

许多 kernel 的 TTIR 中本来没有 `scf.for`，但 program-id 维度上存在规则、互不
干扰的重复工作。没有 Bridge，这类代码就没有可供软件流水、完全展开+重排或完全
展开+向量化消费的循环 subject。Bridge 的目的只是把这种 program 级重复暴露为
显式循环，不负责决定后续采用哪条 route。

### 4.3 怎么做

Bridge discovery 先证明 program 间独立：每个逻辑 program 的地址足迹必须能够
由 program-id 的仿射表达确定，写入不能跨逻辑 program 冲突，控制流和副作用必须
能够安全复制或谓词化。物化时，它计算组内逻辑 program-id，构造循环并克隆原
program body；原始 program-id 的使用被替换为当前循环迭代对应的逻辑 id。

物化后仍要验证：

- 构造出的循环数和 factor 与计划一致；
- program 独立性证书仍成立；
- Bridge lineage 没有丢失；
- 若继续执行 route，最终同时保留 Bridge factor 和 route factor 的独立身份。

**对应代码：** 只读发现由
[`LoopBridgeDiscoverPass`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L4573-L4692)
实现，program 间地址独立性由
[`certifyBridgeProgramIndependence`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L1886-L1996)
证明，实际改写入口是
[`LoopBridgeProgramCoarseningPass`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L5727-L6498)。
Python 侧用
[`route_subject_after_bridge_v1`](../python/triton/l_lite/factor_ontology.py#L414-L437)
表达“Bridge 只产生 subject、不导入 route 语义”。

### 4.4 子域 A0：不构造（identity Bridge）

**是什么：** `bridge_factor = 1`，保留输入 IR。

**为什么：** 原始 kernel 可能已经有合法循环；同时 Original 和“已有循环 + route”
必须留在候选空间中，才能与 Bridge 构造路径公平比较。

**怎么做：** 不新增循环，只把 Provider 找到的已有循环作为 route subject。若原 IR
没有合适循环，后续 route 必须以 `route_subject_absent` 类型化拒绝，不能虚构对象。

**对应代码：** identity、existing、runtime 和 absent 四种 subject 状态见
[`LoopRouteSubjectV1`](../python/triton/l_lite/factor_ontology.py#L291-L412)；
factor 为 1 时直接返回 existing subject 的分支见
[`route_subject_after_bridge_v1`](../python/triton/l_lite/factor_ontology.py#L414-L437)。

### 4.5 子域 A1：单轴 program coarsening

**是什么：** 沿一个 launch axis，把连续若干 program 合成一个循环；常用请求因子
为 2、4、8，规则本身接受正的 2 的幂。

**为什么：** 这是最直接、最容易证明 program 间独立的 Bridge 形式，也能给三条
route 提供统一的显式 `scf.for` subject。

**怎么做：** 证明唯一 program-id、仿射地址关系和不相交写入，计算分组后的物理
grid，再在新循环中恢复每个逻辑 program-id。动态 launch 标量只作为本次编译的
事实绑定，不作为性能特征。

**对应代码：** runtime 标量和 grid extent 的事实绑定在
[`LoopBridgeDiscoverPass`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L4588-L4657)；
单轴 program loop 的构造、克隆和 lineage 写入在
[`LoopBridgeProgramCoarseningPass`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L6294-L6498)。
最小编译检查见
[`hbv-loop-bridge-scan.mlir`](../test/Triton/hbv-loop-bridge-scan.mlir#L1-L27)。

### 4.6 子域 A2：结构化控制流与提前 return

**是什么：** program body 中存在单个提前 `return` 时，把可证明的 continuation
转成谓词化执行，再进行 Bridge。

**为什么：** 提前 return 会使直接克隆的循环体在第一次不活跃迭代时退出整个物理
program，破坏后续逻辑 program 的执行。

**怎么做：** discovery 证明 return 条件可界定、continuation 中的操作可安全谓词
化，然后用 active predicate 保护后续工作。存在不可谓词化副作用、无界除法或
嵌套控制流时拒绝；不会针对某个函数名写特殊分支。

**对应代码：** 允许的 continuation 操作规则在
[`continuationOperationIsPredicatable`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L374-L388)，
CFG 识别和谓词化改写在
[`normalizeSingleEarlyVoidReturnCFG`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L404-L532)，
正反例测试在
[`hbv-loop-bridge-early-return.mlir`](../test/Triton/hbv-loop-bridge-early-return.mlir#L1-L58)。

### 4.7 子域 A3：多轴 Bridge

**是什么：** 同时对 x/y/z 中至少两个轴分组；整体 Bridge cardinality 等于三个
axis divisor 的乘积。

**为什么：** 某些重复结构分布在多个 launch 轴，仅按 x 轴合并无法表达真实的
逻辑组。

**怎么做：** Provider 为每个使用的轴分别证明 program 独立，Plan 显式绑定三维
divisor 向量，materializer 用 mixed-radix 方式恢复各轴逻辑 id。向量必须保留到
最终 artifact identity；只保存乘积会混淆不同的多轴变换。

**对应代码：** 多轴 subject 的不可变语义和乘积校验在
[`LoopBridgeAxisVectorSubjectV1`](../python/triton/l_lite/factor_ontology.py#L356-L412)；
divisor 解析、逐轴证明和 mixed-radix body 克隆在
[`LoopBridgeProgramCoarseningPass`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L5733-L6051)；
Plan 决策阶段重新核对完整 divisor 向量在
[`HBVLoopDecisionPass`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L6883-L6918)。

### 4.8 子域 A4：连续分区递推

**是什么：** Bridge 形成的相邻逻辑 program 之间存在可证明的有符号商/余数递推，
而不是完全互不相关的 body。

**为什么：** 重复从头计算分区边界会浪费工作；但递推关系属于 Bridge 构造出来的
subject 状态，不等同于 route 的重排或向量化机制。

**怎么做：** 仅在直接 program-id 分区关系可由整数算术严格推出时绑定递推证书，
由循环迭代携带前一分区的边界。若关系不闭合，则仍可走普通 Bridge，或在 Bridge
本身不合法时类型化拒绝。

**对应代码：** 分区模式发现由
[`findDirectPidPartition`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L1998-L2070)
完成；Bridge 在 CFG/单轴路径中分别重验并物化该递推，入口见
[`LoopBridgeProgramCoarseningPass`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L6091-L6258)
和[普通单轴分支](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L6330-L6498)；
决策阶段的证书复核见
[`HBVLoopDecisionPass`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L6857-L6882)。

## 5. 因果域 B：原生软件流水 route

### 5.1 是什么

软件流水让不同循环迭代的加载、计算和写回在不同 stage 上重叠。其 route factor
表示 pipeline stage count，不表示循环 trip count，也不表示展开宽度。

### 5.2 为什么

当循环存在可提前发起的 load，并且迭代间依赖允许重叠时，软件流水可以隐藏内存
延迟。它也是判断另外两条新 route 覆盖度的重要原生参考。

### 5.3 怎么做

L-lite 不重写 Triton 原生软件流水算法。它负责：

1. 选择一个由 Provider 明确定位的循环；
2. 绑定 stage count；
3. 调用原生 pipeline；
4. 检查最终仍存在唯一 focal loop，并且出现原生流水物化事实。

stage count 与 Bridge factor 独立。例如 Bridge 构造 4 次循环，并不意味着只能用
4 stages。

**对应代码：** `pipeline_stage_count` 与“保留 live focal loop”的语义在
[`ROUTE_FACTOR_MEANINGS_V1`](../python/triton/l_lite/factor_ontology.py#L85-L101)；
软件流水 factor 不受 subject trip 限制的准入分支在
[`decide_loop_route_factor_admission_v1`](../python/triton/l_lite/factor_ontology.py#L493-L579)；
决策 pass 只写入原生 `tt.num_stages` 在
[`HBVLoopDecisionPass`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L6926-L6936)，
物化阶段不重写算法、只核对请求仍存在于唯一 subject 在
[`HBVLoopMaterializePass`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L9018-L9034)。

### 5.4 子域 B1：已有循环的软件流水

**是什么：** 对原 TTIR 已存在、原生 pipeline-capable 的循环直接应用软件流水。

**为什么：** 这是 Triton 原生能力，也是 L-lite 的基准 route。

**怎么做：** Provider 使用原生能力条件证明循环存在可调度的 load/compute 阶段，
然后由 Triton 原生 pass 物化。仅“存在 `scf.for`”不足以证明可流水化。

**对应代码：** 每个可观测循环的 native pipeline 能力及另两条 route 覆盖矩阵由
[`_build_loop_mechanism_coverage_v1`](../python/triton/l_lite/autotune.py#L54-L144)
投影；最终必须保持唯一 software-pipeline subject 的验证在
[`HBVValidateLoopPlanPass`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L9359-L9371)。

### 5.5 子域 B2：Bridge 构造循环的软件流水

**是什么：** 先由 Bridge 把 program body 变成循环，再尝试原生软件流水。

**为什么：** 这检验 Bridge 构造的循环是否能像普通 TTIR 循环一样被原生 pipeline
消费。

**怎么做：** Bridge 必须把可流水 load 直接暴露在循环体中，且保持地址、依赖和
控制流结构。若 body 被 helper call 隐藏、load 依赖前序计算、含不能分 stage 的
reduction/间接地址或嵌套区域，原生 pipeline 可能合法地拒绝。此时要在失败阶段
区分“Bridge 没有暴露普通循环结构”和“循环本身没有可流水机制”。

**对应代码：** Bridge 为下游 route 暴露 body 的 CFG 正规化注释和实现见
[`normalizeSingleEarlyVoidReturnCFG`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L390-L532)；
Bridge 物化后生成的 loop subject 在
[`LoopBridgeProgramCoarseningPass`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L6429-L6498)，
decision 随后把该 subject 绑定为 pipeline focal loop 在
[`HBVLoopDecisionPass`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L6837-L6953)。

### 5.6 子域 B3：嵌套循环的软件流水

**是什么：** 一个 route 作用于一个 Provider 定位的嵌套 subject，而不是给内外层
分别选择互相独立的 route。

**为什么：** 内外层的调度和依赖会相互影响；把它们误拆成两个自由 route 会产生
没有统一 artifact identity 的候选。

**怎么做：** Provider 必须给出嵌套维度、父子 locator 和独立的 pipeline subject
证书。stage count 仍独立于嵌套展开上限。当前实现对没有独立嵌套证书的循环明确
拒绝，不能由“内外层 trip count 都已知”推断可流水化。

**对应代码：** 嵌套 subject 的维度与父子 identity 在
[`LoopNestDimensionV1` 和 `LoopNestedRouteSubjectV1`](../python/triton/l_lite/factor_ontology.py#L588-L663)；
结构 scope 的枚举在
[`enumerate_loop_nest_structural_scopes_v1`](../python/triton/l_lite/factor_ontology.py#L726-L738)。
当前原生 pipeline 对 nested region 的具体能力仍由原生后端负责，L-lite 不从
这些 Python 结构规则推断其可流水化。

## 6. 因果域 C：完全展开 + phase-major 重排

### 6.1 是什么

这条 route 先把选定范围内的迭代完全展开，再把原来的 iteration-major 顺序重排
为 phase-major 顺序。直观上，原来是：

```text
迭代 0: load → compute → store
迭代 1: load → compute → store
```

重排后更接近：

```text
load(0), load(1) → compute(0), compute(1) → store(0), store(1)
```

实际顺序由依赖拓扑决定，不是机械地把所有同名操作放在一起。

### 6.2 为什么

完全展开消除循环控制和归纳变量；phase-major 重排进一步暴露跨迭代的 load-level
parallelism、公共地址计算和更长的独立指令窗口，使后端有机会改善调度。

### 6.3 怎么做

Provider 先证明所选迭代之间没有禁止重排的依赖，并给出精确 trip 或有证书的动态
主尾关系。materializer 克隆迭代、保留 SSA 对应关系和 store 顺序，再按依赖允许的
phase 重新排列。最终验证要求 focal loop 被消除，或只留下有明确 main/tail lineage
的组循环和尾部；仅设置 unroll attribute 不算物化成功。

route factor 是“本组完整展开并重排的宽度”。当 factor 等于所选 subject 的完整
展开上限时，subject 可以完全消失；当 factor 小于 trip count 时，每个主组内部
完全展开，外层组循环或有序尾部仍可存在。

**对应代码：** route factor 的 main/tail 守恒、完整消除条件和 typed reason 在
[`decide_loop_route_factor_admission_v1`](../python/triton/l_lite/factor_ontology.py#L493-L579)；
一般 phase-major 重排由
[`materializePhase`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L7489-L7753)
实现；统一分派及失败即停止在
[`HBVLoopMaterializePass`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L9045-L9099)；
最终检查 phase-major postcondition 在
[`HBVValidateLoopPlanPass`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L9515-L9523)。

### 6.4 子域 C1：已有精确静态循环

**是什么：** 原 TTIR 中已有静态边界、精确 trip count 的 `scf.for`。

**为什么：** 它提供最直接的完整展开与依赖证明。

**怎么做：** factor 不得超过 trip count；整除时生成若干完整组，不整除时保留
有序 tail。factor 等于 trip count 时可完全消除 focal loop。

**对应代码：** 静态 subject 的 main/tail 计算在
[`decide_loop_route_factor_admission_v1`](../python/triton/l_lite/factor_ontology.py#L529-L579)；
decision 给目标循环写入 unroll factor、保持 subject identity 的主体在
[`HBVLoopDecisionPass`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L7135-L7345)，
最终 main/tail lineage 的判定在
[`HBVLoopMaterializePass`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L9105-L9146)。

### 6.5 子域 C2：有证书的动态主尾循环

**是什么：** trip count 在运行时才知道，但 Provider 已证明可按 factor 分成完整
main groups 和保持原序的 tail。

**为什么：** 动态循环不应被一概排除；关键是能否证明分组覆盖全部原迭代且不重复、
不越界。

**怎么做：** 编译器构造 main/tail 或有序 partial-unroll 结构，完整展开每个 main
group，并让 tail 继续保持原始语义。没有 runtime main-tail certificate 时拒绝。

**对应代码：** 无动态证书即拒绝的本体规则在
[`decide_loop_route_factor_admission_v1`](../python/triton/l_lite/factor_ontology.py#L554-L579)；
仿射动态顺序证书由
[`certifyAffineRuntimeOrderPreserving`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L4502-L4571)
生成；两种物化路径分别是
[`materializeAffineRuntimePartialUnroll`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L8595-L8662)
和
[`materializeAffineRuntimeMainTail`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L8664-L8803)，
其选择及后置检查见
[`HBVLoopMaterializePass`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L8927-L9017)。

### 6.6 子域 C3：Bridge 构造 subject

**是什么：** 对 Bridge 新构造的 program loop 完全展开并做 phase-major 重排。

**为什么：** Bridge 自身只是把多个逻辑 program 串行放入一个循环；不做 route
时仍是 program-major。重排 route 才是第二个独立干预。

**怎么做：** 先验证 Bridge factor 和构造 lineage，再内联可能隐藏 body 的 helper，
暴露每个逻辑 program 的阶段，按依赖拓扑重排。当前完整消除物化通常要求 route
factor 覆盖构造 subject 的 trip；这一等式是当前 materializer 能力限制，不是两个
factor 在语义上相同。

**对应代码：** Bridge→route V2 合法性和 factor 的独立 owner 在
[`certify_loop_bridge_route_composition_v2`](../python/triton/l_lite/contract.py#L173-L241)；
helper 展开由
[`inlineBridgeHelpers`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L7882-L7969)
负责，Bridge 专属 phase-major 重排由
[`materializeBridgePhase`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L7757-L7880)
负责。当前等式限制有单独测试，见
[`test_full_unroll_equality_is_a_typed_materializer_limit`](../python/test/unit/l_lite/test_control.py#L60-L72)。

### 6.7 子域 C4：静态完整嵌套循环

**是什么：** 一个连接的静态 loop nest 作为一个 subject，由一条 phase-major route
控制。

**为什么：** 内外层都展开时，完整展开上限可能是外层 trip、内层 trip 或两者乘积；
用一个无结构的标量会混淆“展开了哪一层”。

**怎么做：** Provider 保存每层 locator、父子关系和 trip vector，并用 dimension
mask 指明所选结构范围。对 phase-major route，route factor 必须等于该范围的完整
展开上限。`route × factor` 仍只有一组；不会额外生成 outer route 和 inner route。

**对应代码：** 嵌套维度、subject、结构 mask 和完整展开上限分别在
[`LoopNestDimensionV1`](../python/triton/l_lite/factor_ontology.py#L588-L617)、
[`LoopNestedRouteSubjectV1`](../python/triton/l_lite/factor_ontology.py#L619-L663)和
[`LoopNestStructuralScopeV1`](../python/triton/l_lite/factor_ontology.py#L665-L724)；
phase factor 必须等于 scope cardinality 的规则在
[`decide_loop_nested_route_factor_admission_v1`](../python/triton/l_lite/factor_ontology.py#L781-L836)。
C++ 对完整静态 nest 的逐层绑定在
[`HBVLoopDecisionPass`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L6955-L7047)。

### 6.8 子域 C5：Provider 绑定的多 subject / 嵌套叶子

**是什么：** 一组彼此独立的已有循环，或嵌套结构中有独立 sink 证书的叶子循环。

**为什么：** 一个 kernel 可能同时包含多个结构相同但位置不同的循环；只选“第一个
循环”既不稳定，也容易造成隐性定制。

**怎么做：** Provider 用稳定 locator 和 member list 绑定每个对象，逐成员保存
factor、能力证书及父子关系。物化后必须逐成员核对 identity；任一成员发生漂移，
整个候选拒绝。

**对应代码：** `ParsedLoopPlan::ProviderBoundMember` 保存 locator、父子、嵌套证书和
成员 factor，定义在
[`HBVLoop.cpp`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L166-L176)；
嵌套叶子独立性由
[`certifyNestedInnerDimension`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L6586-L6623)
重验；逐 member 选择自己的 route factor 并调用相同因果域 materializer 在
[`HBVLoopMaterializePass`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L9055-L9095)。

## 7. 因果域 D：完全展开 + logical-group 向量化

### 7.1 是什么

这条 route 同样从完整展开开始，但目标不是仅改变指令阶段顺序，而是把多个迭代或
多个同构逻辑对象打包成一个 logical group，使加载、算术、归约或状态更新以张量/
向量形式执行。

### 7.2 为什么

多个标量路径经常执行同构操作。逐路径执行会重复地址生成和标量指令；打包后可以
减少指令数量、形成更宽的访问，并给后端的向量执行和 SLP 合并提供明确结构。

### 7.3 怎么做

Provider 证明组成员在形状、类型、操作图、依赖和副作用上可共同执行。materializer
完整展开目标范围，收集同一逻辑位置的值，使用 join/reshape/broadcast/reduce 等
TTIR 结构形成组，再把结果正确拆回各自消费者。最终验证要求看到 logical-group
归约、向量化 load group、state-axis packed graph 或对应子域的专属 artifact；只
展开而没有形成组不算成功。

**对应代码：** logical route 的两种顶层 subtype 在
[`LoopLogicalVectorSubtypeMeaningV1`](../python/triton/l_lite/factor_ontology.py#L105-L158)
中定义；一般已有循环的物化入口是
[`materializeLogical`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L8401-L8521)，
顺序保持的 load 向量化在
[`materializeOrderPreservingLoadVectorization`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L8234-L8399)，
最终 logical artifact 检查在
[`HBVValidateLoopPlanPass`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L9438-L9513)。

### 7.4 子域 D1：普通 grouped-load / grouped-iteration

**是什么：** 对已有或 Bridge 构造的 `scf.for`，按 route factor 把连续迭代完整
展开并组成逻辑向量组。

**为什么：** 这是完全展开+向量化的基础形式，适用于每次迭代执行同构加载、算术
或整数归约的循环。

**怎么做：** factor 必须是至少 2 的 2 的幂，且不能超过精确 subject trip；动态
subject 必须带主尾证书。整组被展开后，编译器检查至少一个真实的 grouped artifact，
否则视为物化失败。

**对应代码：** 普通 grouped-load subtype 和 factor kind 在
[`LOGICAL_VECTOR_SUBTYPE_MEANINGS_V1`](../python/triton/l_lite/factor_ontology.py#L141-L158)；
factor 合法性在
[`decide_loop_route_factor_admission_v1`](../python/triton/l_lite/factor_ontology.py#L518-L579)；
一般 logical materializer 及“没有真实 reduction/load group 即失败”的逻辑在
[`materializeLogical`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L8401-L8521)。

### 7.5 子域 D2：Bridge factorized logical group

**是什么：** Bridge 构造 program loop 后，把不同逻辑 program 的相同阶段打包；
可进一步进行 invariant hoisting、tensor-lane fusion 和有证明的 exact split elision。

**为什么：** 跨 program 的同构路径比普通 loop iteration 更容易包含重复的不变量
和相同 lane 操作，向量化可同时消除这些重复。

**怎么做：** Bridge 先建立 program 独立性；logical route 再内联 body、识别跨
program 同构节点、hoist 与当前 program ordinal 无关的值，并合并 tensor lanes。
每个可选动作都有独立布尔绑定和后置条件，不能用 kernel 身份默认打开。

**对应代码：** Bridge logical 的完整实现及 factor、hoisting、lane fusion、split
elision 参数在
[`materializeBridgeLogical`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L7971-L8221)；
物化后的 `logical_group_bridge` / `factorized_bridge_GH/GF/GHF` identity 分支在
[`HBVLoopMaterializePass`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L9124-L9145)，
最终按实际 join 和 reduction 计数验收在
[`HBVValidateLoopPlanPass`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L9467-L9513)。

### 7.6 子域 D3：动态主尾 logical group

**是什么：** 动态 trip 循环的完整 main groups 走向量化，不能组成完整组的 tail
保持原执行方式。

**为什么：** 动态范围不总是 factor 的倍数，强行打包会越界或改变归约结果。

**怎么做：** 使用 Provider 证明的 runtime main-tail partition；main 走 logical
group，tail 走有序标量/原循环路径。若 guard、mask 或 tail identity 不能证明，
该候选拒绝而不是回退。

**对应代码：** 共享的动态 subject 证书门在
[`LoopRouteSubjectV1.runtime`](../python/triton/l_lite/factor_ontology.py#L342-L352)
及
[`decide_loop_route_factor_admission_v1`](../python/triton/l_lite/factor_ontology.py#L554-L579)；
full-unroll route 的 main/tail 后置 identity 在
[`HBVLoopMaterializePass`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L9105-L9146)
和
[`HBVValidateLoopPlanPass`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L9410-L9437)。

### 7.7 子域 D4：exact-prefix 向量化

**是什么：** 对“静态容器宽度 + 动态有效前缀”的整数归约，把有效前缀用 predicate
保护后在静态容器上向量化归约。代码中的 artifact route 名为
`predicated_exact_prefix_reduction`。

**为什么：** 源程序可能没有适合普通 unroll 的显式循环，但其动态前缀递推与
“完整展开后把多个值组成向量再归约”具有相同的顶层因果机制。

**怎么做：** Provider 必须证明：有效长度、2 的幂容器宽度、元素大小、越界元素的
单位元以及精确整数归约语义。其 `route_factor = 1` 是 adapter 的语义哨兵，不表示
宽度为 1；真实物化宽度来自容器。它是向量化域的子域，不是第四条 route，也不
伪造一条 Bridge lineage。

**对应代码：** Provider-closed subject 在
[`LoopExactPrefixSubjectV1`](../python/triton/l_lite/factor_ontology.py#L162-L214)，
factor=1 与 subtype/admission 规则在
[`decide_loop_logical_exact_prefix_admission_v1`](../python/triton/l_lite/factor_ontology.py#L258-L287)；
它以 direct candidate 加入 autotune、同时将顶层 mechanism 映射回 logical route 在
[`add_loop_exact_prefix_autotune_candidate_v1`](../python/triton/l_lite/autotune.py#L272-L333)。
C++ 识别与物化分别在
[`collectExactPrefixReductions`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L1279-L1286)
和
[`materializeExactPrefixReduction`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L1288-L1357)，
最终专属后置条件在
[`HBVValidateLoopPlanPass`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L9341-L9358)。

### 7.8 子域 D5：sibling state-axis SLP

**是什么：** 对连续产生的 N 个同构状态对象，把这些对象沿新增 state axis 打包，
共同执行一段操作图；N 不固定为 4。

**为什么：** 例如多个状态先分别做 row normalization，再共同做 column
normalization。共性不是特定算子名字，而是“多个兄弟值具有同构操作图，并在后续
存在可证明的跨状态组合”。

**怎么做：** Provider 比较每条 sibling lane 的完整操作图、形状、类型和外部使用，
只允许规则白名单内且无不安全副作用的节点；必要时 padding 到可物化宽度。编译器
stack/reshape 这些状态，逐节点打包执行，再拆分输出。最终逐组检查 packed root、
packed node 和 cross-state consumer 数量，不能依赖 `softmax`、`sinkhorn` 或某个
kernel 名准入。

**对应代码：** 通用操作能力谓词从
[`isStateAxisElementwiseCapability`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L1359-L1386)
开始，结构发现由
[`collectStateAxisNormalizations`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L1591-L1627)
完成，实际 stack/reshape/逐节点打包在
[`materializeStateAxisNormalization`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L1829-L1884)；
decision 逐组复核 Provider graph 在
[`HBVLoopDecisionPass`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L6771-L6804)，
最终以精确节点计数闭合在
[`HBVValidateLoopPlanPass`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L9291-L9340)。

### 7.9 子域 D6：嵌套 subject 的 logical group

**是什么：** 一条 logical route 作用于嵌套结构中由 dimension mask 选定的范围。

**为什么：** 嵌套循环的 outer/inner trip 可能相同；仅记录乘积无法说明到底打包了
哪一层，也无法验证最终 artifact。

**怎么做：** 保留完整 trip vector 和 dimension mask。route factor 可以在 2 到
所选范围完整展开上限之间取合法的 2 的幂；factor 表示 logical grouping width，
不是另一个 inner/outer route。物化器仍须证明选中范围能形成真实 logical group。

**对应代码：** 不可丢失的 dimension vector/mask 定义在
[`LoopNestStructuralScopeV1`](../python/triton/l_lite/factor_ontology.py#L665-L724)；
logical factor 的幂次和上限规则在
[`decide_loop_nested_route_factor_admission_v1`](../python/triton/l_lite/factor_ontology.py#L781-L836)；
叶子循环的直接依赖证明在
[`certifyNestedInnerDimension`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L6586-L6623)。

## 8. 联合优化如何成立，而不混合因果域

把 Bridge 和 route 分开不会丢掉联合优化。联合优化由“顺序组合 + 笛卡尔积”表达：

```text
Bridge(b) 先改变 route subject
Route(r, f) 再作用于 Bridge 后的 subject
Autotune 在所有请求的 (b, r, f) 中比较最终 kernel
```

这样既能回答“Bridge 单独造成了什么结构变化”，也能回答“同一个 Bridge subject
交给不同 route 后哪个最终 kernel 最快”。把两者揉成一个因果域反而无法区分收益
来自 program coarsening，还是来自重排/向量化。

**对应代码：** V2 笛卡尔图先调用 Bridge subject 转移、再调用 route factor
admission，实现见
[`build_loop_intervention_cartesian_graph_v2`](../python/triton/l_lite/composition.py#L380-L472)
与
[`certify_loop_bridge_route_composition_v2`](../python/triton/l_lite/contract.py#L173-L241)；
物化后 attestation 强制 `intervention_order == ("bridge", "route")`，见
[`attest_loop_bridge_route_materialization_v1`](../python/triton/l_lite/composition.py#L584-L702)。

对于嵌套循环，当前设计坚持“一条 route + 一个 factor + 一个结构 scope”。scope
说明该 route 穿过哪些维度，factor 说明 route 的局部参数；不会生成相互独立的
outer route 和 inner route。由此候选 identity 可被复查，也避免指数级的无意义
组合。

## 9. L-lite 与 L 主项目的边界

两者应共享：

- IR/Provider 事实；
- 因果域和 factor 的语义本体；
- 合法性规则与类型化拒绝；
- Bridge 与 route 的物化实现；
- artifact identity、正确性和后置条件验证。

两者不共享后端选择方式：

- **L-lite：** 不预测，使用原生 exhaustive autotune 实测；
- **L 主项目：** 使用 HBV 的强/弱语义传播、分域后端统计代理、残差、半径和发布门，
  目标是在保留论文级归因的同时减少 acquisition 开销。

因此，L-lite 不是 L 主项目的简化预测器，而是相同变换能力的实验对照组。

## 10. 当前能力声明应如何理解

代码中存在某个 ontology、Provider 字段或 adapter parser，不自动等于所有真实
kernel 都已经闭环。对一个具体子域，应分别检查：

1. **规则闭合：** 准入只依赖可解释事实，没有 kernel/算子定制；
2. **合法性闭合：** 正例被接纳，反例在最早责任层类型化拒绝；
3. **物化闭合：** 目标 IR 与最终 artifact identity 可观察，回退为零；
4. **正确性闭合：** Original 与候选输出在规定容差内一致；
5. **覆盖闭合：** 独立新家族上仍能按相同规则工作。

公开快照已经包含上述通用架构、Bridge 基本编译测试、factor/组合本体和 L-lite
控制层离线测试。性能测量在能力面闭合和环境满足要求前保持冻结。因而本手册描述
的是当前代码支持的机制和证书责任，不把尚未完成的全量 benchmark 结果包装成已
证明的普遍覆盖率。

**对应代码：** 最终 C++ validator 会拒绝残留的临时 lineage、route/subject/artifact
不对应、Bridge 与 route factor 丢失，以及“pass 返回成功但目标 artifact 不存在”，
总入口是
[`HBVValidateLoopPlanPass`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L9156-L9525)。
原生 autotune 控制器不安装 prune callback 或 perf model，见
[`LoopNativeAutotuneControlV1`](../python/triton/l_lite/autotune.py#L487-L592)
和
[`build_loop_native_autotune_control_v1`](../python/triton/l_lite/autotune.py#L594-L616)。

## 11. 代码导航

| 目标 | 位置 |
|---|---|
| factor、subject、嵌套 scope 和 exact-prefix 子型语义 | `python/triton/l_lite/factor_ontology.py` |
| Bridge → route 组合合法性 | `python/triton/l_lite/contract.py` |
| 笛卡尔积和两次干预的独立 lineage | `python/triton/l_lite/composition.py` |
| exhaustive autotune domain、失败候选和原生缓存 | `python/triton/l_lite/autotune.py` |
| Bridge discovery/materialization、route materialization、后置验证 | `lib/Dialect/Triton/Transforms/HBVLoop.cpp` |
| 通用 loop unroll 支撑 | `lib/Dialect/Triton/Transforms/LoopUnroll.cpp` |
| pass 注册与 pipeline 名称 | `include/triton/Dialect/Triton/Transforms/Passes.td` |
| Bridge 编译级测试 | `test/Triton/hbv-loop-bridge-*.mlir` |
| L-lite 控制层测试 | `python/test/unit/l_lite/test_control.py` |

## 12. 最短阅读路径

如果只想快速理解实现，建议按以下顺序：

1. 先读本手册第 1、2、8 节，理解 Bridge 与 route 为什么必须分开；
2. 再读第 4 节，理解循环 subject 是如何产生的；
3. 按兴趣阅读第 5、6、7 节中的某条 route；
4. 查看 `factor_ontology.py`，确认 factor 在不同 route 中为什么不能混用；
5. 查看 `HBVLoop.cpp` 的最终 postcondition，理解“真的物化成功”比“pass 没报错”
   多了哪些要求；
6. 最后查看 `autotune.py`，理解 L-lite 为什么不使用 L 主项目的性能预测和剪枝。
