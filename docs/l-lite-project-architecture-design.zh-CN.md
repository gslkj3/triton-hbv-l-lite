# L-lite 项目架构设计书（通俗评审版）

> 评审日期：2026-08-26
>
> 适合读者：第一次接触 Triton、GPU 编译器或本项目的项目经理、评审专家和工程师
>
> L-lite 实现快照：`25bca1f4ec7404878ba4b52e19d4a5919a1e41ce`
>
> 原生 Triton 基线：`7c56a5e40f7fd928dfd5c72902d5def0097db73a`（Triton 3.6.0）

本文刻意不用“只有编译器工程师才看得懂”的写法。正文先讲清楚项目要解决什么问题、
三个优化选择分别做了什么、为什么需要合法性与物化证明，以及项目目前真正完成到哪
一步。工程师需要的文件、类、函数和证书细节仍然保留在后半部分。

如果评审时间有限，只读第 1、2、3、8、12、16、17 节，就能理解本项目的主要设计、
当前边界和下一步工作。

---

## 1. 先用一分钟理解 L-lite

### 1.1 它解决什么问题

一个 GPU 程序通常会把大量相似工作交给很多 GPU program 并行执行，程序内部也可能
存在循环。编译器会对这些工作做优化，但每一种局部优化通常只回答：

> “我眼前这个循环能不能使用这一种优化？”

L-lite 想多回答一步：

> “从整个 kernel 看，先怎样构造循环，再对这个循环使用哪一种优化、使用多大的
> factor，最后得到的完整方案最好？”

这里的 kernel 可以先理解为“一项交给 GPU 执行的完整任务”，factor 可以理解为
“一次合并、展开或处理多少份相似工作”。

### 1.2 L-lite 做了什么

L-lite 把选择拆成前后两个独立步骤：

1. **Bridge：是否把若干原本分散在不同 GPU program 中的相似工作，收进一个显式
   循环。**
2. **Route：对 Bridge 之后真实存在的循环，选择一种优化方式。**

Route 有三条：

1. 使用 Triton 原生的软件流水；
2. 完全展开后重排；
3. 完全展开后向量化。

L-lite 枚举这些选择的组合，为每个组合真实编译出候选，再交给 Triton 原生 autotune
逐一实测并选择最快者。

用一句不严谨但容易理解的话概括：

> L-lite 像一个“把所有合规施工方案都真正造出来，再逐个试跑”的对照系统。

主 L 项目则是在相同施工方案上，用 HBV 预测体系尽量少试跑、提前选出安全且可能更快
的方案。两者的优化能力应当相同，区别只在“怎么选”。

### 1.3 当前最重要的真实结论

| 问题 | 当前答案 |
|---|---|
| Bridge、重排和向量化是不是只写在测试里的手工特例？ | 不是。它们是注册到 Triton 的真实 C++/MLIR Pass。 |
| 准入是否根据 kernel 名、算子名或测试集名单判断？ | active 规则不读取这些名称，而是看 IR 结构、依赖、地址和副作用。 |
| 软件流水是不是 L-lite 重新写了一套？ | 不是。L-lite 负责把它放入候选空间、绑定对象和参数；最终流水化由 Triton 原生后端完成。 |
| 所有候选是否只要“合法”就一定生成了目标代码？ | 不是。所以项目另有物化和后置验证。 |
| 当前公开仓是否已经能从普通 `@triton.jit` 一键跑完整 L-lite？ | 还不能。候选到 PlanBundle、编译 binding、正确性验证之间的公开产品接线仍缺失。 |
| 当前公开证据能否证明性能收益？ | 不能。公开测试证明的是部分编译能力和控制逻辑，不是完整 GPU 性能结论。 |

这不是把已实现内容说小，而是把“编译器能力已经存在”和“完整产品已经闭环”严格分开。

---

## 2. 用一个日常例子理解 Bridge 和三条 Route

假设厨房要连续制作四份同样的套餐。每份套餐包含：取食材、加工、装盘。

原来的安排可能是四名厨师各做一份：

```text
厨师 0：取第 0 份食材 → 加工第 0 份 → 装盘第 0 份
厨师 1：取第 1 份食材 → 加工第 1 份 → 装盘第 1 份
厨师 2：取第 2 份食材 → 加工第 2 份 → 装盘第 2 份
厨师 3：取第 3 份食材 → 加工第 3 份 → 装盘第 3 份
```

GPU 上的 program 并不等于真实厨师，但这个比喻足以帮助理解结构变化。

### 2.1 Bridge：先把分散工作变成循环

当 `bridge_factor=4` 时，Bridge 尝试把四份逻辑工作交给一个物理 program，通过显式
循环逐份执行：

```text
一个物理 program：
  for lane in [0, 1, 2, 3]:
    logical_program = physical_program * 4 + lane
    执行 logical_program 对应的工作
```

Bridge 的目的不是直接决定“取食材、加工和装盘怎样重排”，而是把原来藏在 program
维度上的重复结构显式暴露为循环，让后续 route 能看见并处理。

Bridge 只在能证明不同 logical program 互不破坏数据时才允许这么做。例如两份工作
写入同一块输出、存在未证明的读写覆盖，或者包含不可安全复制的副作用，就要拒绝。

`bridge_factor=1` 表示不合并 program，也不新造循环。它是 Bridge 的 identity 选择。

### 2.2 Route 1：软件流水

软件流水像这样的厨房安排：

```text
正在加工第 0 份时，同时开始取第 1 份食材；
正在装盘第 0 份时，同时加工第 1 份，并开始取第 2 份食材。
```

它的核心是让不同循环迭代的“取数据”和“计算”发生重叠，减少等待。

它不是把循环消掉。相反，软件流水通常需要一个仍然存在、后端能够识别的循环，还要
有适合跨迭代提前搬运的数据服务。Bridge 能构造循环，不代表这个新循环天然适合软件
流水；如果循环体根本没有后端可流水化的加载阶段，软件流水候选应被明确拒绝。

L-lite 只负责选择 subject 和 stage 数、保存证据并检查请求没有丢失。真正的异步搬运
与调度由 Triton 原生 GPU 后端完成。

### 2.3 Route 2：完全展开 + 重排

先把循环完整写开：

```text
取第 0 份 → 加工第 0 份 → 装盘第 0 份
取第 1 份 → 加工第 1 份 → 装盘第 1 份
取第 2 份 → 加工第 2 份 → 装盘第 2 份
取第 3 份 → 加工第 3 份 → 装盘第 3 份
```

再在依赖允许时，把相同阶段排到一起：

```text
取第 0/1/2/3 份
加工第 0/1/2/3 份
装盘第 0/1/2/3 份
```

这就是“完全展开 + 重排”的直觉。

需要特别强调：真实代码不必刚好包含“一个 load、一个 compute、一个 store”。如果循环
只有 store、只有计算，或者操作种类更多，重排仍应按真实操作及其依赖进行。没有 load
只表示不存在 load 阶段，不应该因此被拒绝。项目使用 operation-neutral 的规则，就是
为了避免把某种固定操作配方写成隐性特例。

### 2.4 Route 3：完全展开 + 向量化

仍然先把循环写开，然后把结构相同、彼此独立的操作合成更宽的一次操作。

简化代码如下：

```python
for i in range(4):
    y[i] = x[i] * 2
```

可以想象成：

```python
y[0:4] = x[0:4] * 2
```

真实 Triton IR 的向量化远比这个例子严格。它必须证明被合并操作的类型、属性、操作数
关系、mask、地址和副作用相互对应，不能只因为源码“长得像”就合并。

项目还支持一种 state-axis 情况。例如 `r0、r1、r2、r3` 经历结构完全相同的操作图，
可以沿“状态轴”打包。这里不是固定只支持四路：一路、两路、五路或 n 路都应由实际
结构和可证明关系决定，而不是读取变量名或某个算子组合。

### 2.5 为什么 Bridge 与 Route 必须分开

Bridge 回答的是：

> “原本分散的工作，能不能安全地形成这个循环？”

Route 回答的是：

> “对已经存在的这个循环，采用哪种优化？”

如果把两者揉成一个巨大 Pass，最后变快或变慢时就说不清：到底是 program 合并带来的，
还是重排、向量化或软件流水带来的。分开后仍可以联合优化，因为候选空间会枚举二者的
笛卡尔积。

例如：

```text
bridge_factor ∈ {1, 2, 4, 8}
route ∈ {软件流水, 完全展开+重排, 完全展开+向量化}
route_factor ∈ 各 route 自己允许的集合
```

每个 `bridge_factor × route × route_factor` 都是一个独立候选。分开归因不等于分开
选优。

---

## 3. factor 到底是什么

“factor”在三种机制中不是同一个意思。这是评审时最容易混淆的地方。

| 参数 | 通俗含义 | 例子 |
|---|---|---|
| `bridge_factor` | 一个物理 program 接管多少个 logical program | 4 表示将四份逻辑工作构造成四次循环迭代 |
| 软件流水 factor | 流水 stage 数 | 3 表示最多让三阶段工作交叠 |
| 重排 factor | 一次展开并参与重排的循环组宽 | 4 表示四次迭代为一组 |
| 向量化 factor | 一次尝试打包的逻辑 lane 数 | 4 表示把四份同构操作组成一组 |

因此不能把三个 factor 存成一个含义模糊的整数，也不能因为两个数字碰巧相等，就说它们
代表同一件事。

### 3.1 factor 不整除循环次数怎么办

假设循环执行 10 次，factor 是 4：

```text
主组：4 + 4
尾部：2
```

前 8 次可以按两个完整组处理，最后 2 次作为 tail 保留。项目必须记录哪些操作属于
main，哪些属于 tail，不能在原生展开后丢掉身份。

### 3.2 动态循环怎么办

有些循环次数只有运行时才知道。项目不能假装它是静态常数，而要提供可检查的 main-tail
证书，或把动态有效范围放入静态容器并用 predicate 屏蔽无效 lane。

`exact-prefix` 就是完全展开+向量化中的一种动态子型：它不是第四条 route。它处理的是
“前 N 个元素有效，但 N 运行时才知道”的结构。

### 3.3 嵌套循环怎么办

如果 IR 原本有内层循环，Bridge 又在外面构造一层循环，就会出现内外嵌套。

本项目不为内层和外层各选一条互相独立的 route。一个候选仍只有：

```text
一条 route + 一个 route factor + 一个结构范围
```

结构范围说明这次变换覆盖外层、内层，还是二者合起来。对于完全展开，允许上限可能是：

```text
1
outer_extent
inner_extent
outer_extent × inner_extent
```

这样既能表达内外层全展开，也避免候选空间无控制地膨胀为
`outer_route × inner_route`。

相关 Python 定义在
[`factor_ontology.py`](../python/triton/l_lite/factor_ontology.py)：

- [`LoopRouteSubjectV1`](../python/triton/l_lite/factor_ontology.py#L291-L352) 描述循环对象；
- [`route_subject_after_bridge_v1`](../python/triton/l_lite/factor_ontology.py#L414-L428) 只描述 Bridge 后对象怎样变化；
- [`decide_loop_route_factor_admission_v1`](../python/triton/l_lite/factor_ontology.py#L493-L586) 按 route 解释 factor；
- [`LoopNestedRouteSubjectV1`](../python/triton/l_lite/factor_ontology.py#L619-L664) 描述嵌套对象；
- [`LoopNestStructuralScopeV1`](../python/triton/l_lite/factor_ontology.py#L665-L725) 描述结构范围。

---

## 4. “能看见、合法、做出来、做对、变快”是五件不同的事

可以继续用建筑来比喻：

1. **能看见：** 找到了可以改造的房间；
2. **合法：** 图纸证明这堵墙理论上可以改；
3. **物化：** 施工队真的按图纸改了墙；
4. **正确：** 改完房屋仍安全、功能没有变化；
5. **变快：** 入住后实际通行效率更高。

对应到 L-lite：

| 层次 | 回答的问题 | 主要责任方 |
|---|---|---|
| 可观察性 | IR 中有没有原生循环，或者 Bridge 能不能构造循环？ | Discover / Facts |
| 合法性 | 依赖、地址、副作用、factor 和对象关系是否允许这项变换？ | ontology / contract / Decision |
| 物化 | 编译器是否真的生成目标结构，而不是悄悄回到 Original？ | Bridge / Unroll / Materialize / Validate |
| 正确性 | 候选运行结果是否与 Original 和独立 oracle 一致？ | correctness harness |
| 性能 | 正确候选中谁更快？ | Triton 原生 autotune |

### 4.1 为什么不能把非法候选直接交给 autotune

原生 autotune 的主要责任是计时和选最快者，它不会替新增 Pass 证明程序等价。一个错误
候选可能运行出错误结果，甚至因为少做了工作而“错误地获胜”。

因此，Pass 必须先承担自身的变换等价性与物化责任。不能安全执行的候选仍保留在完整
请求账本中，记录它的编译尝试时间和类型化失败原因，但不能真的在 GPU 上运行错误代码。

### 4.2 为什么不能提前删除慢候选

L-lite 是 exhaustive autotune 对照，不能使用主 L 的性能预测提前删掉“可能慢”的候选。
正确流程是：

1. 先固定完整请求网格；
2. 每个请求都经历同样的候选构建尝试；
3. 合法、物化、正确的候选进入真实计时；
4. 静态拒绝和编译失败仍保留 identity、原因和耗时；
5. 总开销包含失败候选成本。

这既不让错误代码上 GPU，也不替 autotune 隐藏本应承担的探索成本。

---

## 5. 为什么需要六个 TTIR Pass

六个新增 Pass 不是六种优化。它们像同一条生产线上的六个工位，各自只负责一个问题。
其中有些工位调查或验收，有些工位真实施工；“质检工位”只是一种帮助理解责任分离的比喻，
并不是说六个 Pass 都只读。

```text
发现 Bridge 是否可构造
  → 构造或保留 Bridge subject
  → 对 Bridge 后真实 IR 提取完整循环事实
  → 根据 Plan 做出闭合决定
  → 配合原生 LoopUnroll 物化 route
  → 最终检查实际结果
```

当前建议执行顺序是：

```text
triton-loop-bridge-discover
triton-loop-bridge-program-coarsening
triton-hbv-loop-facts
triton-hbv-loop-decision
triton-loop-unroll
triton-hbv-loop-materialize
triton-hbv-validate-loop-plan
```

其中 `triton-loop-unroll` 是 Triton 原生 Pass，L-lite 只做了最小扩展；另外六个是新增
Pass。之所以看到七段，是因为“六个新增 + 一个原生展开”。

### 5.1 `TritonLoopBridgeDiscover`：先调查能不能构造 Bridge

它像现场勘察员，主要检查：

- program-id 怎样进入地址计算；
- 不同 logical program 的写入是否重叠；
- 读写之间是否存在未证明 alias；
- 有没有 atomic、volatile、未知副作用或无法处理的 call；
- runtime scalar 与 grid 信息是否闭合；
- 提前 return 能否安全正规化。

它只输出事实，不决定哪个候选更快。入口是
[`LoopBridgeDiscoverPass`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L4993)，核心 program
独立性证明位于
[`certifyBridgeProgramIndependence`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L2152-L2270)。

### 5.2 `TritonLoopBridgeProgramCoarsening`：真正构造 Bridge

它读取已证明的 Bridge factor，把若干 logical program 放进一个显式 `scf.for`，并在
循环里恢复原本的 logical program-id。

它还负责多轴 mixed-radix 坐标、可证明纯 helper 的处理和特定提前 return 的谓词化。
入口是
[`LoopBridgeProgramCoarseningPass`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L6191)。

### 5.3 `TritonHBVLoopFacts`：给循环建立事实档案

它像检测报告，记录每个可见循环的：

- 边界、步长、静态或动态 trip count；
- carried value、load/store/reduce 和内存体积；
- 原生软件流水能力及拒绝原因；
- 重排、向量化、嵌套和 main-tail 能力；
- Bridge 来源、地址关系和结构证书；
- exact-prefix、state-axis 等结构事实。

它也不做性能判断。入口是
[`HBVLoopFactsPass`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L5115)。

### 5.4 `TritonHBVLoopDecision`：把外部计划与真实事实对上

Python 控制层会为候选生成 PlanBundle。Decision 检查：

- 计划指定的 subject 是否就是事实层看到的对象；
- route 和 factor 是否有正确含义；
- Bridge→route 的顺序和证书是否闭合；
- 所需能力是否真的存在；
- 是否出现 unknown、ambiguous 或证据不一致。

任何一项不闭合都拒绝，不能猜。入口是
[`HBVLoopDecisionPass`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L7210)。

### 5.5 原生 `TritonLoopUnroll` 的最小扩展：给复制品贴标签

原生展开器会复制循环体，但原本不会保留 L-lite 需要的“哪份操作来自第几组、属于 main
还是 tail、服务哪个 subject”等 lineage。

这像复印四份文件后，如果不在每页写来源，后面的重排员就不知道哪页属于哪一份。L-lite
没有重写展开算法，而是在原生 Pass 上补充来源标签和 main/tail 归属。

实现位于
[`LoopUnroll.cpp`](../lib/Dialect/Triton/Transforms/LoopUnroll.cpp#L24-L195)。

### 5.6 `TritonHBVLoopMaterialize`：真正实施 route

它根据 route 分派到不同物化器：

- 软件流水：确认目标 loop 和 stage 请求完整保留，物理流水化留给原生后端；
- 重排：根据展开 lineage 和依赖闭包做 operation-neutral phase 重排；
- 向量化：精确打包 load、store 或一般同构 elementwise group；
- 动态/嵌套/state-axis：按已注册证书和子型处理。

失败会设置明确的 materialization failure，并让编译失败，不会静默回到 Original。
入口是
[`HBVLoopMaterializePass`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L10108)。

### 5.7 `TritonHBVValidateLoopPlan`：施工后验收

它不相信“前面的 Pass 返回成功”就等于真的完成，而是检查终态：

- PlanBundle 是否被完整消费；
- 临时标签是否清理；
- route、subtype、factor、subject identity 是否仍一致；
- Bridge 与 route 两次干预是否都保留 lineage；
- full-unroll 后目标循环是否按预期消失或只保留有解释的 tail；
- 软件流水的 live subject/stage 请求是否仍完整；
- 每条 route 的目标 postcondition 是否成立。

入口是
[`HBVValidateLoopPlanPass`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L10496)。

需要注意：这个 validator 检查的是 TTIR 终态。最终 PTX、cubin 或 SASS 是否出现预期
机器码事实，还需要后端 artifact harness 继续验证。

---

## 6. 三个工程部件为什么都需要

评审材料中常把实现概括成三个部件：

1. 六个 TTIR Pass；
2. 对原生 LoopUnroll 的最小扩展；
3. Python 候选控制层。

三者的关系可以类比为：

```text
Python 控制层：列出所有施工方案，并给每个方案唯一编号
TTIR Pass：检查现场、按方案施工、完工验收
LoopUnroll 扩展：施工中复制材料时保留每份材料的来源标签
```

少一个会发生什么：

| 缺少部分 | 后果 |
|---|---|
| 没有 Python 控制层 | 不能稳定表达完整候选笛卡尔积，也不能保证每个候选有唯一 identity |
| 没有编译器 Pass | 只剩 Python 枚举，候选并未真实改变 TTIR |
| 没有 LoopUnroll lineage | 展开后的操作失去来源，后续重排、向量化和验证难以可靠归属 |
| 没有最终 validator | Pass 可能部分失败或回退，却仍被当作成功候选送去计时 |

“这三个职责都必要”不等于“必须永远保持当前三个文件布局”。例如 10,922 行的
`HBVLoop.cpp` 后续完全可以拆分；必要的是责任和证据链，未必是当前物理文件大小。

---

## 7. Python 控制层在做什么

### 7.1 `factor_ontology.py`：先把名词定义清楚

文件：[`factor_ontology.py`](../python/triton/l_lite/factor_ontology.py)

它规定 Bridge factor、stage、重排组宽、向量组宽、subject、动态子型和嵌套 scope 的
不可变含义。它不预测性能，也不读取计时结果。

这一步像先统一“米、平方米、公斤”的单位，避免不同模块都拿一个整数，却各自用不同
含义解释。

它的输入不是一个已经测过性能的 kernel，而是一组结构化事实：subject 是原生循环还是
Bridge 循环、trip 是静态还是动态、选了哪条 route、factor 是多少、嵌套范围是什么。
它的输出也是结构化对象，而不是一个“可/不可”的布尔值。例如 route factor admission
会同时给出 main extent、tail extent、main group 数、是否完整消除 source loop，以及
拒绝时的类型化原因。

为什么需要这么详细的输出？假设 trip=10、factor=4。简单返回“合法”会让后续模块不知
道还剩两次 tail，也无法检查这两次是否真的保留。返回完整结构后，Decision、Materialize
和 Validate 才能使用同一份定义，避免每个模块重新计算并产生不同答案。

所有对象还通过规范化 JSON 生成稳定引用。所谓稳定引用，就是只要语义字段相同，候选
身份就相同；字段变化，引用也变化。它防止用 Python 列表下标或临时内存地址当 identity，
因为这些值在另一次运行中可能改变。

### 7.2 `contract.py`：判断两次干预能否组合

文件：[`contract.py`](../python/triton/l_lite/contract.py)

它先确定 Bridge 之后到底有什么 subject，再问所选 route/factor 能否作用在该 subject
上。V1 处理较简单的组合，V2 支持更完整的动态、嵌套和证书语义。

核心函数包括
[`certify_loop_bridge_route_composition_v1`](../python/triton/l_lite/contract.py#L79-L125)
及其 V2 版本。

Contract 的输入是“Bridge 请求 + Bridge 前对象事实 + route 请求”，处理顺序固定为：

1. 先应用 Bridge 的结构语义，得到 Bridge 后 subject；
2. 再在这个确切 subject 上解释 route factor；
3. 检查两次干预是否拥有各自所需的证书；
4. 输出组合合法性、Bridge/route 各自的 factor identity 和类型化拒绝原因。

它不能跳过第一步，直接拿原 IR 的 trip 去判断 Bridge 后循环；也不能因为 route factor
恰好等于 Bridge factor，就把二者合成一个参数。

例如 Bridge factor 4 构造出 trip=4 的循环，当前完整展开 materializer 可能要求 route
factor 也等于 4。这只是“当前能力要求完整覆盖 Bridge loop”，不是说两个数字本质相同。
未来 materializer 支持 factor 2 的 main/tail 后，contract 可以放宽能力限制，而不需要
修改 factor 的语义定义。

### 7.3 `composition.py`：建立完整候选表和不可变 identity

文件：[`composition.py`](../python/triton/l_lite/composition.py)

它枚举请求网格，给每个 cell 保存：

- Bridge factor；
- route 与 route factor；
- subject；
- 合法或类型化拒绝；
- 两次干预的顺序 lineage；
- 物化 attestation。

V2 图入口是
[`build_loop_intervention_cartesian_graph_v2`](../python/triton/l_lite/composition.py#L380-L472)，
物化证明入口是
[`attest_loop_bridge_route_materialization_v1`](../python/triton/l_lite/composition.py#L584-L702)。

这里的“完整候选表”包括 Original，也包括最终被拒绝的请求单元。假设有 4 个 Bridge
factor、3 条 route，每条 route 各有 4 个 factor，表面上有 48 个组合；composition 不会
先根据经验只保留看起来快的 10 个，而是给 48 个请求各建一行。

每一行随后可能变成：

- `requested`：已经请求，尚未判断；
- `typed rejected`：某项结构规则明确不成立；
- `compile failed`：理论上可请求，但实际编译失败；
- `materialization failed`：编译经过部分步骤，却没有产生目标结构；
- `correctness failed`：产物可执行但结果不等价；
- `executable`：可安全交给 autotune。

物化 attestation 不是再做一次性能预测。它对照真实 artifact 报告，逐项检查 Bridge 是否
按指定 factor 发生、route 是否按指定 factor 发生、顺序是否为 Bridge 后 Route，以及
两次干预 lineage 是否完整。如果候选请求 vector factor 4，结果 artifact 只含 Original，
即使编译没有报错，也必须判为 silent fallback。

### 7.4 `autotune.py`：把可执行候选交给原生 autotune

文件：[`autotune.py`](../python/triton/l_lite/autotune.py)

L-lite 不修改 Triton 的“逐候选计时并选择最小值”算法，也不安装 performance model、
early stop 或预测剪枝。它只把 candidate ref 绑定到预构建 kernel，再让原生 Autotuner
工作。

当前 facade 是
[`LoopNativeAutotuneControlV1`](../python/triton/l_lite/autotune.py#L487-L592)。

它接收的是 `candidate_ref → KernelInterface binding` 映射。binding 可以理解为“已经编译
好、可以按统一接口启动的一份候选 kernel”。facade 把 candidate ref 放入原生
`triton.Config`，在运行时由 mux 分派到正确 binding。

它保留原生 autotune 的两项核心行为：逐配置 benchmark，以及按最小测量时间选择 winner。
它没有安装性能模型，也没有因为某候选预计慢而提前停止。因此 L-lite 的选择结果仍是
exhaustive measurement 对照，而不是 HBV 预测结果。

facade 还必须把逐候选 timing、总 benchmark 时间、winner identity 和 key cache 行为
公开出来。仅报告 winner 的最终时间是不够的，因为项目还要与主 L 比较 acquisition：
两者可能找到相近 winner，但 L-lite 为此测了全部候选，主 L 只做少量测量。

### 7.5 PlanBundle：Python 决定怎样无歧义地交给 C++

PlanBundle 是候选控制层与编译器 Pass 之间的合同。Python 不能只传一句“帮我向量化”，
因为同一模块可能有多个循环、多个 Bridge origin 和多个 factor。PlanBundle 至少要闭合：

- schema 版本和项目类型；
- candidate、route、mechanism route 和 subtype；
- subject locator 与 subject ref；
- Bridge factor、route factor 和多轴 divisor；
- dynamic binding、guard、feedback 和 fallback 语义；
- compiler commit 等最小 provenance。

C++ parser 对未知字段和字段数采用 fail-closed：不能识别的新旧 schema 不会被“尽量
猜测”后继续编译。这种严格看起来不方便，但能防止 Python 以新含义写入字段，旧 C++ 却
用旧含义解释，最终生成一个表面成功、实际身份错误的候选。

历史 adapter 可以保持 evidence-readable，也就是旧实验账本仍能被读取；但若其行为已经
被新证据否决，就必须标为 not production-executable，不能因为 parser 还能读就重新进入
active 候选空间。

### 7.6 candidate identity：为什么不能只用 `route=vector,factor=4`

两个候选即使 route 和 factor 相同，也可能作用于不同 subject、不同 Bridge factor、不同
动态 binding 或不同 compiler commit。如果只用二元组命名，缓存可能把一个候选的结果
错误复用给另一个候选。

完整 identity 至少由这些部分决定：

```text
原 kernel specialization
+ target/compiler provenance
+ Bridge 干预及 factor/divisor
+ route、subtype 与 route factor
+ subject locator/ref
+ 必要的 runtime binding schema
```

identity 的用途不仅是缓存。Plan、TTIR、PTX、cubin、correctness 和 timing 报告都必须能
回到同一个 candidate identity，才可能证明“测量的就是计划中那个候选”。

---

## 8. 一个候选从出生到被 autotune 测量的完整过程

以候选 `Bridge(4) → 完全展开+向量化(4)` 为例：

### 第一步：发现

编译器确认原始 kernel 的 program-id 和地址关系，证明每四个 logical program 能安全
合并。

具体输入是 specialization 后的 TTIR、runtime scalar binding、原始 grid extent 和目标信息。
Discover 沿每个 load/store pointer 追踪根对象、program-id 系数和局部偏移范围，并递归检查
region effect。输出是一份只读 discovery certificate；它不会在此时构造循环，也不根据性能
选择 factor。

如果写区间重叠、共享 root 的读写 alias 未证、出现 atomic/volatile/未知 call，候选在这一
最早层得到稳定拒绝码。后面的模型或 autotune 不再有权把它“试试看”。

### 第二步：建立 subject

Bridge 构造四次迭代的显式循环，并保存“这个循环来自 Bridge factor 4”的来源。

ProgramCoarsening 根据证书把 physical grid 与 logical grid 联系起来，在每个 lane 恢复原
program-id，并对最后不满四份的分组施加边界 guard。多轴时还恢复 mixed-radix 坐标；存在
允许的提前 return 时，先变为 lane-local predicate。

输出不只是一个 `scf.for`，还包含 Bridge origin、factor、subject ref、ordinal、axis divisor
和 dependence lineage。若循环出现但这些身份缺失，后续 route 不应猜它就是目标 subject。

### 第三步：事实提取

Facts 检查循环的 trip、carried value、操作图、mask、地址、tail 和向量化能力。

这里重新观察的是 Bridge 后真实 IR，而不是复用 Bridge 前的旧假设。它还分别计算 native
pipeline、phase reorder、logical vector、runtime main-tail、nested 和 state-axis 能力。每个
能力都有成功字段或类型化失败原因。

例如 Bridge 的确造出 trip=4 loop，但 body 没有可流水化 load service；Facts 可以同时输出：
软件流水不具备能力、operation-neutral 重排仍可能具备能力。这样拒绝保持 route-local。

### 第四步：决策闭合

Decision 把 PlanBundle 中的 `route=vector, factor=4` 与这个确切 subject 对齐。对象或
证书不一致就拒绝。

它还检查 schema/adapter 版本、mechanism route 与 subtype、Bridge→route 顺序、factor
含义、dynamic binding 和 Provider certificate。所有需要的事实必须唯一：找不到是 absent，
找到多个是 ambiguous，两者都不能由“选择第一个”解决。

Decision 的输出是编译器内部明确 route plan 和将由下游消费的属性。它只宣布“根据事实，
这项计划允许进入物化”，不宣布物化已经发生。

### 第五步：展开

原生 LoopUnroll 将四次迭代复制开，并由 L-lite 扩展保留每份操作的 lineage。

如果 trip 与 factor 相等，source loop 可完全消失；如果不整除，则创建 factor-wide main 和
ordered tail。L-lite 标签说明每个 clone 来自哪个 source operation、哪一个 main group、
哪一条 lane，以及 tail 属于什么 partition。

这些标签是临时施工标记。Materialize 使用后，Validate 要求不再需要的临时 role 已清理，
同时保留必要 terminal lineage。标签永久残留同样是失败，因为可能污染后续原生 lowering。

### 第六步：向量化物化

Materialize 检查四组操作是否同构、独立且类型对应，然后将其精确打包。不能打包就明确
失败。

对 load，它比较 pointer、mask、other 和 cache 语义；对 store，它还检查写入顺序与地址
冲突；对一般 operation，它比较名称、类型、属性和 operand graph。若是 state-axis，还要
识别 lane-local 与 cross-state edge。任何一个环节不闭合，都写出 materialization failure，
而不是只打包其中两路后声称 factor 4 成功。

输出要能观察到 factor-wide logical group、vector container 或相应 terminal postcondition，
并继续保留它来自 Bridge factor 4 的组合 identity。

### 第七步：后置验证

Validate 确认目标循环按要求处理、vector group 真的出现、Bridge 和 route factor 没有
丢、也没有回退成 Original。

它还检查 PlanBundle 已消费、临时 role 清理、route/subtype/artifact route 对应、dependence
certificate 未被后续改写破坏，以及 tail 或 surviving loop 都有解释。任何不一致让候选编译
失败，因此不能进入 binding 映射。

这一层的“artifact”仍主要是 TTIR route terminal。它证明中间编译计划完成，但不能越权
宣称最终机器码一定出现特定 GPU 指令。

### 第八步：artifact 与正确性

完整产品还应检查 PTX/cubin identity，并将候选输出与 Original 和独立 oracle 对比。

ArtifactChecker 先确认 candidate 与 Original 的最终产物没有意外相同，并检查 route 必要的
后端事实。CorrectnessQualifier 再使用相同输入分别执行 Original、candidate 和 reference，
按预注册数值合同比较所有输出及未写区域。

若 artifact 与 Original 相同，属于回退或后端消除；若输出错误，属于 correctness failure；
若数值语义允许不同运算顺序但证据不足以决定容差，标为 ambiguous。三类都不能进入正式
autotune，但原因和已支付开销必须保留。

### 第九步：性能

只有以上步骤全部通过，才交给原生 autotune 真实计时。最终报告同时保留这个候选的
编译、验证和计时开销。

autotune 在同一 specialization/key 上多次运行可执行 bindings，采用原生计时和 winner
规则。环境无效时整组 measurement disposition 作废，不选择一个“看起来最好”的残缺样本。

最终一行候选报告应能从 timing 反查 candidate ref、Plan hash、artifact hash、correctness、
GPU UUID 和环境角色。只有这条 identity 链闭合，才能说“这个时间属于这个变换方案”。

这条流程解释了为什么“直接写两个 Pass，然后把参数列表扔给 autotune”仍然不够：
autotune 只负责最后一步，不替前八步背书。

---

## 9. 合法性规则是不是按 workload 定制的

### 9.1 active 规则不允许读取什么

当前 active 准入不应读取：

- kernel 或 Python 函数名；
- 算子类别名；
- benchmark、仓库或测试集身份；
- “这个 shape 过去赢过”的样本 identity；
- 温度、功率、时钟或 utilization；
- 历史 winner 和真实性能标签。

### 9.2 active 规则允许读取什么

它允许读取与变换正确性直接有关的事实：

- loop 的边界、步长、trip、nest 和 carried value；
- SSA 依赖和 operation effect；
- load/store 的 pointer root、mask、alignment 与仿射 footprint；
- program-id axis、grid extent 和 mixed-radix mapping；
- dtype、tensor width 和显式资源预算；
- route factor 与 Provider 证书。

### 9.3 “规则式”不等于“规则很短”

判断一段复杂 IR 是否能安全变换，规则可能很长。关键不是代码行少，而是每个条件都能
回答：

1. 它保护了哪项语义或硬件约束；
2. 它是否适用于所有满足条件的结构；
3. 它是否偷看了 workload identity 或性能结果。

例如“必须有一个 load 和一个 store”不是重排的通用必要条件；它会形成隐性配方定制。
相反，“所有被交换顺序的 operation 之间没有未解决的 SSA 或 memory-effect 依赖”是
通用规则。

### 9.4 当前仍需继续审计的风险

公开源码中保留了多代 adapter 和少量窄结构 matcher。active authority 可以排除旧路径，
但源文件尚未物理删除所有历史实现。因此评审时应区分：

- 当前生产/实验 authority 使用什么规则；
- 源码中还存在哪些只为历史证据可读而保留的路径。

后续宜把 active registry 与 historical evidence reader 物理分层，并把窄历史命名改为
结构语义命名。

### 9.5 Bridge 的规则式子域逐项说明

“规则式子域”不是“按算子分组”。它的意思是：同一种 Bridge 原理遇到不同 IR 结构时，
需要不同但可复核的证明办法。每个子域都必须回答五个问题：看到了什么、为什么安全、
怎样改写、怎样验证、何时拒绝。

#### 子域一：identity，也就是 Bridge factor 等于 1

**看到了什么：** 用户选择 `bridge_factor=1`。

**它是什么意思：** 一个物理 program 仍只承担一个 logical program，不合并 program，
也不因为 Bridge 新建循环。它相当于“Bridge 这一站不施工”。

**为什么还要保留它：** 如果只列出 factor 2、4、8，就无法公平回答“完全不做 Bridge，
直接优化原始循环是否更好”。factor 1 还是 O/B/C 分解的重要基准：Bridge factor 1 时，
O 与 B 必须是同一 artifact，Bridge 收益精确为零。

**怎样处理：** Bridge Pass 不构造循环，但控制层仍保存这个候选的 identity。后面的 route
如果能在原 IR 中找到合法循环，仍可以工作。

**何时拒绝：** factor 1 本身不会因地址独立性被拒绝，因为它没有合并 program。但如果
原 IR 没有 route 能使用的循环，那么对应 route 候选仍会以“subject 不存在”被拒绝。

**例子：** 原 kernel 已经有一个 `for i in range(8)`。Bridge factor 1 不改 program 结构，
完全展开+重排仍可以尝试作用于这个原生循环。

#### 子域二：单轴 program coarsening

**看到了什么：** kernel 使用某一个 program-id 轴，例如 `pid = tl.program_id(0)`，不同
program 处理沿该轴排列的数据块。

**它是什么意思：** 将相邻的若干 logical program 合并到一个 physical program。若 factor
为 4，physical program 0 依次模拟 logical program 0～3，physical program 1 模拟 4～7。

**为什么可能安全：** 如果每个 logical program 只读写自己的数据区间，不同 program 之间
没有重叠写入，也没有本 program 的写依赖另一个 program 的未完成读，那么顺序执行这几份
工作不会改变结果。

**怎样证明：** Discover 沿指针表达式追踪 program-id 的系数、每个 program 内的局部
最小/最大偏移、mask 和访问宽度。对于 store，要证明不同 program 的写区间不重叠；共享
pointer root 上同时存在读写时，还要排除跨 program alias。

**怎样改写：** 缩小 physical grid，在一个新的 `scf.for` 中计算
`logical_pid = physical_pid * factor + lane`，再克隆原工作。最后一组不满 factor 时使用
边界 predicate，不能越过原始 logical grid。

**正例：** 每个 program 写 `out[pid * 256 : pid * 256 + 256]`。相邻 program 的输出区间
首尾相接而不重叠，可以构造 Bridge。

**反例：** 每个 program 写 `out[pid * 128 : pid * 128 + 256]`。相邻区间重叠 128 个元素，
执行顺序可能改变结果，应拒绝。

**还要注意：** 不能只看 `pid * stride`，也要看 program 内部所有局部偏移。如果 stride
是 256，但局部实际访问到 300，仍可能跨界重叠。

#### 子域三：多轴 Bridge

**看到了什么：** kernel 同时使用两个或三个 program-id 轴，例如二维 grid 中的 x、y。

**它是什么意思：** 不只沿一个方向合并，而是选择每个轴的 divisor。例如 x 轴合并 2、
y 轴合并 4，一个 physical program 最多承担 `2×4=8` 个 logical program。

**为什么不能简单使用一个整数：** factor 8 可能是 `8×1`、`4×2` 或 `2×4`。这几种选择
恢复 logical 坐标的方式、地址连续性和局部性都不同，所以必须保存 divisor 向量，不能只
保存乘积 8。

**怎样改写：** 循环 lane 通过 mixed-radix 拆成 x/y 局部坐标，再与 physical program
坐标组合，恢复每个 logical program 的原始 x/y program-id。

**怎样证明：** 每个轴的 extent、divisor、坐标方向和地址 footprint 都要闭合；divisor
乘积必须等于 Bridge factor。不同二维 logical program 的写入仍须互不重叠。

**正例：** 一个二维 tile kernel，每个 `(pid_x, pid_y)` 独占一个不重叠矩形块，x 合并 2、
y 合并 2 后，一个 physical program 顺序处理四个矩形。

**反例：** 地址只使用 `pid_x + pid_y`，多个二维坐标映射到同一输出位置，无法证明独占，
应拒绝。

#### 子域四：包含提前 return 的 Bridge

**看到了什么：** kernel 先判断当前 program 是否越界，越界时立即 `return`，否则继续执行。

简化示例：

```python
pid = tl.program_id(0)
if pid >= n_programs:
    return
do_work(pid)
```

**为什么直接克隆会出错：** Bridge 后，一个 physical program 的循环中可能只有某些
logical pid 越界。如果仍保留函数级 return，第一个越界 lane 会结束整个 physical
program，使后面本来有效的 lane 也不执行。

**怎样改写：** 将“是否继续”变成每次循环迭代自己的 active predicate：

```text
for lane:
  logical_pid = ...
  active = logical_pid < n_programs
  if active:
    do_work(logical_pid)
```

**为什么安全：** 原来每个 logical program 都独立决定是否执行 continuation；改写后仍是
每个 logical program 独立决定，只是不再用函数级 return 结束其他 lane。

**准入限制：** 当前只允许能够正规化的结构，例如唯一、立即的 void return，以及能够
安全谓词化的 continuation。若 return 携带值、控制流复杂、或者 continuation 包含无法
谓词化的未知 effect，就拒绝。

**反例：** return 前后存在必须严格执行一次的全局副作用，或者多个分支最终以不同值
return。不能简单套 active predicate。

#### 子域五：pure helper call

**看到了什么：** kernel 主体没有直接写出全部计算，而是调用 helper 函数。

**问题在哪里：** Bridge 要复制或移动工作。如果 helper 内部做了什么完全未知，就不能
证明不同 logical program 独立，也不能安全克隆。

**怎样允许：** 若 call graph 可以闭合证明 helper 是纯的，或者 helper 能安全内联，再对
内联后的真实操作做地址、依赖和 effect 分析，就可以进入 Bridge。

**这里“纯”是什么意思：** 对 Bridge 而言，至少不能有未声明的全局写、atomic、volatile、
未知 region effect 或依赖外部可变状态的行为。仅仅函数名叫 `helper` 不构成证明。

**正例：** helper 只做张量加法、乘法并返回结果，没有内存副作用。

**反例：** helper 内部调用未知外部函数，或修改一个多个 program 共享的全局计数器。

**为什么这是共性能力：** helper 透明化以后，三条 route 看见的仍是普通循环体。它不应
分别在软件流水、重排和向量化里各写一个 workload adapter。

#### 子域六：recursive effect container

**看到了什么：** 循环体中有 `if`、嵌套 region、reduction combiner 等结构，内存操作
藏在更深层，不全位于顶层 block。

**为什么顶层扫描不够：** 如果只看最外层 operation，可能错误地认为一个 `if` 没有副作用，
却漏掉它的 region 内存在 store 或 atomic。

**怎样证明：** 分析器递归进入每个 region，收集 load、store、atomic、volatile 和未知
effect；对 pointer root 与 alias 关系也递归检查。只有整个容器的 effect 闭合，才允许复制。

**正例：** `if mask: store(out + pid_offset, value)`，条件和写地址都能证明是每个 program
私有，可以保留结构并克隆。

**反例：** region 内有 atomic add 到所有 program 共享的位置。即使顶层只看到一个 `if`，
仍必须拒绝。

**与“匹配特定操作组合”的区别：** 这里不要求 `if` 中必须出现哪几个算子，而是递归检查
真实存在的任意 operation effect。

#### 子域七：连续分区递推

**看到了什么：** 每个 virtual program 负责全局序列中一个连续分区，并在自己的分区内
执行递推或扫描。

**它为什么单独成为子域：** 普通单轴 Bridge 主要证明不同 program 的地址区间独立；连续
分区递推还要证明每个分区的起止边界、分区顺序，以及 carried state 是否只在分区内部，
不能跨分区偷偷依赖前一个 program 的最终状态。

**怎样改写：** Bridge 循环逐个恢复 virtual program 对应的连续区间，并在每个区间内执行
原递推。分区边界必须静态或由闭合证书给出。

**正例：** 每个 program 对自己独占的固定长度片段计算局部前缀，输出只属于该片段，且
下一个片段不读取上一个片段的末状态。

**反例：** program 1 的初始累计值来自 program 0 的最终累计值。将它们合并后虽然可能
顺序上碰巧可运行，但这已经改变了原有并行语义和同步假设，不能按独立 Bridge 准入。

### 9.6 三条 route 的子域逐项说明

Bridge 的子域回答“循环怎样来”；route 子域回答“对这个循环具体怎样优化”。同一个循环
可以对某条 route 合法、对另一条 route 不合法，拒绝原因不能互相借用。

#### 软件流水子域一：原 IR 已有的 live loop

**看到了什么：** 原始 TTIR 已经存在没有被完全消除的循环，循环中有后端可识别的加载、
计算和跨迭代调度机会。

**怎样做：** L-lite 把 stage factor 写到确切 subject 上，验证 subject/stage 请求保留；
TritonGPU 后端再建立真实 pipeline schedule。

**为什么可能变快：** 下一迭代的数据搬运与当前迭代计算重叠，可以隐藏部分内存等待。

**典型拒绝：** 循环内没有可流水化加载、存在阻止跨迭代提前执行的依赖、stage 请求超过
能力，或者循环在后续 Pass 前已经消失。

#### 软件流水子域二：Bridge 构造出的 loop

**看到了什么：** 原 IR 没有目标循环，但 Bridge 已构造一个普通、带来源证书的 `scf.for`。

**它是否天然可流水化：** 不是。Bridge 只证明多份 logical program 可以安全顺序放在一个
循环里，没有证明循环体具有异步加载服务或能跨迭代重叠。

**怎样判断：** Facts 必须重新对 Bridge 后的真实 body 做原生 pipeline capability 分析，
检查 load service、payload width、依赖和 live-loop 状态。

**正例：** 每个 logical program 都先从全局内存加载独立 tile，再做较长计算；下一个 lane
的 tile 可以提前搬运。

**反例：** Bridge body 只有 store，或者下一 lane 的地址依赖当前 lane 刚加载并计算出的
结果。此时没有独立可提前加载阶段，软件流水拒绝是机制不适用，不代表 Bridge 本身错误。

#### 重排子域一：静态完整展开

**看到了什么：** trip count 静态已知，route factor 完整覆盖目标循环，例如 trip=4、
factor=4。

**怎样做：** 原生 LoopUnroll 复制四份循环体；L-lite 根据 source operation lineage、SSA
依赖和 memory effect 构造稳定拓扑顺序，把可以同行的 operation phase 聚合。

**物化成功长什么样：** focal loop 消失；每份 clone 的来源仍可追踪；新的操作顺序没有
违反依赖；validator 能看到 phase reorder postcondition。

**拒绝原因：** 需要交换顺序的操作之间存在真实数据依赖、内存顺序无法证明，或展开后的
lineage 不完整。

#### 重排子域二：main + tail

**看到了什么：** trip count 大于 factor，或不能整除 factor，例如 trip=10、factor=4。

**怎样做：** 每四次迭代组成一个 main group，在 main group 内展开重排；最后两次保留为
ordered tail。tail 不能为了追求“完全展开”被丢弃或越界执行。

**怎样验证：** validator 同时看到 main partition 的重排证据和 tail lineage。目标循环可能
以有解释的 tail 形式存在，不能用“还有 loop”简单判定物化失败。

#### 重排子域三：operation-neutral phase

**看到了什么：** 任意操作构成的循环体，不要求固定 load/compute/store 配方。

**怎样划分 phase：** phase 来自 source operation group 与依赖闭包。存在 load 时可以形成
load 相关组；只有 store 时就只对真实 store/计算组排序；一般纯计算也按其操作和依赖处理。

**为什么叫 operation-neutral：** 准入条件描述“哪些操作能交换顺序”，不描述“必须出现
哪些操作”。核心入口是
[`materializeOperationNeutralPhase`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L8124)。

**反例：** 如果实现写成“必须正好一个 load、一个 compute、一个 store”，那是在匹配配方，
不是 operation-neutral，应当被去定制审计否决。

#### 向量化子域一：exact load packing

**看到了什么：** 展开后多份 load 在结构上同构。

**必须逐项对应什么：** pointer 关系、元素类型、mask、other 值、边界检查、cache 和
eviction 语义。仅仅都叫 load 不够。

**怎样做：** 将 lane 对应的 pointer、mask 和 other 形成更宽容器，再发出语义等价的宽
load；结果按 lineage 交还各 consumer。

**拒绝例子：** 四个 load 中一个使用不同 mask 或不同 cache 语义，不能粗暴打包。

#### 向量化子域二：exact store packing

**看到了什么：** 多份 store 的 pointer、value、mask 与 effect 顺序可精确对应。

**为什么比 load 更谨慎：** store 会改变外部可见内存；错误重排或合并可能覆盖别的 lane，
也可能改变有意义的写入顺序。

**怎样做：** 只有地址互不冲突、value 图同构、mask 对应且副作用顺序闭合，才合成宽 store。

**拒绝例子：** 两个 lane 写到同一地址，或者后一个 store 有意覆盖前一个 store，不能当作
独立向量 lane。

#### 向量化子域三：一般 elementwise operation packing

**看到了什么：** 多份算术、比较、select 等一般 operation 的完整图同构。

**同构具体指什么：** operation 名称、输入输出类型、属性和 operand 之间的对应关系相同；
外部 invariant operand 可以共享，lane-local operand 必须一一映射。

**怎样做：** 将 lane-local operand 组成宽值，在宽值上执行一次同类 operation，再把结果
按 consumer 需要继续传播或拆分。

**拒绝例子：** 一路是乘法、一路是加法；或者同为乘法但一路有不同舍入/饱和属性。源码
行数相似不代表 operation graph 同构。

通用入口是
[`materializeExactOperationVectorization`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L9413)。

#### 向量化子域四：Bridge logical packing

**看到了什么：** Bridge 产生的 logical lane 都执行同构工作，lane identity 和地址平移
关系有证书。

**怎样做：** 将多个 lane 的 tensor 值用 `tt.join` 等结构组成逻辑宽值，尽量提升 invariant、
融合 tensor lane，并在能证明时消除无意义的 split/join。

**为什么要保留 Bridge lineage：** 向量化后仍必须证明这些 lane 对应原来的哪些 logical
program，不能只看一个更宽 tensor 就丢掉 program ownership。

入口是
[`materializeBridgeLogical`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L9114)。

#### 向量化子域五：exact-prefix

**看到了什么：** 最大容器宽度静态已知，但真实有效长度 N 在运行时确定，而且有效元素
恰好是从 0 开始的连续前缀。

**怎样做：** 选择足够容纳最大范围的静态、通常为 power-of-two 的容器，构造
`lane < N` predicate；无效 lane 使用不影响语义的 neutral value，然后进行向量操作或
reduction。

**为什么不是第四条 route：** 它仍然是在把多次/多 lane 同构工作装进宽容器，只是有效
lane 数动态变化，所以属于完全展开+向量化的动态子型。

**拒绝例子：** 有效元素不是连续前缀，而是任意稀疏集合；或者找不到某个 reduction 的
正确 neutral value；或者最大容器超过显式预算。

#### 向量化子域六：sibling state-axis SLP

**看到了什么：** 同一循环或相邻区域中，连续生成 n 个状态对象，它们经历同构操作；中间
还可能有状态之间的合并、归一化或 reduction。

**它不要求什么：** 不要求状态数量等于 4，不要求变量名为 r0～r3，也不要求固定出现
clamp、softmax 或 Sinkhorn 组合。

**怎样证明：** 比较每个状态的完整 operation graph、类型、属性、reduction combiner、
跨状态 consumer 和更新顺序。只有能建立一一对应的 lane 映射，才沿新状态轴打包。

**怎样处理状态间操作：** 例如各状态分别归一化后又求和，前半部分可以沿状态轴并行，
状态间求和则是该轴上的 reduction。必须保留这个依赖，不能把整个代码块当作互不相关的
n 路 elementwise。

**拒绝例子：** 第三个状态使用完全不同的 recurrence，或者某一路更新依赖前一路刚更新
后的值，无法证明同时打包等价。

#### 嵌套结构子域：同一 route 穿过一层或多层 loop

**看到了什么：** outer 和 inner loop 均存在，或 Bridge 构造 outer、原 IR 保留 inner。

**怎样表达：** route identity 只有一份；structural scope 说明 factor 对哪几层的组合
extent 生效。完整展开可选择 outer、inner 或二者乘积上限。

**为什么不使用两条 route：** 若 outer 选重排、inner 选向量化，就变成两个独立干预，
因果归属、候选空间和验证条件都会成倍复杂。当前设计先保证一次候选只有一个 route 机制，
但允许它覆盖嵌套结构。

**拒绝例子：** 内外循环存在 loop-carried 依赖，导致把二维迭代空间展平后顺序发生变化；
或者乘积 extent 超过显式预算。

### 9.7 一个规则式子域怎样才算闭环

评审每个子域时，可以统一使用下面这张表，而不是只问“代码里有没有 if 分支”：

| 闭环项目 | 必须提供的证据 |
|---|---|
| 对象定义 | 哪种 IR 结构属于这个子域，哪种不属于 |
| 机制理由 | 为什么这类结构可以进行该变换 |
| 准入规则 | 使用哪些 IR/SSA/address/effect 事实 |
| 真实物化 | 哪个 Pass、哪个 materializer 具体改写 |
| 后置条件 | 改写后必须观察到什么，哪些临时状态必须消失 |
| 正例 | 至少一个完整 pipeline 成功并通过正确性 |
| 最早层反例 | 每种关键限制都在所属责任层明确拒绝 |
| 防回退 | 失败候选不会变回 Original 冒充成功 |
| 去定制审计 | 不读取 kernel、算子、benchmark 或历史 winner identity |
| 能力同步 | 同一 CandidateCompiler 同时供 L-lite 与主 L 使用 |

任何一栏只有一句“支持”都不算完成。尤其是“发现了对象”不能替代“route 物化成功”，
“TTIR 属性存在”不能替代“最终机器码出现目标机制”。

---

## 10. 动态循环、提前 return、helper 和 state-axis 如何处理

这些能力不应各自变成“某个 benchmark 特例”，而应成为三条 route 可共享的结构能力。

### 10.1 动态循环

循环次数运行时才知道时，必须有 runtime binding、main-tail 或 exact-prefix 证书。不能
把一次测试观察到的 N 当作永远固定的 trip count。

从三个 route 看，动态并不是同一处理：软件流水可以保留 live dynamic loop，只要后端
能够对其调度；重排需要知道每个 main group 如何形成以及 tail 如何按原顺序执行；向量化
需要知道哪些 lane 有效、无效 lane 使用什么 predicate 或 neutral value。因此“支持动态
循环”不是一个全局开关，而是共享事实与各 route 后置条件的组合。

runtime binding 还必须与 specialization identity 绑定。若本次编译假设上限为 128，运行
时却传入 1024，不能继续使用原计划。要么 guard 失败回 Original，要么重新生成适用的新
候选，不能让越界容器继续执行。

### 10.2 提前 return

如果 kernel 中存在可证明结构的单一提前 void return，Bridge 可以把后续 continuation
放进 active predicate。无法证明 effect 可以安全谓词化时就拒绝。

通用检查从
[`continuationOperationIsPredicatable`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L410-L429)
开始，并不读取算子名。

正规化应发生在共享结构层，而不是让重排和向量化分别猜一遍 return 的含义。Bridge 先把
函数级退出变为 lane-local active predicate，后续 route 只处理普通的结构化条件。如果
软件流水能消费这种正规化循环，而另外两条 route 不能，首先应检查共享 predicate/region
物化能力是否缺失，而不是为具体 kernel 新建 adapter。

### 10.3 helper call

helper 必须能够证明纯、内联或在闭合 call graph 中没有未知副作用。递归、未知 call 或
含 effect 的 helper 默认拒绝。

“默认拒绝”不是说 helper 永远不支持，而是证明责任不能由名称或人工承诺代替。后续可
增加更强的 call summary：明确列出 helper 读取和写入哪些 pointer root、是否依赖 program-id、
是否有同步或 atomic。只要 summary 可由编译器复核，helper 就能像内联代码一样进入通用
依赖分析。

### 10.4 state-axis

若多个对象具有同构操作图，可尝试沿状态轴打包。判据应比较完整 operation graph、类型、
reduction combiner 和跨状态 consumer，而不是匹配 `r0/r1/r2/r3` 或某段 Sinkhorn
源码。状态数量由真实 cardinality 决定，不固定为四。

state-axis 的复杂点是“状态之间也可能发生操作”。如果 n 个状态先各自做归一化，再求共同
列和，然后各自除以这个列和，前半段是 n 路同构 operation，列和是跨状态 reduction，后半
段又回到 n 路同构 operation。物化器必须理解整张图的 lane-local 边和 cross-state 边，
不能把它简化成“连续出现 n 条相似语句”。

### 10.5 只有 store 或没有 load 的循环

这类循环仍可能适合重排或向量化。它不适合软件流水的原因可能是没有可重叠的 load
service，但这个拒绝只能属于软件流水 route，不能污染另外两条 route 的准入。

反过来也一样：一个循环适合软件流水，不自动说明它能向量化。软件流水可能只需要存在
可提前搬运的加载，而向量化还要证明多 lane 操作图、地址、mask 和类型精确对应。项目的
覆盖矩阵必须按“subject × route”报告，而不能只写“这个循环可优化”。

---

## 11. 相对原生 Triton，代码到底改了什么

原生 Triton 已经提供：

- 通用 `scf.for` 展开；
- TritonGPU 软件流水分析和 lowering；
- 原生 Autotuner 的配置遍历、计时、winner 选择和 key cache；
- NVIDIA/AMD 常规后端 codegen。

L-lite 没有重写这些能力，而是在其上增加“候选控制、因果身份和证明层”。

### 11.1 修改的原生文件

| 文件 | 改动 | 为什么需要 |
|---|---|---|
| [`Passes.td`](../include/triton/Dialect/Triton/Transforms/Passes.td#L93-L125) | 注册六个新增 Pass | 让能力成为正式 MLIR Pass，而非测试脚本 |
| [`CMakeLists.txt`](../lib/Dialect/Triton/Transforms/CMakeLists.txt#L5-L29) | 编译 `HBVLoop.cpp` | 把实现接入 `TritonTransforms` |
| [`LoopUnroll.cpp`](../lib/Dialect/Triton/Transforms/LoopUnroll.cpp#L24-L195) | 保存 source trip、main/tail 和操作 lineage | 让展开后的复制品可被重排、向量化和验证追踪 |
| [`passes.cc`](../python/src/passes.cc#L39-L59) | 导出 Python Pass wrapper | 允许 Python 显式组装 pipeline |

#### `Passes.td` 的具体作用

MLIR Pass 不是写一个 C++ 类就会自动出现在编译器里。`Passes.td` 声明 Pass 名、所属
operation 和可配置选项，使 `triton-opt`、C++ PassManager 和 Python wrapper 能以统一名字
找到它。如果注册名称、C++ 实现和 Python 导出不一致，用户可能看到 wrapper 存在，却在
实际 pipeline 中调用了不同 Pass 或根本无法调用。

#### `CMakeLists.txt` 的具体作用

它决定 `HBVLoop.cpp` 是否真正编进 `TritonTransforms`。只把源码放进仓库、不加入构建目标，
测试脚本也许能读取文件，但安装后的 `libtriton.so` 不含这些能力。构建接线是“真实编译器
能力”与“文档中的设计代码”之间的最基本区别。

#### `LoopUnroll.cpp` 为什么必须修改原生文件

重排和向量化需要在 unroll 后知道 clone 来源。若另写一个完全独立展开器，会重复原生
MLIR 的 main/tail 处理，容易与 Triton 后续演化分叉。最小修改原生 LoopUnroll 的优点是
继续复用成熟展开行为，只增加 L-lite 需要的 lineage；风险是升级 Triton 时必须检查这组
标签是否仍与新版 unroller 的 clone 和 epilogue 行为一致。

#### `passes.cc` 的具体作用

它把 C++ Pass 暴露给 Python PassManager。L-lite 的候选由 Python 生成，因此 CandidateCompiler
需要按冻结顺序为每个 PlanBundle 显式添加 Pass。wrapper 只提供“能添加”，不自动修改
Triton 默认 pipeline，这也是当前尚未一键产品化的原因之一。

### 11.2 新增的主要实现文件

| 文件 | 主要内容 |
|---|---|
| [`HBVLoop.cpp`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp) | 发现、事实、Plan 解析、Bridge、route 物化和验证 |
| [`factor_ontology.py`](../python/triton/l_lite/factor_ontology.py) | factor、subject、subtype 和 nested scope 的定义 |
| [`contract.py`](../python/triton/l_lite/contract.py) | Bridge→route 组合合法性 |
| [`composition.py`](../python/triton/l_lite/composition.py) | 笛卡尔图、候选 identity 和物化证明 |
| [`autotune.py`](../python/triton/l_lite/autotune.py) | 原生 exhaustive autotune 适配层 |
| [`__init__.py`](../python/triton/l_lite/__init__.py) | 公开 Python API |

#### 为什么 C++ 与 Python 都有合法性相关代码

Python 侧先建立完整请求图和语义 contract，方便在编译前给所有 cell 类型化身份；C++ 侧
必须根据真实 IR 事实再做一次 fail-closed Decision，因为 Python 不拥有最终模块中的每个
SSA、effect 和 subject locator。两者不是互相替代，而是“请求是否合法表达”与“真实 IR
是否满足请求”的两道门。

两道门也带来漂移风险：如果 Python 认为一个字段含义为 A，C++ 认为是 B，候选会在错误
层失败。因此后续建议用生成式 schema 或 round-trip 测试，确保 ontology、serializer 和
parser 使用同一个定义来源。

#### 为什么 `HBVLoop.cpp` 很大是风险

10,922 行单文件同时包含事实提取、历史 adapter、Plan 解析、Bridge、多个 materializer 和
validator。它的优点是早期开发容易共享 helper；缺点是一个 helper 的修改可能同时影响
多个因果域，历史路径也可能意外重新进入 active 分支。拆文件不是为了美观，而是建立更
清楚的代码 authority 和依赖方向，例如 Facts 不应反向调用性能选择逻辑，历史 reader 不应
注册生产 materializer。

实现快照相对原生基线共 27 个文件级改动、15,490 行新增。大部分新增行集中在
`HBVLoop.cpp`，这是当前可维护性风险之一，不应把“代码很多”误当成“架构必须如此
集中”。

---

## 12. 当前公开版本已经完成什么、还没完成什么

### 12.1 已经完成

- Bridge 和 route 已作为真实编译器 Pass 实现，而不是手写 kernel；
- Bridge factor 与 route factor 有独立语义；
- Python 能构造 V1/V2 候选图并保存类型化拒绝；
- 编译器包含 Discover、Facts、Decision、Bridge、Materialize 和 Validate；
- 原生 LoopUnroll 保存了 L-lite 所需 lineage；
- active 准入主体使用 IR、SSA、地址与 effect 规则，不依赖 workload allowlist；
- 源树可以构建；
- 已有 4 个 Python 控制层测试和 13 个公开 MLIR 测试通过。

验证记录见
[`l-lite-capability-validation-2026-08-25.md`](l-lite-capability-validation-2026-08-25.md)。

这里的“完成”必须逐项限定：

- **真实 Pass 已实现**，表示源码可构建、Pass 可注册和显式调用，不表示默认 JIT 会自动
  运行；
- **V1/V2 图可构造**，表示候选语义和请求网格能形成，不表示 V2 的每个 cell 已生成可执行
  binding；
- **active 规则去名称化**，表示准入主体不读取 workload 名，不表示所有历史 matcher 已从
  物理源码删除；
- **测试通过**，表示已列测试覆盖的断言成立，不表示未覆盖 route、artifact 或 GPU 性能也
  自动成立。

因此评审报告后续应给每项“完成”附证据角色和适用 population，而不是只写一个绿色状态。

### 12.2 尚未完成

公开仓还缺少一个统一入口，把普通 `@triton.jit` kernel 自动变成完整候选产品：

```text
普通 kernel
  → census
  → V2 候选图
  → 为每个候选生成 PlanBundle
  → 运行唯一的完整 Pass pipeline
  → 得到 KernelInterface binding
  → artifact 验证
  → correctness 验证
  → 原生 autotune
  → 完整 acquisition 报告
```

具体缺口是：

1. 公开的 `candidate → PlanBundle → compiled binding` 工厂尚未提供；
2. V2 图尚未直接接通当前 native autotune facade；
3. 公开 MLIR 测试主要覆盖 Bridge/事实层，没有系统覆盖三条 route 的完整七段 pipeline；
4. 公开仓缺少统一 PTX/cubin identity、GPU oracle 和端到端性能 harness；
5. 因而当前公开证据不能宣称全部 route 已在公开流程中端到端物化，也不能宣称已取得
   性能收益。

第一个缺口是控制面的主缺口。现在外部调用者必须自己把候选翻译成 PlanBundle、组装
PassManager、编译并提供 binding。不同调用者可能采用不同 pipeline 或漏掉 validator，
导致同名 L-lite 实际不是同一实验。

第二个缺口是版本 authority 的主缺口。V2 已表达更完整的 subject/factor 语义，但当前
autotune domain builder 仍主要接 V1 图。如果直接测 V1，就不能把结果表述为 V2 active
能力的完整对照。

第三、四个缺口是证据链缺口。局部 MLIR 正例只能说明某项改写在一个输入上出现；只有
完整 pipeline、最终 artifact 和数值 oracle 连起来，才能排除“前一步成功、后一步消失”
或“代码更快是因为算错”的情况。

第五项是结论边界。项目源码规模再大，也不能替代未执行的性能实验。性能收益必须来自
独立、环境合格、包含完整 acquisition 的真实测量。

### 12.3 为什么“Pass 已注册”仍不等于“一键可用”

Pass wrapper 已允许调用者显式组装 pipeline，但当前分支没有把这些 Pass 注入 Triton
默认 TTIR pipeline。也没有公开 binding 工厂自动为每个候选完成整条编译链。

因此准确定位是：

> 当前是“已接入并可构建的编译器能力面 + 候选控制框架”，还不是“普通 kernel 输入后
> 一键返回 exhaustive winner 的完整产品”。

---

## 13. 为什么建议把 L-lite 做成完整候选编译产品

结论：应该做，但采用显式 opt-in API，不改变所有 Triton 用户的默认行为。

建议形态：

```python
product = triton.l_lite.compile_candidates(
    kernel,
    specialization=...,
    launch=...,
    bridge_factors=(1, 2, 4, 8),
    route_factors={...},
    correctness=oracle_contract,
)

winner = product.autotune(*args, grid=grid, key=(...))
report = product.report()
```

为什么需要统一产品入口：

- 如果每个人手工构造 PlanBundle，实际执行的候选可能不一样；
- 如果 binding 工厂不唯一，L-lite 与主 L 可能比较的并不是同一批代码；
- 如果失败候选没有统一 disposition，探索成本会在报告里消失；
- 如果 artifact 和 correctness 没有统一验证，错误或回退候选可能进入计时。

### 13.1 建议的产品流水线

```text
LoopCensus
  → CandidateEnumerator
  → PlanBundleSerializer
  → CandidateCompiler
  → ArtifactChecker
  → CorrectnessQualifier
  → NativeAutotuneControlV2
  → AcquisitionReport
```

### 13.2 每个模块只做什么

| 模块 | 通俗责任 | 绝不能做的事 |
|---|---|---|
| LoopCensus | 列出真实观察对象和能力 | 看计时或 kernel 名决定对象 |
| CandidateEnumerator | 固定完整笛卡尔积 | 根据预测收益提前删候选 |
| PlanBundleSerializer | 把候选准确翻译成编译计划 | 手写未注册字段 |
| CandidateCompiler | 按唯一 pipeline 编译并记录结果 | silent fallback |
| ArtifactChecker | 确认最终产物身份和后端事实 | 用“Pass 成功”代替 artifact 证据 |
| CorrectnessQualifier | 与 Original 和 oracle 比较 | 用性能决定正确性容差 |
| NativeAutotuneControlV2 | 对全部可执行候选实测 | 使用主 L 预测剪枝 |
| AcquisitionReport | 汇总成功和失败的全部成本 | 隐藏失败候选开销 |

#### LoopCensus 详细职责

输入是一份已经 specialization 的 JIT function、launch/grid 事实和目标信息。它运行发现与
事实层，输出稳定的 loop/Bridge subject 列表、每个 subject 的 trip/nest/effect 能力和
Provider certificate。Census 不能决定 route 胜负；同一份 census 应同时供 L-lite 和主 L
使用。

#### CandidateEnumerator 详细职责

它读取 census 与预注册 factor domain，建立 Original 和全部
`Bridge × Route × RouteFactor × StructuralScope` 请求。它可以依据静态语义标注某 cell
类型化拒绝，但不能根据旧计时或预测收益不生成这个 cell。这样报告才能区分“请求过但不
合法”和“从未进入候选域”。

#### PlanBundleSerializer 详细职责

它把 Python 候选对象转换成 C++ parser 唯一接受的 schema，并计算 plan hash。serializer
要做 round-trip：序列化后再解析，所得 route、factor、subject 和证书必须与原对象完全
相同。未知 adapter、缺字段或多字段都应在这里尽早失败。

#### CandidateCompiler 详细职责

它是唯一编译 authority。对每个 cell，它绑定 runtime scalar/grid 事实、注入 PlanBundle、
按冻结顺序运行七段 TTIR pipeline、继续原生 lowering，并输出 typed compile outcome、
TTIR/PTX/cubin hash 与 KernelInterface binding。任何 Pass 或 postcondition 失败都必须形成
显式结果，不能吞掉异常后编译 Original。

#### ArtifactChecker 详细职责

TTIR validator 只能证明中间层目标。ArtifactChecker 继续检查 PTX/cubin 是否与 Original
不同、软件流水是否出现相应后端事实、向量化/重排是否在 lowering 中仍留下可识别结果。
它不能规定“机器码必须长成唯一形状”，因为后端可能有多种等价实现；检查项应由 route
机制所需的必要事实定义。

#### CorrectnessQualifier 详细职责

它在性能之前运行 Original、candidate 和独立 oracle。容差由 dtype、算法数值语义和预注册
正确性合同决定，不可因为某候选很快而放宽。NaN、Inf、动态 tail、mask 和输出未写区域都
需要单独处理。结果为 correct、incorrect 或 ambiguous；只有 correct 能进入性能计时。

#### NativeAutotuneControlV2 详细职责

它接收完整请求 disposition 与所有 correct binding。可执行候选真实 benchmark；不可执行
cell 保留 identity 并以失败 disposition 计入报告，但不会启动错误 kernel。它继续使用
原生 winner 规则，不调用主 L 预测器。

#### AcquisitionReport 详细职责

它按候选汇总 census、serialization、compile、artifact、correctness 和 benchmark wall time，
还报告总耗时与最终 winner。失败成本不能丢失，因为对照的核心问题之一就是 exhaustive
搜索付出了多少代价。报告还必须记录环境 authority，区分 CPU 编译时间和 GPU 测量时间。

最低完成标准不是“API 能返回一个 winner”，而是：每个请求 cell 有且只有一个可复核
结果；每个可执行候选的 Plan、artifact、correctness 和 timing identity 完全一致；没有
候选无解释地消失。

---

## 14. 测试该怎么分层

单个“测试通过”不能同时证明全部层次。建议按以下顺序验证：

### 14.1 规则单元测试

验证 factor、subject、组合 contract 和类型化拒绝。无需 GPU。

这类测试直接构造 Python 对象，检查相同输入是否产生稳定 identity、不同 factor 是否使用
各自语义、main/tail 算法是否正确、V1/V2 contract 是否在同一责任层拒绝错误组合。它还应
覆盖边界值，例如 factor=1、factor 大于 trip、trip 不能整除 factor、动态证书缺字段、嵌套
scope 乘积溢出。

单元测试的价值是错误定位快：如果 `trip=10,factor=4` 算出了 tail=0，不需要编译任何
kernel 就能发现 ontology 错了。但它使用人工构造事实，不能证明 C++ Facts 从真实 IR 提取
出的字段也正确，更不能证明 materializer 真实改写。

### 14.2 MLIR Pass 正反例

每条规则都要有：

- 能合法通过并物化的正例；
- 在最早责任层被拒绝的反例；
- 防止 silent fallback 的反例。

正例要检查具体 IR 后置事实，不能只检查进程退出码为 0。例如 Bridge 正例应检查新的
`scf.for`、logical pid 恢复、factor/origin lineage；重排正例应检查操作顺序与 phase 标记；
向量化正例应检查宽值或 packing postcondition。

反例应锁定最早 owner。例如跨 program 写区间重叠应由 Discover/Decision 拒绝，不能先构造
错误循环，再等 correctness test 发现。每个反例还要检查类型化原因，防止将来代码因另一个
无关错误“碰巧仍失败”。

silent-fallback 反例则人为制造“请求 route，但物化后没有目标事实”的情况，确认 validator
会失败。它保护的是编译器不会把 Original 当作优化候选，而不是保护数值等价。

### 14.3 完整 pipeline 测试

对每条 route 运行完整的
`Discover→Bridge→Facts→Decision→Unroll→Materialize→Validate`，而不只单测一个局部
Pass。

完整测试要为同一个 PlanBundle 运行唯一冻结顺序，并检查每一站消费/产生的属性。它能发现
局部 Pass 测试看不到的接口问题，例如 Facts 在 Bridge 前运行导致看不到新循环，Unroll
提前清除了 Materialize 需要的 subject，或者 Validate 读取了旧 adapter 字段。

至少要按 Bridge 子域、三条 route、静态/动态、main-tail、嵌套与 state-axis 建立覆盖矩阵。
矩阵中的每个成功格都应有一个对应最早层失败格，避免只测 happy path。

### 14.4 artifact 测试

分别检查 TTIR、PTX、cubin/SASS identity。软件流水尤其要检查后端最终是否真的产生
相应异步搬运或调度事实，不能只看 `tt.num_stages` 请求存在。

artifact 测试分三层：

1. **TTIR identity：** Plan 与结构变换是否存在；
2. **PTX identity：** lowering 后指令类别、地址形态或 typed live-shape 是否与机制一致；
3. **cubin/SASS identity：** 最终机器码是否仍不是 Original，并包含机制必要事实。

hash 不同只能证明字节不同，不能证明为什么不同；hash 相同则能明确发现候选回退。更强的
检查需要 route-specific necessary facts。例如软件流水可检查 async copy/commit/wait 或等价
调度证据；向量化可检查宽访问或相应指令/布局变化。检查应允许后端的多种合法实现，不能
把某一版 SASS 文本写成唯一模板。

### 14.5 正确性测试

Original、候选和独立 oracle 三方对比。动态边界、tail、mask、alias 和不同 factor 都要
覆盖。

正确性输入要覆盖正常值、边界值、非整除 tail、最小/最大动态 extent、不同 mask、NaN/Inf
策略和可能 alias 的反例。只用随机均匀输入容易漏掉尾部少写、错误 neutral value 或精度
退化。

Original 不能成为唯一 oracle，因为 Original 和 Candidate 可能共享同一个上游 bug。理想
情况是同时有简单参考实现或数学定义。若独立 oracle 成本过高，报告必须说明正确性证据只
是候选与 Original 一致，不能写成独立数学正确。

浮点变换可能改变运算顺序，容差必须预先按 dtype/算法定义，并同时报告绝对误差、相对误差、
ULP 或任务特定指标。不能看到 candidate 加速后再决定放宽多少。

### 14.6 性能测试

只有前面通过后才测性能。报告至少包含：

- kernel 与输入 identity；
- 可观察循环数；
- 原生软件流水可优化比例；
- 重排/向量化对同一对象的支持情况；
- 请求候选数、各类失败数和可执行数；
- 全部编译、正确性和 autotune acquisition；
- winner 的 Bridge/route/factor；
- kernel 加速比；
- 端到端高频 kernel 信息。

性能测试还必须记录 GPU UUID、驱动/runtime、外部 context、稳定性和共享状态。无法证明
环境符合协议的数据只能作为工程诊断，不能成为正式收益证据。

一次正式角色应把 O/B/C 放在同一个不可拆分测量单元内，交错或随机化顺序，并持续监控
外部 GPU context 与 stationarity。这样 Bridge 和 route 的相对时间在尽量相同的环境中获得。
若中途出现外部 PID、request 切换、明显降频或漂移，整组 O/B/C 作废，不能只删除其中一个
不理想样本。

“允许共享 GPU 的功能测试”和“要求独占的高精度性能测试”要分开标记。编译成功、artifact
检查和大多数正确性测试可以在共享环境执行；微小相对收益、区间校准和正式发布证书通常
需要严格环境。不能因为所有任务都使用 GPU，就一律要求同样昂贵的独占，也不能把共享环境
数据混进正式 timing。

### 14.7 当前公开 13 个 MLIR 测试具体证明了什么

现有测试主要覆盖 Bridge 的基本扫描、提前 return、仿射局部跨度、多轴 mixed-radix、纯
helper、递归 effect container、operation-neutral 准入、未分组轴地址平移、runtime mask
authority 和 body semantics。

例如 affine-local-span 测试的价值是证明 Discover 不会只看 program stride 而漏掉 program
内部偏移；operation-neutral 测试防止重新引入“必须存在固定 load/store 配方”；pure-call
测试区分可闭合 helper 和未知副作用 helper。

这些测试尚不能组合推出三条 route 全部完成，因为它们没有系统运行完整七段 pipeline，
也没有检查最终 PTX/cubin 和 GPU oracle。评审时应把“已覆盖 Bridge 通用规则”与“完整 route
产品闭环”分开打勾。

### 14.8 每个公开测试文件的具体责任

| 测试文件 | 它在验证什么 | 它没有验证什么 |
|---|---|---|
| [`hbv-loop-bridge-scan.mlir`](../test/Triton/hbv-loop-bridge-scan.mlir) | 基本 Bridge 对象扫描、发现属性与简单构造事实 | 三条 route 完整物化和 GPU 正确性 |
| [`hbv-loop-bridge-early-return.mlir`](../test/Triton/hbv-loop-bridge-early-return.mlir) | 可正规化提前 return 的正例，以及不能安全谓词化结构的拒绝 | 任意复杂 CFG 都受支持 |
| [`hbv-loop-bridge-affine-local-span.mlir`](../test/Triton/hbv-loop-bridge-affine-local-span.mlir) | program stride 与 program 内局部 min/max 共同参与不重叠证明 | 动态未知局部跨度可被乐观准入 |
| [`hbv-loop-bridge-affine-local-span-materialize.mlir`](../test/Triton/hbv-loop-bridge-affine-local-span-materialize.mlir) | 通过局部跨度证明后能真实构造 Bridge loop | route 物化和最终机器码 |
| [`hbv-loop-bridge-pure-call-closure.mlir`](../test/Triton/hbv-loop-bridge-pure-call-closure.mlir) | pure helper、effect helper 与嵌套 call closure 的边界 | 所有外部 call 都可分析 |
| [`hbv-loop-bridge-recursive-effect-container.mlir`](../test/Triton/hbv-loop-bridge-recursive-effect-container.mlir) | 递归 region 中的 alias、atomic、volatile 和未知 effect 能被发现 | 含这些 effect 的程序能够被 Bridge 改写 |
| [`hbv-loop-bridge-recursive-effect-container-materialize.mlir`](../test/Triton/hbv-loop-bridge-recursive-effect-container-materialize.mlir) | 递归 effect 闭合的结构可进入 Bridge 构造 | route 后端终态 |
| [`hbv-loop-bridge-operation-neutral.mlir`](../test/Triton/hbv-loop-bridge-operation-neutral.mlir) | Bridge/重排准入不要求固定 load+compute+store 配方 | 所有任意 operation 都一定可交换或向量化 |
| [`hbv-loop-bridge-ungrouped-axis-translation.mlir`](../test/Triton/hbv-loop-bridge-ungrouped-axis-translation.mlir) | Bridge 时未分组 program 轴的地址平移事实 | 任意非仿射轴关系 |
| [`hbv-loop-bridge-ungrouped-axis-translation-materialize.mlir`](../test/Triton/hbv-loop-bridge-ungrouped-axis-translation-materialize.mlir) | 上述地址关系在真实构造中得到保留 | 多 route 性能收益 |
| [`hbv-loop-mixed-radix-independence.mlir`](../test/Triton/hbv-loop-mixed-radix-independence.mlir) | 多轴 divisor、mixed-radix 坐标与 program 独立性的正反例 | 任意多轴 factor 组合都合法 |
| [`hbv-loop-runtime-mask-authority.mlir`](../test/Triton/hbv-loop-runtime-mask-authority.mlir) | runtime scalar mask 的来源和 binding 必须闭合 | 动态循环完整 main-tail/route 产品 |
| [`hbv-loop-body-semantics.mlir`](../test/Triton/hbv-loop-body-semantics.mlir) | runtime scalar 改变循环 body 时 Facts 会反映真实语义 | 性能模型能正确预测该变化 |
| [`test_control.py`](../python/test/unit/l_lite/test_control.py) | V1 候选图、factor 组合限制、原生全候选计时 facade 与 key cache | V2 全候选 binding 编译和 GPU artifact |

表中第三列很重要。测试证明范围必须由断言决定，不能因为文件名包含 `materialize`，就推导
整个 CandidateCompiler、PTX、正确性和性能都已闭环。

---

## 15. 评审时可以直接问的十个问题

1. Bridge 的独立性证明是否对每个读写地址和副作用都闭合？
2. 某条准入规则能否说明它保护的通用语义，而不是服务某个 kernel？
3. 只有 store、没有 load 的循环，是否只在真正需要 load 的软件流水 route 被拒绝？
4. 动态、嵌套和 state-axis 能力是否由结构证书表达，而不是算子名称？
5. Bridge factor 和 route factor 是否从 schema 到报告都保持不同含义？
6. 每个候选是否有唯一 PlanBundle、subject、binding 和 artifact identity？
7. Materialize 失败是否必然阻止候选进入 autotune？
8. 公开测试是否真的执行完整 pipeline，而不只是局部 Pass？
9. autotune 总开销是否包含失败候选的编译和验证成本？
10. L-lite 与主 L 是否使用同一个 CandidateCompiler，只在选择方式上分叉？

---

## 16. 主 L 项目与 L-lite 有什么关系

L-lite 与主 L 应共享：

- 相同的循环观察对象和 Provider 事实；
- 相同的 Bridge 和三条 route；
- 相同的 factor/subject 定义；
- 相同的合法性、物化、artifact 和正确性验证；
- 相同的完整候选编译器。

它们只在最后选择候选时分开：

```text
                                ┌─ L-lite：全部实测，原生 autotune 选最快
共享 CandidateCompiler ─────────┤
                                └─ 主 L：预测收益与风险，只发布安全候选
```

### 16.1 为什么需要主 L

exhaustive autotune 能找到快候选，但对大量候选逐一编译、运行、校验会花很长时间。对于
一次性或低复用 kernel，这些开销可能永远摊不回来。

主 L 的目标是只关注长期重复的 hot key，用可解释模型预测候选的相对收益和不确定性，
在安全下界与生命周期净价值都为正时才发布。

举例说，某候选让一次执行从 100 微秒降到 90 微秒，每次省 10 微秒；但生成和验证候选
花了 1 秒，那么至少要重复约十万次才能只抵消这 1 秒 acquisition，还没计算推理、缓存和
干扰成本。对于只运行十次的 kernel，即使候选本身快 10%，整个系统仍是负收益。

因此主 L 不追求“优化尽可能多的 kernel”，而追求“在极热 key 上，使用更少 acquisition
拿到安全的长期净收益”。L-lite 则故意支付完整 exhaustive acquisition，为主 L 提供同一
候选域下的性能上界对照。

### 16.2 主 L 不是“拿一个大黑盒直接猜时间”

它采用两段式思路：

```text
上游：解释每个 Pass 做了什么，以及这些变化怎样传播
后端：对难以完全白盒推导的寄存器分配、spill、调度等，用低容量统计代理
```

也就是“上游因果闭环 + 后端统计代理闭环”。后端确实复杂，但不能因此让统计模型吞掉
前面所有错误。

如果 Strong 把一个本应由 Bridge 造成的 program 数变化写错，后端回归模型也许能通过
kernel identity 或更多自由参数把训练误差降下来，但这种“准确”无法迁移，也无法解释。
最早责任原则要求先修 Strong，再评估后端代理。只有上游状态闭合后仍存在寄存器分配、
spill 或调度这类难以白盒推导的整体响应，才由粗粒度代理承担。

---

## 17. 主 L 的完整故事线（通俗版）

### 17.1 第一步：IR / Provider 事实

先从编译器当前能直接观察的 IR、输入 specialization 和显式硬件能力中提取事实，例如：

- 循环边界、trip、tail 和依赖；
- program 数与数据规模；
- dtype、输入输出字节和算术量；
- 明确暴露的架构容量。

每个字段都要有独立 producer。不能因为某个字段能降低训练误差就塞进来。

“独立 producer”是指这个字段能在不看真实性能结果的情况下，由 IR、specialization、目标
描述或已定义编译阶段直接计算。例如 dtype 来自 IR 类型，trip 来自循环边界或 runtime
binding，目标 shared-memory 容量来自明确硬件接口。

温度、功率虽然也能观测，却没有直接描述 Pass 机制的因果意义。它们适合用来判定这次性能
测量是否稳定，不适合进入 Provider→Strong→Weak→时间预测主链。否则同一候选仅因机器当时
更热就得到不同“机制预测”。

Provider 的输出不应只是散落字段，而应附 schema、单位、来源阶段、适用 subject 和证书
状态。一个字段如果只能在 ptxas 或真实运行后得到，就必须标明它不是 pre-decision 输入，
不能偷渡进发布前模型。

Provider 失败时不能由 Weak 插值补齐。字段缺失、歧义或 producer 版本不匹配，应在最前层
停止对应候选，或返回外层循环重新定义可观察对象。

### 17.2 第二步：强语义状态

强语义只记录当前 Pass 直接造成的变化：

| 因果域 | 直接变化 |
|---|---|
| Bridge | logical/physical program 关系与新循环 |
| 软件流水 | stage 请求、live loop 与跨迭代服务结构 |
| 完全展开+重排 | main/tail、操作组与重排终态 |
| 完全展开+向量化 | logical group、vector container 与子型终态 |

寄存器数、指令数和执行时间可能随后变化，但通常是连锁反应或后端响应，不应一股脑写进
强语义。

强语义的输入是 Pass 前 Provider 状态和一个明确干预；输出是“如果这个 Pass 按计划成功，
直接改变了哪些结构”。例如 Bridge 直接改变 logical/physical program 映射和是否存在新
loop；它不直接决定最终寄存器数。向量化直接改变 logical group 与宽容器；它不直接声称
GPU 一定使用哪条机器指令。

为什么要限制这么窄？因为论文要回答因果方向。如果把 Pass 后所有相关字段都称为 Strong，
就会把中间变量、后端选择和最终结果混在“原因”里，无法解释模型在新架构为何失效。

Strong 的验收不是时间预测准，而是与 Pass 前后 IR/Provider 事实逐字段对应：干预应该改变
的字段都改变，不应该直接拥有的字段没有被错误纳入。若某种 IR 结构反复出现 Strong 终态
不一致，应回到因果域或子域重推，而不是增加时间残差。

### 17.3 第三步：弱语义传播

弱语义负责把两类信息传到完整预编译终态：

1. 强语义中受 Pass 改变的核心属性；
2. 虽不受当前 Pass 改变、但时间预测需要的可识别环境属性。

第二类可能包括 dtype、元素宽度、输入输出字节、算术量、trip/tail 和显式架构能力。
这些主要环境因素必须尽量显式传播，不能全部扔给环境残差。

可以把 Weak 理解为“从局部施工图推导整栋建筑施工后的可观察状态”。Strong 只说某堵墙
被移动；Weak 还要传播因此改变的通道长度、并保留没有被这次施工改变但仍决定使用体验的
楼层和人数信息。

Weak 主体映射可以包含学习机制，但每条输入边都要说明其来源和方向。可确定的代数关系应
直接计算，例如 logical program 数除以 Bridge factor 得到 physical program 数；需要从
历史编译代际学习的映射则使用低容量、延迟更新机制，并报告可识别性。

Weak 输出构成“完整预编译终态”，供后端代理使用。若后端代理需要一个字段，而 Weak 没有
producer，正确做法是回到 Provider/Weak 增加可解释传播，或承认该字段不可获得；不能在
代理里用 kernel 名替代。

### 17.4 第四步：跨代学习残差

学习残差只修正同一因果分支跨编译版本或同类目标的小幅系数漂移。它不能：

- 修复上游字段缺失；
- 吸收因果分支翻转；
- 读取同一代真实性能再修正同一代发布；
- 变成比主体模型更大的另一个黑盒。

主体代理定义发生变化时，旧残差应清零重新开始。

“跨代”意味着第 g 代执行后得到的偏差，只能帮助第 g+1 代或更后版本。否则同一候选先运行
得到真实时间，再用这个时间“预测”自己，形式上误差很小，实际上已经变成查表，不是预执行
模型。

在同一架构上，机制分支相同但系数有小漂移，可以由学习残差调整。例如编译器升级后某个
typed-PTX 统计量与后端响应的比例轻微变化。若符号翻转、原本增加某字段会加速而新架构上
变为减速，这属于模型有效性风险，不是“小残差”。

残差是否仍“小”要用独立数据判断：均值接近零、容量低于主体模型、没有稳定 subgroup
bias，且校准后能用较窄区间覆盖。若残差承担了主要预测幅度，它已经不是残差，外层循环
必须重审主体映射。

### 17.5 第五步：后端统计代理

寄存器分配、spill、instruction scheduling 和 issue overlap 很难从 TTIR 精确白盒推导。
主 L 在上游闭合后，用较粗的 typed PTX/live-shape 状态预测这一整块后端响应。

统计代理可以是回归模型，但要满足：

- 输入在真实执行前可获得；
- 不读取 kernel/family identity；
- 不读取温度、功率、测量顺序或同代真实时间；
- 在整 family 隔离的 holdout 上稳定；
- 在精度同样可接受时使用最少字段、交互和模型容量。

代理的输入分两层。P0 是机制核心，例如 program 数、字节、算术量、trip/tail、stage/group
和显式目标能力。P1 是 candidate 编译后、真实执行前可得的 typed PTX/live-shape，例如粗
指令类别、memory instruction 数、peak typed live slots、live-area 和 O→B/B→C 轴向比值。

P1 不是因为“更接近后端”就自动允许。它必须在 paired、whole-family bootstrap 中，相对
P0 在独立外层 fold 显著降低发布风险，才进入模型。若 P0 与 P1 都在可接受误差内，则选择
P0，因为字段更少、迁移更容易。

为什么采用“粗粒度整体响应”而不继续精确预测每个寄存器和 spill？因为后端分配与调度有
大量隐含交互，最细颗粒度可能依赖无法观测状态。只要粗代理拥有可解释输入边界、在独立
family 上稳定，并能支撑发布决策，就应通过深度停止门，不无限复制后端。

反之，如果某个 route factor subgroup 持续出现符号错误，不能简单给 RBF 多加中心。外层
循环要检查参数集合、交互图、因果域划分和模型颗粒度是否正确。

### 17.6 第六步：为什么需要 O/B/C

每个联合候选保留三个真实 artifact：

```text
O = Original
B = 只做 Bridge
C = Bridge 后再做一条 Route
```

于是总收益可拆成：

```text
Bridge 收益：      log(T_O / T_B)
Route 条件收益：   log(T_B / T_C)
总收益：           log(T_O / T_C)
总收益 = Bridge 收益 + Route 条件收益
```

这样 Bridge 模型只解释 Bridge，route 模型只解释“给定 Bridge 终态后 route 又做了什么”。
若只看 O→C，就会把两种机制混在一起。

最终是四个组件：一个 Bridge 组件，加软件流水、重排、向量化三个 route-local 组件。

O、B、C 必须由同一个 CandidateCompiler、同一 specialization 和兼容的 artifact identity
产生。不能拿旧版本 Original、当前 Bridge 和另一编译器版本 Candidate 拼在一起，否则
差值同时包含编译器代际变化。

正式 timing 时，O/B/C 还应处于同一个原子环境角色中。若 O 在空闲 GPU 测、B/C 在受干扰
GPU 测，代数公式仍成立，但输入时间不再代表同一环境下的组件收益。

Bridge factor 1 是特殊但重要的恒等情况。B 不应重新独立编译出一个有随机差异的“近似 O”
再做两次 noisy timing，而应通过 artifact identity 证明 B=O，并令 Bridge component 精确为
零。这既减少噪声，也检验分解是否尊重机制语义。

### 17.7 第七步：残差、区间与发布

模型中心先通过独立 qualification。之后才允许学习小幅时间残差、环境 common-mode
漂移和机制区间。发布看的是总收益安全下界，不是中心点：

```text
lower_bound(total benefit) > 0
```

还必须考虑这个 hot key 未来会复用多少次，能否摊销候选编译、验证、推理、存储和执行
干扰。冷 key 直接保留 Original。

区间不是在中心模型失败后任意加宽到“终于覆盖”。先在 qualification 上确认中心方向和
subgroup 无系统偏差，再在独立 calibration 上估计 q80/q95 等机制半径。若某 subgroup
中心持续偏向一边，它说明模型 owner 仍有问题，不属于对称尾部不确定性。

发布还要防 false adoption：模型安全下界判断候选更快，实际却比 Original 慢。这类错误
比“错过一个本来更快的候选”更危险，因为它主动造成线上退化。门槛、区间和 hot-key
价值应优先控制 false adoption，同时报告保守拒绝损失。

生命周期价值至少包含：预计复用次数乘单次节省，减候选编译、correctness、必要测量、
代理推理、cache/storage 和干扰成本。只有该净值的安全下界为正，才发布一个候选；同一个
hot key 最多发布一个收益下界最高的 route/factor。

### 17.8 第八步：真实执行反馈

正式运行可保存 key、Plan/artifact identity、环境有效性和真实执行时间。这种存储开销
通常很小，值得保留。但本次执行结果只能更新下一代残差或 exact lookup，不能反过来
证明本次发布原本就是正确的。

建议快照在候选完成编译、即将执行以及执行完成时形成闭合记录，至少包含 specialization
key、候选 ref、Plan hash、TTIR/PTX/cubin hash、目标 UUID/架构、环境 validity、预测中心/
区间、发布决定和 `T_execution`。如果运行失败或环境中途失效，也要保存 disposition。

exact measured lookup 只服务完全相同、identity 闭合的 hot key。它能让后续重复执行直接
复用真实结果，却不能把一个 key 的实测推广给形状相似但 identity 不同的新 key。新 key
仍需通过模型与安全门。

---

## 18. 主 L 为什么要按因果域分别建模

Bridge、软件流水、重排和向量化改变代码的方式不同。如果强行让它们共享同一个回归
结构，可能为了统一而删掉某个因果域真正需要的状态，也可能让一个域的参数去修补另一
个域。

正确顺序是：

1. 每个域先用自己的最小充分状态闭环；
2. 允许轻量复用相同的归一化、特征变换或低容量算法；
3. 各域独立成功后，再研究哪些组件可以有条件地部分共享；
4. 只有新共享模型在独立数据上不降低归因和发布能力，才替代域内模型。

“参数最少”也不是单纯比数字少。若两个模型的误差、方向、区间覆盖和 false adoption
都在可接受范围内，才依次选择：

```text
更少 causal fields
  → 更少 interaction blocks
  → 更低模型自由度
  → 更浅的后端解释深度
```

### 18.1 Bridge 组件具体预测什么

Bridge 组件预测 O→B：program ownership 合并、physical program 数、Bridge loop 与 program
footprint 变化怎样影响时间。它可以读取 O 与 B 的命名状态，但不应拥有某条 route 的 stage、
vector subtype 或 phase postcondition。

如果 Bridge factor 增大后，有些 family 加速、有些 family 减速，模型要检查字节、算术量、
program 数、trip 与 typed backend response 的交互，而不是为每个 family 建 indicator。Bridge
factor 1 提供精确零点，用于检查组件是否产生无意义偏置。

### 18.2 软件流水组件具体预测什么

它预测给定 B 后，stage 与跨迭代 load/compute service 如何影响 B→C。核心状态应包含 live
loop、pipeline-capable load service、payload、依赖、stage 请求及必要 typed PTX response。

若某个循环没有 pipeline service，它应在合法性/物化层被拒绝，而不是让时间模型学习“此类
候选通常没收益”。模型只在已合法、真实物化的软件流水 population 内工作。

### 18.3 重排组件具体预测什么

它预测 main group 展开、operation phase 调整、tail 与 live-range 后端响应对 B→C 的影响。
它不能假设一定有 load/store，也不能用某个算子类别作为输入。操作组数量、依赖边、main/
tail 和 typed live-shape 是更接近机制的候选字段。

### 18.4 向量化组件具体预测什么

它预测 exact packing、logical lane/container、tail、subtype 和 state-axis 结构对 B→C 的影响。
load packing、store packing、一般 operation packing、exact-prefix 与 state-axis 可以作为同一
因果域内的显式子型；若独立证据显示它们的后端响应无法由同一状态/交互闭合，再在域内拆分，
而不是一开始按 workload 建多个模型。

### 18.5 什么时候允许部分共享

假设四个组件都需要 bytes、arithmetic elements 和 trip normalization，可以复用同一字段
producer 和归一化代码；这叫轻量共享。若进一步发现 Bridge 与 route 的部分响应在条件变量
给定后使用相同函数形式，可以预注册一个 shared block，再在 whole-family holdout 上与四个
独立模型成对比较。

只有共享模型不增加任何域的系统偏差、区间宽度或 false adoption，并降低总容量与迁移成本，
才替代独立 block。不能因为代码更整齐，就牺牲某一域的最小因果闭环。

---

## 19. 内层循环和外层循环怎样避免越修越错

### 19.1 内层循环：修实现，不改科学问题

内层只修已经明确属于某一层的实现错误，例如：

- schema 序列化错误；
- 证书公式的代码写错；
- 一个注册 proxy block 的数值实现错误；
- fold-local normalization 或容量门的 bug。

内层禁止为了修一个失败样本加入 kernel 名、shape identity、温度、功率、测量顺序或
same-generation outcome。这类局部补丁会让全局因果链越来越不可信。

内层修复必须带回归测试，证明错误公式或接线已经修正，同时未改变 population、字段定义、
holdout 角色和发布阈值。例如修复序列化漏写 `tail_extent`，可以重跑 CPU/MLIR 测试；它不
授权重新选择模型中心或查看 sealed 标签。

若修复过程中发现必须新增一个从未注册的模型字段，那已经不是内层实现错误，而是科学状态
定义变化，应停止内层并进入外层回退。这个边界防止“先加字段把测试修绿，再补理论”。

### 19.2 外层循环：回到最早出错的位置重新推导

如果某个 subgroup 出现系统性偏差，按最早责任回退：

```text
事实不对               → 回 Provider
Pass 直接变化描述不对  → 回 Strong
终态传播不对           → 回 Weak
对象放错因果域         → 重做 domain partition
字段或交互不足         → 重审 backend proxy state
同一状态仍不稳定       → 重审模型族或颗粒度
仅在独立校准出现尾部   → 才讨论 residual / interval
测量环境无效           → 整个角色作废重测
```

核心原则是：宁可暂时不给发布证书，也不把无法解释的变量塞进链条换取表面精度。

外层每次重推需要冻结一份失败证据：哪个独立 family/subgroup 失败、失败方向和幅度、为何
排除测量污染、当前最早 owner 为什么无法解释。然后只从该 owner 重新生成下游状态，旧的
Weak、代理、残差与区间不再拥有 authority。

例如 vector factor 8 在所有高 tail-fraction family 上系统性反转。先核查 Facts 的 tail、
Strong 的 main/tail 终态和 Weak 传播。如果都正确，再审查向量化代理是否缺少 tail
interaction；只有这些闭合后仍是无方向尾部，才进入区间。

计划何时升版也服从相同原则：同一科学假设、population 和证据角色内的预期失败留在一个
计划内循环；只有已执行不可变证据会被新方法破坏，或出现双 writer/authority 污染，才提纯
新版本。新版引用旧版冻结证据，不复制两套可执行 authority。

---

## 20. 环境、残差、查表和迁移风险不能混在一起

| 对象 | 它负责什么 | 它不能替谁背锅 |
|---|---|---|
| 弱语义学习残差 | 同一因果映射跨代的小漂移 | 因果翻转、字段缺失 |
| domain time residual | 合格分域代理剩余的小时间偏差 | 系统性 subgroup 错误 |
| environment residual | 主要环境因素已传播后的小 common-mode 漂移 | 温度/功率特征、外部干扰或无效测量 |
| mechanism radius | 当前架构、当前域内不可消除尾部的覆盖 | 架构迁移或错误模型 |
| exact measured lookup | 同一 hot key 实测后的复用 | 未见 key 的泛化证明 |
| architecture migration risk | 新架构上因果边可能翻转的有效性风险 | 当前架构发布区间 |

当前架构的模型内不确定性可以进入残差或机制半径；跨架构后因果边可能翻转的风险只进入
独立迁移报告。两者不能相加、互相吸收，也不能用迁移风险扩大当前发布区间。

温度、功率和 utilization 可以用于判断测试环境有效不有效，但不能作为因果模型输入。
否则模型可能学到“这次机器恰好热了”，而不是代码变换的可迁移机制。

### 20.1 弱语义学习残差的例子

同一 typed PTX/live-shape 状态在编译器小版本升级后，主体映射系数轻微漂移；可以用上一代
独立反馈估计低容量 correction。若需要为某个 kernel 单独保存修正，它已接近 exact lookup，
不能继续叫通用学习残差。

### 20.2 domain time residual 的例子

向量化域中心模型在各 subgroup 无系统方向错误，但整体仍有均值接近零的小幅时间差，可以
在 fresh calibration 上拟合窄修正。若所有 factor 8 都被低估 10%，这是 factor-local 机制
缺失，应回代理，不是 time residual。

### 20.3 environment residual 的例子

主要可识别环境状态已经显式传播后，同一有效测量环境可能仍有很小 common-mode 漂移。
environment residual 可以估计这个整体偏移，但前提是测量 protocol 已排除外部进程和明显
降频。无法证明独占的数据不能靠 residual“校正成有效”。

### 20.4 mechanism radius 的例子

在同一合格模型与 population 内，少量 family 受无法进一步分辨的后端临界分配影响，误差
呈无稳定方向尾部。可在整 family 校准 q80/q95 半径，用于保守发布。若扩大半径后几乎所有
候选都无法发布，结论是当前模型没有足够发布能力，而不是“覆盖率终于达标”。

### 20.5 exact lookup 的例子

同一个 specialization 和 artifact 已正式运行多次，可保存其稳定实测时间，下次直接复用。
它非常适合极热重复 key；但换 shape、compiler commit 或 artifact hash 后 identity 已变，旧
值不能无证明复用。

### 20.6 architecture migration risk 的例子

当前架构上 peak live slots 增加通常导致某种后端响应，新架构上寄存器文件、分配器或指令
规则改变后，这条边可能符号翻转。迁移风险报告要给出该边失效会增加多少误差、可能造成
多少 false adoption，并让使用者选择保守使用或在新架构重拟合。当前阶段它不扩大当前架构
误差半径，也不自动触发重开发。

---

## 21. 什么情况下可以说“HBV 服务边界之外”

不能因为后端复杂、问题难找，就直接宣布“不可观测”。合法出口要求前面每个 in-scope
模块先完成自己的责任：

```text
Provider 事实可信
  → 强语义可信
  → 弱语义传播可信
  → 后端粗代理在独立 family 上可信
  → 小残差和区间可信
```

之后仍存在的误差，若有足够证据表明来自无法暴露的后端分配、不可观测架构状态或无法
保证的外部测量条件，才可归入 HBV 服务边界之外。

最终还要生成“工程成功版 → 论文证书版”的变量剥离账本，逐项说明：

- 旧工程为什么使用这个变量；
- 它看起来带来多少精度或收益；
- 为什么没有独立因果 producer；
- 剥离后哪些 key、候选或收益消失；
- 为什么不能由学习残差、时间残差、环境残差、机制半径或查表吸收；
- 最终是修复上游、保守排除，还是归入 HBV 服务边界之外。

这个账本是最终审计报告，不是为了控制内外循环而动态调参。

### 21.1 可以归到边界之外的例子

固定架构、编译版本和可观察 typed PTX 状态后，后端在寄存器分配临界点仍可能选择激进高
寄存器优化或产生真实容量压力，而公开编译接口没有提供足够状态区分两者。若上游、粗代理、
独立校准和环境协议都已闭合，且进一步拆解需要不可获得的后端内部状态，这部分可作为服务
边界证据。

另一个例子是外部使用者无法控制的 GPU 干扰。HBV 可以检查 NVML context 和 stationarity，
但不能保证系统中所有未暴露干扰都消失。检测失败的角色应作废；无法获得合格环境时，正确
结论是“本次没有性能 authority”，不是预测一个校正值。

### 21.2 不能归到边界之外的例子

- Facts 把实际 tail 记错；
- Strong 漏掉 route 直接改变的状态；
- Weak 没传播时间模型明确需要且可观察的 dtype 或 bytes；
- 某 factor subgroup 系统性偏差却没有重审交互；
- 使用受污染 timing 得到不稳定结果；
- 只因为后端复杂，就未尝试较粗、可独立验证的统计闭环。

这些都仍在 HBV 自己的责任范围。只有证明每个模块完成自身职责，尾部归因才有论文可信度。

### 21.3 变量剥离为什么不能简单算成残差

如果某个旧变量没有独立 producer，例如 kernel 名或测量顺序，它可能非常擅长记住开发集，
剥离后也可能让收益明显下降。但残差与半径只能描述合格模型的小尾部，不能给没有因果身份
的变量换一个名字继续使用。

账本必须量化剥离前后的逐 key 决策变化：哪些候选原先被发布、剥离后为何转为 Original，
收益损失是来自区间变宽、中心方向变化、population 排除还是 acquisition 不再可摊销。这样
评审者可以判断收益减少换来了什么可信度，而不是只看到一个更小总加速比。

---

## 22. L-lite 与主 L 的公平对照标准

| 阶段 | L-lite | 主 L | 必须相同吗 |
|---|---|---|---|
| 循环 census / Provider | 使用 | 使用 | 是 |
| factor / subject 定义 | 使用 | 使用 | 是 |
| CandidateCompiler | 使用 | 使用 | 是 |
| 合法性、物化、artifact、正确性 | 使用 | 使用 | 是 |
| 请求候选域 | 完整枚举 | 从同一完整域做安全选择 | 能力面必须相同 |
| 候选选择 | 全部真实计时 | 分域预测与安全下界 | 不同 |
| 残差、区间 | 不需要 | 需要 | 不共享 |
| 生命周期发布门 | 只做对照报告 | 是生产 authority | 不共享 |

如果 L-lite 和主 L 使用不同物化器，最后比较“autotune 与 HBV 谁好”没有意义，因为两边
根本不是在同一候选空间比赛。后续新增通用循环对象或 route 能力时，应进入一个共享
CandidateCompiler，再同时被两种选择路径使用。

### 22.1 “候选域相同”具体怎样证明

两边应从同一 census hash、factor domain 和 composition graph 开始。每个 candidate ref 的
PlanBundle、编译 commit、TTIR/PTX/cubin hash、correctness disposition 都来自同一个只读
CandidateCompiler 产物。L-lite 将所有 correct binding 交给 autotune；主 L 对同一列表做
预测与发布门。报告应能逐 candidate ref 对齐，而不是靠 route 名近似匹配。

### 22.2 开销怎样公平计算

L-lite 报告完整候选编译、全部 correctness 和全部 benchmark；主 L 报告共享候选生成中实际
支付的部分、proxy inference、必要校准、缓存和运行反馈。若共享 CandidateCompiler 当前仍
生成全部候选，主 L 不能假装没有这部分编译开销；只有未来真正实现 lazy/selected compile，
才能在实测账本中减少。

失败候选成本也要对齐。L-lite 的非法 cell 不上 GPU，但其 census、证明和编译尝试时间仍在
acquisition 中；主 L 若使用相同合法性层，也应报告自己实际支付的对应成本。

### 22.3 结果怎样比较

至少比较四项：最终 winner/发布候选的性能收益、总 acquisition、break-even reuse count、
false adoption 与保守错过。只比较最终加速比会忽略主 L 的核心价值；只比较选择耗时又可能
掩盖 HBV 选择了较慢候选。

建议还报告“L-lite winner 是否在主 L 安全候选集中”。若不在，应给出主 L 拒绝原因和因此
损失的收益；若主 L 选择不同候选，报告二者单次差距与生命周期净值差距。

---

## 23. 当前项目风险与整改优先级

### P0：成为完整公开对照前必须闭环

1. 提供唯一的 `candidate → PlanBundle → binding` 编译入口；
2. 将 V2 候选图接入 native autotune；
3. 为三条 route 增加完整 pipeline 正反测试和 silent-fallback 测试；
4. 增加统一 TTIR/PTX/cubin identity 与 GPU correctness harness；
5. 把所有失败候选的编译、验证和测试成本纳入 acquisition。

P0 的验收应是一个公开 demo：输入普通 Triton kernel 与 factor domain，不使用外部私有
脚本，得到完整请求图；每个 cell 有唯一 disposition；所有 executable cell 有 artifact 与
correctness 证据；原生 autotune 返回 winner；报告总成本。只完成其中一个 API 类或单个
MLIR 测试不能关闭 P0。

五项之间有依赖顺序。没有唯一编译入口，就无法保证 V2 图每个 cell 使用同一 pipeline；没有
完整 route 测试，就无法相信 artifact checker 的正反边界；没有 correctness，autotune winner
可能是错误程序；没有 acquisition ledger，对照结论又无法评价搜索开销。

### P1：论文可信度与可维护性

1. 将超大的 `HBVLoop.cpp` 按 Facts、Bridge、Plan、Phase、Logical、Validate 拆分；
2. 将 active adapter registry 与历史 evidence reader 物理分层；
3. 为每个 active adapter 建立 schema→Provider proof→materializer→postcondition 注册表；
4. 把窄历史名称改为纯结构语义；
5. 尽量由同一 schema 生成 Python contract 与 C++ parser，减少版本漂移。

P1 不只是整理代码。拆分后要用 dependency/authority 测试保证 Facts、Decision、Materialize
和 Validate 的方向不倒置；active registry 明确列出哪些 adapter 能执行；历史 adapter 只能
读取旧证据；生成式 schema round-trip 覆盖缺字段、多字段、版本不匹配和 hash 稳定性。

每个 active adapter 注册项应像一张责任卡：它消费哪些 Provider 字段、证明哪个结构、调用
哪个 materializer、期待什么 postcondition、由哪些正反测试覆盖。这样评审者不需要在
10,000 行文件中猜 parser、decision 和 materializer 是否指向同一版本。

### P2：使用体验

1. 提供最小公开 demo 和一条命令完成候选编译与 autotune；
2. 自动输出每个 observable loop 的三 route 覆盖矩阵；
3. 把类型化拒绝翻译成人类可读报告；
4. 在 CI 中增加文档链接和完整 pipeline 检查。

P2 的目标是让外部使用者不需要阅读全部 C++ 才知道候选为什么失败。覆盖矩阵应按 subject
列出软件流水、重排和向量化的可观察、合法、物化、正确性状态；拒绝报告使用稳定错误码和
中文/英文说明；demo 命令固定编译与环境信息，便于评审者复现。

CI 除了检查代码能构建，还应验证文档中的相对路径存在、公开示例真的可运行、每条 active
route 至少有完整 pipeline 正例，以及默认 Triton 在未 opt-in 时行为不变。

---

## 24. 明天评审可以采用的验收清单

### 24.1 架构

- [ ] Bridge 与 Route 是两个顺序独立干预；
- [ ] 分开归因，但以笛卡尔积联合选优；
- [ ] exact-prefix 明确属于完全展开+向量化子型；
- [ ] 嵌套循环只选择一条 route、一个 factor 和一个结构 scope；
- [ ] 软件流水物理 lowering 明确复用原生 Triton；
- [ ] active 准入不读取 workload identity。

### 24.2 编译与正确性

- [ ] 每个候选有唯一 subject、PlanBundle、binding 和 artifact identity；
- [ ] 完整 pipeline 顺序唯一；
- [ ] 每条 route 有正例、最早层拒绝反例和防回退反例；
- [ ] validator 失败会阻止候选进入 autotune；
- [ ] Original 是唯一无干预基线；
- [ ] correctness 与性能测试完全分离。

### 24.3 公开可复现性

- [ ] 普通 kernel 能由公开 API 生成完整候选 bindings；
- [ ] V2 图已经接入原生 autotune；
- [ ] 公开测试覆盖完整七段 pipeline；
- [ ] GPU correctness 和 artifact identity 可独立复现；
- [ ] 性能报告包含完整环境证据和全部 acquisition。

如果前三组仍有未勾选项，应把它明确列为后续工作，不能用“已有 C++ 代码”代替产品
闭环。

### 24.4 每个勾怎样才可以真正打上

“Bridge 与 Route 独立”需要 schema、Python contract、C++ Decision 和 O/B/C artifact 四层
都保存两次干预，不能只靠架构图。笛卡尔积联合选优则要在请求账本中看到每个合法
`bridge_factor × route × route_factor` cell，不能只列论文公式。

“exact-prefix 属于向量化子型”需要 ontology 中 mechanism route 映射、Plan parser、
materializer 和报告一致；如果 artifact 层仍把它记成第四条独立 route，也不能勾。

“软件流水复用原生 Triton”需要 L-lite 只设置和验证 subject/stage 请求，同时在后端测试中
观察原生 pipeliner 产生结果。若 L-lite 又维护一套私有流水 lowering，架构责任已变化。

“不读取 workload identity”需要代码搜索、active registry 审计和新 family 负例共同证明。
仅删除显式 kernel 名还不够；shape hash、源码路径或特定操作数量组合也可能成为隐性 identity。

“每个候选 identity 唯一”需要从 composition ref 到 Plan、binding、artifact、correctness、
timing 的 join 检查，每一层恰好一行。缺行、多行或 hash 冲突都不能勾。

“完整 pipeline 顺序唯一”需要 CandidateCompiler 成为唯一入口，并测试跳过、交换或重复某个
Pass 会 fail-closed。文档建议顺序但允许外部任意拼装，还不能勾。

“公开可复现”要求一名没有私有仓库和实验账本的评审者，从公开 commit、文档命令和标准
依赖出发得到相同 disposition。仅作者机器上有结果、但缺脚本或输入 identity，不能勾。

### 24.5 验收状态应该怎样报告

建议每个勾选项使用四态，而不是简单“完成/未完成”：

| 状态 | 含义 |
|---|---|
| `PROVED` | 有与要求同范围的公开证据，且复核通过 |
| `PARTIAL` | 有实现或局部测试，但证据范围小于要求 |
| `CONTRADICTED` | 当前证据明确显示要求不成立 |
| `MISSING` | 尚无足以判断的证据 |

例如当前“Pass 是真实 C++/MLIR 实现”可以是 `PROVED`；“普通 kernel 可由公开 API 生成所有
bindings”是 `MISSING/PARTIAL`；“公开测试证明所有 route 端到端性能收益”不能因为没有失败
日志就标 `PROVED`，而应是 `MISSING`。

---

## 25. 最终评审结论

L-lite 的核心方向是合理的：

- Bridge 与三条 route 没有混成无法归因的 mega-pass；
- 分开归因的同时仍通过笛卡尔积做联合优化；
- factor、subject、subtype 和 lineage 有明确身份；
- 变换采用真实编译器 Pass；
- 合法性、物化、正确性和性能的责任被分开；
- 原生 unroll、软件流水和 autotune 得到复用；
- active 规则的目标是依赖结构与代码语义，而不是 benchmark 定制。

当前公开版本的准确定位是：

> **一个已经接入 Triton、具备主要循环发现/证明/物化源码和候选控制框架的编译器能力
> 面；尚缺少从普通 kernel 到完整候选、正确性验证和 exhaustive winner 的统一公开
> 产品闭环。**

因此明天评审不应只问“代码有多少”或“autotune 能不能选最快”，而应重点确认：

1. 每项规则是否真正通用并可解释；
2. 每个候选是否真实物化且没有回退；
3. 公开端到端 CandidateCompiler 怎样闭合；
4. L-lite 与主 L 如何保证使用完全相同的候选能力；
5. 哪些结论已经有公开证据，哪些仍是待完成计划。

只要这五点诚实、可复核，L-lite 就能成为有价值的实验对照：它证明“付出完整 exhaustive
acquisition 时可以得到什么”，主 L 再证明“能否用更少时间获得接近的收益，并保持论文
级归因与安全发布”。

---

## 附录 A：术语表

| 术语 | 白话解释 |
|---|---|
| Kernel | 一项交给 GPU 执行的完整函数/任务 |
| IR / TTIR | 编译器处理中间阶段的程序表示，可理解为比源码更规则的内部代码 |
| Pass | 编译器按固定规则执行的一次分析或改写 |
| Program | Triton 的一个并行执行单位，不等同于整个 kernel |
| Loop Bridge | 把若干 logical program 工作构造成一个显式循环的前置变换 |
| Route | Bridge 后对循环采用的一条优化路径 |
| Factor | 某种机制一次处理的规模；不同机制含义不同 |
| Subject | 某次 Pass 真正作用的那个循环或结构对象 |
| Provider | 提供可直接复核事实的上游模块 |
| Strong / 强语义 | 只描述 Pass 直接造成的结构变化 |
| Weak / 弱语义 | 把强语义和核心环境属性传播到完整终态 |
| Lineage | 记录一个操作、循环或候选从哪里来的来源链 |
| PlanBundle | 某个候选交给编译器的闭合计划 |
| Materialization / 物化 | 编译器真实生成目标代码结构 |
| Postcondition | 物化完成后必须成立的事实 |
| Artifact | 编译产生的 TTIR、PTX、cubin 等具体产物 |
| Silent fallback | 候选失败后悄悄退回 Original，却仍冒充成功 |
| Autotune | 真实运行多个候选并选择最快者 |
| Acquisition | 为编译、验证、测量和选择候选付出的全部前期开销 |
| Original | 没有应用 L-lite 干预的基线版本 |
| O/B/C | Original、Bridge-only、Bridge+Route 三个顺序 artifact |
| Residual / 残差 | 主体模型合格后仍剩下的小幅偏差 |
| Error radius / 误差半径 | 对当前域内不可消除尾部不确定性的覆盖 |
| Holdout | 完全不参与模型修改、只用于独立验收的数据 |
| Hot key | 会长期、高频重复出现，值得摊销优化开销的 kernel specialization |

---

## 附录 B：建议的代码阅读顺序

非编译器读者可以先读前五项；工程师再继续读 C++ Pass。

1. 本文第 2～4 节：先理解 Bridge、三条 route 和五层责任；
2. [`factor_ontology.py`](../python/triton/l_lite/factor_ontology.py#L1)：理解名词；
3. [`contract.py`](../python/triton/l_lite/contract.py#L1)：理解组合合法性；
4. [`composition.py`](../python/triton/l_lite/composition.py#L1)：理解候选图和 identity；
5. [`autotune.py`](../python/triton/l_lite/autotune.py#L453-L616)：理解原生适配边界；
6. [`Passes.td`](../include/triton/Dialect/Triton/Transforms/Passes.td#L93-L125)：看 Pass 注册；
7. [`HBVLoop.cpp` 常量与 Plan](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L44-L203)；
8. [`LoopBridgeDiscoverPass`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L4993)；
9. [`HBVLoopFactsPass`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L5115)；
10. [`LoopBridgeProgramCoarseningPass`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L6191)；
11. [`HBVLoopDecisionPass`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L7210)；
12. [`LoopUnroll.cpp`](../lib/Dialect/Triton/Transforms/LoopUnroll.cpp#L24-L195)；
13. [`HBVLoopMaterializePass`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L10108)；
14. [`HBVValidateLoopPlanPass`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp#L10496)。

---

## 附录 C：公开实现快照的文件审计

可用以下命令复核 L-lite 实现快照相对原生 Triton 的变化：

```bash
git diff --name-status 7c56a5e40..25bca1f4e
git diff --stat 7c56a5e40..25bca1f4e
git diff 7c56a5e40..25bca1f4e -- \
  include/triton/Dialect/Triton/Transforms/Passes.td \
  lib/Dialect/Triton/Transforms/CMakeLists.txt \
  lib/Dialect/Triton/Transforms/LoopUnroll.cpp \
  python/src/passes.cc
```

实现快照包含 27 个文件级改动：4 个原生文件修改、6 个主要新增实现文件、13 个 MLIR
测试、1 个 Python 测试文件及文档/README 改动。本文是该快照之后的评审交付文档，不
改变实现本身。

---

## 附录 D：27 个实现快照文件逐项说明

### D.1 用户入口与说明文件

- [`README.md`](../README.md)：增加 L-lite 分支的定位、构建和公开能力入口。它是导航，
  不承担编译器证明；README 中的任何“支持”仍要回到源码和测试核验。
- [`l-lite-capability-validation-2026-08-25.md`](l-lite-capability-validation-2026-08-25.md)：
  保存公开分支的构建、CPU/MLIR 测试结果与当时的验证边界。它是一次验证记录，不是后续
  commit 永久有效的通行证。
- [`l-lite-understanding-guide.zh-CN.md`](l-lite-understanding-guide.zh-CN.md)：按因果域介绍
  Bridge 与 route，并给学习者提供源码索引。本文比它更偏完整架构评审和主 L 对照。

### D.2 编译器注册与实现文件

- [`Passes.td`](../include/triton/Dialect/Triton/Transforms/Passes.td)：注册 Discover、Facts、
  Decision、Bridge 构造、Materialize 和 Validate 六个新增 Pass。
- [`CMakeLists.txt`](../lib/Dialect/Triton/Transforms/CMakeLists.txt)：把 `HBVLoop.cpp` 加入
  TritonTransforms 构建，保证安装产物真实含有实现。
- [`HBVLoop.cpp`](../lib/Dialect/Triton/Transforms/HBVLoop.cpp)：集中实现 schema/parser、
  Provider 事实、Bridge 证明与构造、三 route 决策/物化和最终 validator；也是当前最主要
  的拆分与 authority 审计对象。
- [`LoopUnroll.cpp`](../lib/Dialect/Triton/Transforms/LoopUnroll.cpp)：在复用原生 unroll 的
  前提下增加 source trip、main/tail、clone role 和 operation-group lineage。
- [`passes.cc`](../python/src/passes.cc)：将六个新增 C++ Pass 导出给 Python PassManager；只
  提供显式调用能力，不等于自动注入默认 pipeline。

### D.3 Python 候选控制文件

- [`__init__.py`](../python/triton/l_lite/__init__.py)：汇总 L-lite 对外可见的 Python 类型和
  构造函数；它定义 API 面，不拥有具体编译行为。
- [`factor_ontology.py`](../python/triton/l_lite/factor_ontology.py)：定义 factor、subject、
  subtype、nested scope、stable ref 和 route-local admission。
- [`contract.py`](../python/triton/l_lite/contract.py)：把 Bridge 后 subject 与 route factor
  组合成 V1/V2 合法性合同，保存两次干预的独立身份。
- [`composition.py`](../python/triton/l_lite/composition.py)：枚举完整请求图、生成 Original/
  candidate arm、保存 typed rejection 和执行物化 attestation。
- [`autotune.py`](../python/triton/l_lite/autotune.py)：把外部已经编译好的 bindings 接到原生
  Autotuner，收集逐候选 timing、winner 和 cache；当前不生成 bindings。

### D.4 控制层测试文件

- [`test_control.py`](../python/test/unit/l_lite/test_control.py)：测试候选图、组合限制、native
  autotune facade 和 key cache。它是 Python 控制层单元测试，不执行完整 C++ route pipeline。

### D.5 十三个 MLIR 测试文件

以下文件的逐项证明范围已经在第 14.8 节给出，这里按变更清单完整列出，便于与
`git diff --name-status` 一一核对：

1. [`hbv-loop-body-semantics.mlir`](../test/Triton/hbv-loop-body-semantics.mlir)；
2. [`hbv-loop-bridge-affine-local-span-materialize.mlir`](../test/Triton/hbv-loop-bridge-affine-local-span-materialize.mlir)；
3. [`hbv-loop-bridge-affine-local-span.mlir`](../test/Triton/hbv-loop-bridge-affine-local-span.mlir)；
4. [`hbv-loop-bridge-early-return.mlir`](../test/Triton/hbv-loop-bridge-early-return.mlir)；
5. [`hbv-loop-bridge-operation-neutral.mlir`](../test/Triton/hbv-loop-bridge-operation-neutral.mlir)；
6. [`hbv-loop-bridge-pure-call-closure.mlir`](../test/Triton/hbv-loop-bridge-pure-call-closure.mlir)；
7. [`hbv-loop-bridge-recursive-effect-container-materialize.mlir`](../test/Triton/hbv-loop-bridge-recursive-effect-container-materialize.mlir)；
8. [`hbv-loop-bridge-recursive-effect-container.mlir`](../test/Triton/hbv-loop-bridge-recursive-effect-container.mlir)；
9. [`hbv-loop-bridge-scan.mlir`](../test/Triton/hbv-loop-bridge-scan.mlir)；
10. [`hbv-loop-bridge-ungrouped-axis-translation-materialize.mlir`](../test/Triton/hbv-loop-bridge-ungrouped-axis-translation-materialize.mlir)；
11. [`hbv-loop-bridge-ungrouped-axis-translation.mlir`](../test/Triton/hbv-loop-bridge-ungrouped-axis-translation.mlir)；
12. [`hbv-loop-mixed-radix-independence.mlir`](../test/Triton/hbv-loop-mixed-radix-independence.mlir)；
13. [`hbv-loop-runtime-mask-authority.mlir`](../test/Triton/hbv-loop-runtime-mask-authority.mlir)。

这 27 项是实现快照相对原生 Triton 基线的文件级全集。当前架构设计书是在该快照之后更新
的第 28 个文档差异；它只解释现状，没有偷偷增加新的编译能力。
