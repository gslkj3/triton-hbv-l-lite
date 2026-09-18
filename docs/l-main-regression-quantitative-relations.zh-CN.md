# 主 L 回归模型：定量关系、计算方法与实现边界

更新日期：2026-09-18。范围：P3L-0203 工作树及其已有模型文件；不是新一轮性能报告。

这份文档回答的是：代码怎样把状态变成数字，再把数字变成时间或加速比。**算式可复算，不等于算式中的每个系数都已被证明是独立的物理因果效应。** 当前仍有开发模型、历史保留模型和未接入预测的诊断观察，不能混称为已完成的发布模型。

## 1. 先分清当前有哪些计算路径

| 路径 | 实现及用途 | 不能据此声称什么 |
|---|---|---|
| 源码端点成对模型 | `source_service_adapter.py`、`frozen_paired_hinge.py`；用 Original 与候选的状态差预测对数加速比 | 当前适配器仅接受 Bridge factor>1、重排且 route factor=Bridge factor、grid 可整除的范围；不是三个优化的全覆盖 |
| 保留评分模型 | `global-choice-freeze-v1.json`、`retained-model-coverage-v1.json`；已有候选的历史/开发评分 | 模型支持不等于物化成功，更不等于发布证书通过 |
| Bridge→route 分层特征 | `_loop_sequential_three_arm_surrogate.py`；提供各段结构和后端响应特征 | 特征接口存在，不代表每条当前评分都使用它，也不代表 PTX 事实免费可得 |
| 窄域绝对时间模型 | `_loop_canonical_service_time.py`；共同时间与候选差分的独立实现 | 不能把它的公式冒充所有当前候选的实际评分公式 |
| copy-width 后端观察 | `LowerLoops.cpp`、`_loop_copy_width_state.py` | 当前未接入正式时间特征；不是已拟合的精度→耗时关系 |

`unified_development_scores.py` 的合并规则是：源码投影支持时采用源码模型，否则保留原评分。它显式写出 `publication_permitted=False`，不是按实测标签在多个模型中挑最有利结果。

源码路径均相对于主 L 仓库：模型模块位于 `python/triton/hbv/mechanistic/`，本轮脚本和证据位于 `hbv/iterations/P3L-0203/`。L-lite 不运行这些预测器；本文件放在其公仓中仅用于公开主 L 对照方法。

## 2. 输出是什么：时间比，而不是“预测越大越好”的任意分数

设原始耗时为 $T_O$，候选耗时为 $T_C$：

$$y=\ln(T_O/T_C),\qquad \widehat S=\exp(\widehat y).$$

若 Original 的时间已知，可换算 $\widehat T_C=T_O/\widehat S$。单独的比值模型没有给出绝对时间标尺，不能伪称独立预测了两个绝对耗时。

验收的收益恢复率是 $R=S_{\mathrm{selected}}/S_{\mathrm{best}}=T_{\mathrm{best}}/T_{\mathrm{selected}}$，两项均使用实测。例：100ns→最优50ns，选中55ns，恢复率为50/55≈90.91%。预测误差另报，不用恢复率掩盖结构性偏差。

## 3. Bridge 如何定量改变状态

设 grid 为 $(g_0,g_1,\ldots)$，Bridge factor 为 $b$：

$$P_O=\prod_i g_i,\quad P_B=\lceil g_0/b\rceil\prod_{i>0}g_i,\quad Q=P_Bb-P_O.$$

$P$ 是物理 program 数，$Q$ 是分组带来的填充槽数，不是实际额外访存次数。`terminal_logical_work.kernel_profile` 实现此式；当前源码评分适配器进一步要求 $g_0\bmod b=0$。

例如 $g_0=128,b=4$，program 数变为32，但每个 program 接收四组工作。**不能因此直接断言快4倍**：工作重用、依赖、并行供给与后端资源共同决定时间。

分层特征接口另有：

$$x_{B,bytes}=\ln(1+b(B_{in}+B_{out})),\quad x_{B,arith}=\ln(1+bA),\quad x_{B,programs}=\ln P_B.$$

实现：`_loop_sequential_three_arm_surrogate._current_exact_features`。这里表示请求工作量；若发生消除、重用或掩码，它不自动等于真实内存流量。不能和下一节源码图逐操作统计混为同一字段。

## 4. 源码端点模型的17个输入怎样计算

先对源依赖图做候选状态转移与循环不变量投影，再统计每个物理 program 的逻辑工作。不是先编译所有候选再读取其机器码。

| 输入组 | 个数 | 定量定义 |
|---|---:|---|
| load/store/arith/special/reduce/other 的 lower、upper | 12 | 每类逻辑元素槽数上下界 $W_k^-,W_k^+$，输入为 $\ln(1+W_k^-),\ln(1+W_k^+)$ |
| programs | 1 | $\ln(1+P)$；注意这里不是上一节的 $\ln P$ |
| 两类地址依赖站点数 | 2 | 有地址依赖的 load 站点数：不跨回边、经过回边；各取 $\ln(1+n)$ |
| 输入/输出字节 | 2 | 每逻辑 program 的字节上下文，分别取 $\ln(1+B)$ |

工作量规则在 `terminal_logical_work.work`：顺序操作相加；已知 trip 的循环将主体计数乘以 trip；条件分支取各分支下界最小值与上界最大值。张量槽数来自静态形状，规约的 combiner 另按逻辑二元合并次数计入。无法确定 trip、存在未解析控制流时拒绝投影，不猜一个值。

上下界是结构包络，不是运行概率或置信区间；不同字段的极值也不保证能同时达到。两个依赖计数统计的是站点，不是内存延迟、依赖链长度或 cache miss 数。输入输出字节相同的 Original/候选仍可通过交互项影响响应。

实现：`source_service_adapter.endpoint/candidate_state`、`terminal_logical_work`、`load_use_provenance`。

## 5. 17个输入如何算出加速比：完整模型公式

对训练端点集合求均值 $\mu_j$、标准差 $\sigma_j$（常量字段置1），令：

$$z_j(x)=(x_j-\mu_j)/\sigma_j,\qquad h_j(x)=\max(0,(x_j-k_j)/\sigma_j).$$

当前 `carried-export-v1.json` 使用17维模型，基函数为：

$$\phi(x)=[z_1,\ldots,z_{17},h_1,\ldots,h_{17},\{z_i z_j:i<j\}].$$

共有 $17+17+17\times16/2=170$ 项，当前导出165项权重非零。它是共享的分段二次统计代理，**不是17个独立物理公式，也还不能称为已证明最小的模型**。

$$F(x)=w^T\phi(x),\quad \widehat y=F(x_O)-F(x_C),\quad \widehat S=e^{\widehat y}.$$

相同端点必得0对数加速；交换端点符号相反。若三个端点均在同一状态映射和同一模型适用域，$\widehat y(O,C)=\widehat y(O,B)+\widehat y(B,C)$。这来自同一个势函数的代数性质，不代表当前适配器已支持全部 Bridge-only 与 route 候选。

### 5.1 系数怎样学

训练标签是同一输入的 $y_n=\ln(T_{O,n}/T_{C,n})$。计算 $d_n=\phi(x_{O,n})-\phi(x_{C,n})$，删除训练中恒为0的差分列，再以每列均方根 $r_j=\sqrt{\mathrm{mean}_n d_{nj}^2}$ 缩放：

$$\beta=\arg\min_\beta\sum_n(y_n-\sum_j\beta_jd_{nj}/r_j)^2+\|\beta\|_2^2.$$

这是 `Ridge(alpha=1,fit_intercept=False)`；导出 $w_j=\beta_j/r_j$，未使用列置0。推理不再训练、不查目标时间表。

### 5.2 分段为什么出现在这里

一般 hinge 模型的 $k_j$ 是训练端点中位数。当前 `CarriedSupplyPotential` 唯一覆盖的是 programs 字段：$k_{programs}=\ln(1+SM数)$；本次导出值约4.859812，即 SM数128。其他节点仍为训练中位数。

这个分段表达 program 供给规模不同区域的不同统计斜率。**它不是已证明的 occupancy 临界点，更不是寄存器 spill 阈值**：program 到 SM 的映射还受资源与调度约束。训练中位数节点也不能包装成硬件定律。

### 5.3 “交互”具体怎样改变斜率

在不跨节点处，$F$ 对第 $j$ 个输入的导数为：

$$\frac{\partial F}{\partial x_j}=\frac{w_j+w_{h,j}\mathbf1[x_j>k_j]+\sum_{i\ne j}w_{ij}z_i}{\sigma_j}.$$

所以增加逻辑工作量的影响会随 program 数、地址依赖等上下文变化；不是固定“多一次load慢多少ns”。对候选端点，$\partial\widehat y/\partial x_{C,j}=-\partial F/\partial x_{C,j}$。相关特征的系数可能相互抵消，单个系数正负不能当作独立因果结论。

## 6. 三个同级 route 的分层特征怎样定量表达

在 `_loop_sequential_three_arm_surrogate.extract_current_v20_route_given_bridge_features_v2` 中，令 $u=\log_2(route\ factor),v=\log_2(b)$。

| 关系 | 实际进入特征的形式 |
|---|---|
| 上游分组与下游变换强度 | $uv$，不是原始整数 $b\times factor$ |
| 软件流水与 load/store 工作 | $u\ln(1+N_{load})$、$u\ln(1+N_{store})$ |
| 展开与尾部工作 | $u\times known\_member\_tail\_fraction$ |
| 软件流水与后端响应 | $u\Delta_{memory}$、$u\Delta_{live}$ |
| 重排与后端响应 | $u\Delta_{instruction}$、$u\Delta_{live}$ |
| 向量化与后端响应 | $u\Delta_{vector}$、$u\Delta_{live}$ |

其中 $\Delta_A=\ln(A_C/A_B)$。指令数、memory、arithmetic/convert、control/sync、最大块采用这个形式；CFG边、peak slot、live area、q90 slot、vector destination采用 $\ln((1+A_C)/(1+A_B))$，允许0。Bridge组件同理比较B/O。

例如 memory指令数由100变为50，$\Delta_{memory}=\ln .5\approx-0.6931$；stage=3时交互为 $\log_2(3)\ln .5\approx-1.0986$。它对最终评分的贡献仍要乘以对应模型权重，不能直接宣布节约1.0986ns。

这些字段是 PTX（虚拟指令）形状代理，不是最终 SASS（二进制机器指令）资源真值。若实际获得它们需要编译候选，必须计入搜索开销；在没有经过验证的预编译投影前，只能作为监督/诊断或收费的编译特征。

## 7. 保留线性模型的实际数值例子

保留的标准化 Ridge/Huber 模型按下式算对数加速比：

$$\widehat y=a+\sum_jw_j(x_j-\mu_j)/s_j.$$

`retained-linear-prediction-trace-v1.json` 已逐项复算。入口18的一个软件流水历史评分：截距为-0.266887114；`strong_log1p_source_trip_count` 的值1.098612289、均值1.842715832、尺度0.644563526、系数-0.651397508，贡献约+0.751992898。所有36项相加后得到0.062543698，加速比约1.064540976。

同一记录的历史实测加速比约0.998068163，**这是错误预测实例，不是成功示例**。列出它是为了证明可以追溯数字并看清误差；“能逐项展开”并不能替代模型验证。旧模型的 Bridge-specific 分组和类别项也不能被描述为已完成的统一后端响应归因。

## 8. 窄域绝对时间实现：不要与上面的比值模型混淆

`_loop_canonical_service_time.py` 定义共同特征：

$$c(x)=[launches,M,compute/partitions,programs,formation].$$

可观测内存路径为 cache-resident 时 $M=dependency\_steps$，否则使用 global transactions。路径混合或容量过渡不可识别时有类型拒绝，不随意选一边。

令 $W=(W_O+W_C)/2$、cache容量为 $C$，$q=\min(1,\max(0,(W-C)/\max(W,1)))$。差分为：

$$d=[P_O-P_C,q(P_O-P_C),I_O-I_C,L_O-L_C].$$

这里I为formation指令，L为并发live字节。模型为：

$$m=(a^Tc(O)+a^Tc(C))/2,\quad e=b^Td,\quad \widehat T_O=m+e/2,\quad\widehat T_C=m-e/2.$$

实现会将e限制在 $\pm\max(0,2m-2\times10^{-9})$ 内。系数单位为ns/对应单位；q只是容量饱和坐标，不是真实cache miss率。这个模块不能证明所有当前评分都已得到绝对时间预测，也不能替代 spill 分配模型。

## 9. 残差、半径与尚未进入模型的关系

`predict_loop_sequential_surrogate_v1` 目前要求 `learning_residual=exact_zero`，输出学习残差0、无区间、无发布权。不能把研究设计中的残差层写成每条评分已启用的修正。上面的 frozen paired hinge 也没有额外学习残差项；残差或半径接入时须另列其公式、训练对象与数据角色。

copy-width 观察涉及：元素位宽、指针连续性、mask对齐、寄存器到shared布局的copy字节数、调度stage距离。`LowerLoops` 的事实只覆盖它实际访问且stage距离非0的load；空数组不等于没有load，也不能据此断言没有完成后端编译。

203/207中8个连续元素×16bit，与205中4×32bit，均为128bit；这不支持“8lane能3stage、4lane只能2stage”的规则。目前没有把这种规则写入回归。stage是配置，依赖与布局另有约束；改变多个因素的对照不能定量识别单因素效应。

## 10. 如何维护，不让这张表重新变成定性说明

每次新增或改动主要关系，必须同步补充：输入名/单位/可取得阶段，公式及边界，系数文件及摘要，拟合与独立验证角色，实现函数，一个可复算例子，适用范围与反例。改动没有进入实际评分时明确标为观察或实验。

模型更新时运行本目录 `render_regression_quantitative_appendix.py`，更新附录的全部17维参数与170项权重；不能只贴最有利的项。模型系数不是不可变物理常数，不跨架构直接承诺数值有效性。

附录：[当前导出模型的全部数值与源码摘要](l-main-regression-quantitative-parameters.zh-CN.md)。主报告与附录应一起发布。尚未完成的物化、独立验证和安全发布继续在当前计划跟踪，不以本文件存在代替闭环。
