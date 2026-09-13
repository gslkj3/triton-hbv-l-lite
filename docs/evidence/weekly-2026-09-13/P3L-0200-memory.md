# P3L-0200 Memory

- 输入证据：P199-001。
- 科学总体变化：从仅 `bridge_factor > 1` 的 Bridge×route 组合总体，扩展到同样
  合法的 `bridge_factor == 1` route-only 总体。
- 这不是第四种 route，也不是 operator adapter；Bridge 身份路径和既有三 route
  的候选仍由统一规则生成。
- P154 历史计时只有在当前主 L 最终 SASS 完全一致时才能直接复用。

## 执行准备

- P200-001 冻结了六个角色；执行前发现其未绑定完整运行实现，因此没有启动任何
  GPU 角色，保留为总体定义。
- P200-002 在执行前补齐 runner、主 L selector/plugin、Strong、Weak、Cut-B、
  编译证明、runtime 和 C++ pass 共 14 项实现哈希，成为唯一执行 authority。
- route-only 的纯 CPU 单元链已通过 126 项测试，覆盖候选组合、Strong 终态、
  Cut-B、顺序 Strong 与弱状态传播。这说明当前规则代码具备 Bridge 身份路径的
  表达能力；真实 kernel 的最终物化仍须由 P200 当前编译角色证明。
- 两次只读 admission 均检测到 S 系列 request 文件，runner 按约定返回 75，未
  导入 Triton/Torch，未启动 GPU 子进程。

## route-only 最早责任点与修复

- P200-003 对冻结 P163 谱系完成了纯 CPU 清点：三个既有 route 共有 309 条有限
  route-only 候选，308 条可与主 L 最终 SASS 精确连接，其中 116 条达到 1.0175×，
  覆盖 57 个物理请求；但 route-only 的候选编译前状态记录为 0。
- 根因不在物化 pass，也不在残差或机制半径：通用
  `LoopTerminalPrecompileFeatureRecordV1` 已能表达已有循环和 Bridge 构造循环，
  但其采集生命周期被错误绑定在 shortlist 模式，普通全候选编译没有保存它。
- 修复将“采集候选编译前事实”和“用事实做 shortlist 排名”拆为两个正交动作；
  `bridge_factor == 1` 时由当前 route 唯一可用的 Provider 循环提供 trip/main-tail
  事实，factor 合法性复用中央 ontology。没有增加 route、算子 adapter 或 kernel
  身份特征，也没有消费 Candidate C、计时、最终身份或遥测。
- P200-004 的 194 项 CPU 测试通过；旧 P167 的 6326 行投影逐行保持不变。
- P200-005 是仅用于已披露开发总体的状态采集 authority；P200-006/v3 在任何 GPU
  执行前冻结当前实现，取代未执行的 v2，成为唯一执行 authority。
- v3 admission 再次检测到 S 系列 request，返回 75 且没有生成角色报告。当前可以
  继续 CPU 证据工作，但不得启动新 GPU 角色。

## subject-set 外层回退

- P200-008 把 308 条历史 exact-SASS route-only 候选按当前 Strong 规则重放后，
  发现 240 条软件流水候选被误报为“无主体”；其中 102 条候选、51 个请求曾达到
  物质收益。这不是时间模型误差。
- 对照当前主 L 的正式 plan、物化器和已有
  `pipeline-subject-set-strong-state` 证据后确认：软件流水 stage 是 whole-kernel
  共享选项；所有独立、可流水化的顶层循环共同构成一个规则式 subject set，不能
  强制压成唯一循环。完全展开 route 同样可能合法作用于一组独立顶层循环。
- P200-009 修复后，同一冻结真实 kernel 谱系的 308/308 条 route-only 候选均得到
  Strong 支持。集合传播的是成员数、静态/动态成员比例、已知 trip 总量、main/tail
  与消除比例、route opportunity 总量；locator 只用于审计身份，不进入模型。
- 204 项 CPU 测试通过。P200-010/v4 在任何 GPU 执行前取代 v3，成为当前唯一执行
  authority；无需升 P201，因为科学总体和角色未变，修复仍处于同一外层循环。
- 扩展 CPU 回归中，排除已明确不属于本次 L 修复的 CUDA/S 集成依赖缺失、旧 D7
  fixture 和旧错误文本断言后，其余 HBV 单元全集 2381/2381 通过。全量首次运行的
  40 个失败中，36 个是隔离 Python 环境缺少 CUDA extra module，3 个是旧 D7
  inactive active-extent fixture，1 个是旧错误文本；未据此修改 L 模型或代码。
- 随后的同构审计发现 Strong 虽已表达 subject set，预编译 categorical 仍把所有
  unchanged-Bridge 路径记为单一 existing loop。P200-012 改为直接按规则生成的
  plan member 数区分单成员与 whole-kernel set；这防止两类后端路径被错误混入同
  一个回归域。205 项聚焦测试通过。
- P200-013/v5 在任何 GPU 角色执行前取代 v4，成为当前唯一执行 authority。

## route-only Pipeline 最小剂量诊断

- P200-014 首次把 270 条 exact-SASS route-only Pipeline 历史候选做成
  leave-template-out 诊断，但其空模型对同请求内的相同预测分数错误地使用“大
  factor 优先”打破平局，使没有排序能力的空模型看起来也能恢复机会。该产物不
  作为方法结论；P200-015 显式将空模型记为 `unrankable`，并记录这一纠错。
- P200-015 的冻结开发总体包含 90 个物理请求、6 个测试入口、8 个源代码模板组，
  每个请求各有 Pipeline factor 2/3/4，因此共有 270 条 exact-SASS 候选。SVD
  单一测试入口包含三个源码 kernel 模板，不能把 6 个入口误当成 6 个留出组。
- 只使用 Candidate 编译前已知的 `log2_route_factor` 时，8 个
  leave-one-template-out fold 的 Ridge 斜率方向全部为正；因此一个斜率即可在此
  已披露总体内排序三个 stage 剂量。它选出的 factor 4 恢复 55 个物质收益机会中
  的 50 个（90.91%），90 个请求的被选候选中位加速比为 1.02635x、均值为
  1.19075x。
- 这还不是安全模型：候选级 log-MAE 为 0.16631，且 4 个本来存在物质收益机会
  的请求被选成小于 1x，全部 90 个请求中有 16 个负收益选择。因而剂量方向只能
  作为 route-local 最小排序假设，不能替代状态门、误差半径或发布证书。
- P200-015 没有消费 Candidate-C 特征、Q/E、残差、半径或发布标签；它只用于决定
  当前源码精确连接后首先验证哪个最小状态。当前源码重编译仍由 V6 六角色负责。

## Pipeline 终态代理阶梯重放

- P200-016 首次把 P159 在新标签完成前冻结的 R0→R3 终态特征阶梯重放到上述
  270 条 route-only Pipeline exact-SASS 行。它因两项方法问题被标为 INVALID：旧
  Huber 实现有四个未收敛候选仍参与排序；身份审计又把因果量
  `source-loop trip` 误当成了 `source-file identity`。
- P200-017 校正后只保留数值收敛候选，且身份检查只禁止明确的 kernel/operator/
  文件/请求/模板身份字段。270 行、90 请求、8 个可由 location-free AST 解析的
  模板全部闭合；4 个未收敛 Huber 候选被排除。
- 按原 P159 时间中心协议，D 上选中 R2 浅层 gradient boosting：26 个终态坐标、
  224 个树节点，leave-template-out log-MAE 为 0.13008。它在未加半径时发布 69/90，
  其中 35 个达到 1.0175x、12 个实际小于 1x，因此它不是安全发布 authority。
- 若只看“预测后强制实测一个候选”的开发恢复上限，最佳模型也只把 55 个物质
  机会从剂量模型的 50 个提高到 51 个，并仍在物质机会中选出 3 个负收益候选。
  这不足以支持用 224 节点替换最小剂量假设，也证明仅靠增加终态代理容量不会
  自动闭合安全性。
- P200-017 的作用是定位：当前源码精确连接后，必须先核对新的 rule-derived
  Strong/Weak/终态状态是否改变这三个反转；若没有，外层回到 Pipeline 后端路径
  子域或状态定义，而不是继续堆回归容量。Q/E、单侧半径和发布仍未开始。
- P200-018 按 P158 已冻结的后端路径顺序检查
  `terminal response regime × vector destination change`。四个“存在物质机会但
  factor 4 为负收益”的请求中，三个仍在占主导的 `expansion+preserved` 单元，
  一个在 `expansion+changed`；所以 P1/P2 路径类别可以隔离部分明显负收益，却不
  足以闭合关键反转。
- 同一审计还发现旧 `pipeline_feature_record` 把全部 270 行都记成
  `whole_kernel_subject_set`，而 P200-009 的统一规则已确认其中 30 条属于真实的
  singleton exact Strong 子域。旧 categorical 因而不能直接用于再分域；必须由
  V6 当前源码重编译产生新 cardinality，不能对旧标签打补丁。

## V6 执行封装闭合

- V5 尚未执行时的最后启动审查发现：预定的 `venv/bin/python -S` 会去掉 venv
  自身的 site-packages，runner 在 request 消失后将无法导入 torch/pytest。这个是
  执行封装缺口，不是模型、合法性或物化失败。
- runner 现在使用不解析 symlink 的 `sys.executable` 父环境定位当前 venv 的
  site-packages；先清除 S 与 L-lite 的 editable Triton 路径，再显式插入主 L
  Python 和该 venv site-packages。隔离探针已同时确认 torch、pytest 来自
  `/home/jinyiw/venv`，Triton 来自当前 `/workspace/hbv-p3l0094/python`。
- P200-019/V6 在任何角色执行前取代 V5，重新绑定 14 个实现文件与 join builder。
  request 存在时的 V6 admission 返回 75，真实 `.jsonl` 报告保持不存在。

## 当前遗漏与完整已披露收益总体

- 重新按物理 request identity 清点后，P163 旧 FlagGems route-only 物质收益总体
  有 57 个请求，P199 新增总体有 29 个请求，二者交集为 0，合计 86 个请求。
- 57 个旧请求按测试入口分布为：`broadcast_to=4`、`renorm=3`、
  `searchsorted_out[dtype4]=1`、`softmax=7`、`softmax_backward=41`、`svd=1`。
  该不均衡是物理请求的真实分布，后续拟合与留出不能把它误写成六个等权样本。
- 上述 86 是当前总体遗漏，不是全项目分母。P167 的 Bridge×route 总体已有 78 个
  包含物质候选的唯一请求，其中 3 个也有 route-only 物质机会；P199 29 请求与
  二者均不重叠。因此 L-lite 已披露物质机会的完整物理请求并集为 161。
- P200-020 将该口径固化为 create-once 证据，并绑定 66 项协议、角色产物与构建
  实现哈希；78、57、29、交集 3、遗漏并集 86、完整并集 161 均通过重算检查。
  request identity 仅参与集合核算，不进入任何回归特征。
- 随后的范围审计确认 P200-020 不包含三个 PR 受控复现面板。P200-021 保留 161
  的自然 kernel 数值，但将 `complete` 的范围明确限定为 P163/P167/P199 主证据
  链；PR2955 的 24 shape 行/336 次 route-reference 执行与 PR4125/4126 的 32 行
  单独报告和验证，禁止把不同实验单位直接相加。
- P29/P41 与当前源码复核确认：PR4125/4126 的动态 exact-prefix 能力后来已经作为
  `full_unroll_logical_group`（完全展开+向量化）的 provider-closed 子型进入因果
  与物化链；`_loop_factor_ontology.py` 明确声明它不是第四 route。后续只验证当前
  子型物化、预测和收益恢复，不回到 P23 的“新增第四机制”旧路径。
- exact-prefix ontology、Strong/Weak、编译证明、Cut-B、subject-set 与
  provider-bound 物化的 114 项 CPU 回归在隔离当前 L 导入后全部通过。首次普通
  venv 启动在收集阶段误导入 `/workspace/p3s0083...` 的 Triton，未进入测试执行；
  用 V6 同款 `python -S + 显式当前 L/site-packages` 后 114/114 通过。这项失败归于
  测试环境污染，不归于 L 功能，也不能被删去不报。
- P200-022 保存上述隔离重跑的 JUnit；P200-023 绑定 8 项当前实现、7 个测试文件、
  V6 协议与总体范围证据，状态为
  `P200_CURRENT_EXACT_PREFIX_ROUTE_ONLY_CPU_CLOSED`。它只证明规则/物化单元闭环，
  不替代真实 kernel 当前 SASS、PR 面板性能或 Q/E。
- 与 P41 协议做当前哈希对照后，exact-prefix v3 calibration authority 和
  `_loop_reduction_service.py` 保持完全一致；`_loop_cuda_sm89_provider.py` 与
  `_loop_region.py` 已漂移。当前 region V4 还明确规定 subtype 只能作为物化能力，
  不得进入 production selector。这使 P41 已闭合的 exact-prefix 局部 authority 在
  当前主 L 中失去选择入口，属于生产集成工程缺口。
- 修复不能恢复第四 route，也不能按 PR 名称分支。P200 V6 已绑定 `_loop_region.py`，
  所以应先完成当前 29 请求角色，再在后续不重叠版本中把 exact-prefix 挂回
  `full_unroll_logical_group` 的规则子型时间头，并重放 P41 的两个独立 holdout 与
  32 行 PR 诊断。现在修改会使尚未执行的 V6 再次失效，因此被顺序门禁止。
- P200-024 将该结论固化为
  `P200_EXACT_PREFIX_PRODUCTION_INTEGRATION_GAP_TYPED`：P41 authority/service
  当前字节一致、ontology 明确非第四 route、region 明确阻断 subtype，且 region
  已被 V6 绑定。最早失败 owner 因而是 production region subtype dispatch。
- P200 当前执行权只覆盖新增 29 个请求。先完成这六个角色与精确 SASS 连接，再
  冻结旧 57 请求的当前源码重编译；两套 writer 不得重叠执行。最终低搜索恢复率
  的遗漏恢复率分母固定为 86；最终完整开发集恢复率分母固定为 161。独立发布
  证书另取未参与重推的真实 kernel 家族。
# P200-025：当前数据产生前冻结少量补样门

- 六个 V6 当前角色报告仍全部不存在时，冻结
  `p200-025-rule-cell-supplement-protocol-v1.json`。
- 覆盖单元固定为 `route × rule-derived Strong terminal subdomain × factor`；仓库、
  算子、kernel、PR、nodeid 和 request identity 只能作为来源，不能成为坐标。
- 构造样本仅可补合法性/类型拒绝、真实物化、预 C 状态可辨识和最终 artifact
  attestation。它不能提供计时、回归精度、物质收益、Q/E、半径或发布结论。
- 每个缺失规则单元最多两个见证，总数最多十二；超限代表应回到外层扩真实 kernel
  总体或重审规则分区，禁止以 case-by-case 合成样本掩盖总体问题。

## P41 exact-prefix authority 的量化复核

- 当前 v3 authority 与 P41 发布时字节相同，核心 reduction service 也字节相同。
- P41 第二个预注册独立 holdout：24 行中自动采用 12 行，12 行全部物质正收益，
  零物质伤害，采用行全摊销收益区间 `1.343872x–2.593153x`，整体 time-weighted
  全摊销 `1.320999x`。
- 32 行 PR4125/4126 受控历史诊断面板：自动采用 12 行，零新伤害，time-weighted
  全摊销 `1.309411x`，正节省捕获率 `75.579%`。
- 上述结果只证明冻结窄域 authority 曾经成立。当前 production region 阻断非
  base subtype，因此必须在 P200 结束后以新 owner 接回，并重验当前 artifact
  identity、两个独立 holdout、32 行面板和负对照后，才能形成当前版本结论。

## P200-026：当前规则单元清点方法预注册

- 在六个 V6 当前角色报告均不存在时，绑定
  `build_p200_current_rule_cell_census_v1.py`。
- 清点坐标只有 `route_ref × Strong terminal subdomain × route_factor`，只读当前
  pre-C 状态、Strong/Weak 闭合事实和最终 SASS 是否存在，不读任何性能标签。
- 真实 kernel 已到达但没有最终 SASS 的单元，先做最早物化 owner 诊断；只有确认
  不是工程缺口而是缺少规则边界见证后，才允许进入 P200-025 的少量补样门。

## P200-028：已证伪模型路线继承

- P192：log-MAE 改善但物质恢复从 `40/73` 降到 `36/73`，不能再以更低点误差作为
  唯一理由重拟合 selected head。
- P193/P194：`41/73` 的机械改善来自五个常数域先验，因果空洞，不得发布。
- P195：域内 pairwise 更可分，但跨域最终恢复降到 `39/73`；禁止继续调 pair
  权重或增加局部模型。
- P197：无选择加入完整 P186 状态后恢复降到 `38/73`；更多可观测字段不等于更好
  因果状态。
- P198：73/73 请求都有未冲突物质候选，missed 的最近训练距离更大。它支持先修
  样本类型/路径支持，不足以证明 HBV 边界外。
- P199 随后找到了更早 owner：29 个真实正收益请求全部被旧 Bridge×route 总体
  排除。P200 必须先完成 route-only 当前状态与 artifact identity，再讨论模型。

## P200-029/P200-030：S 方法可迁移性审计

- P200-029 首次执行因 H20 跨行字符串匹配及 S5000 顶层 `passed` 字段假设错误而
  typed-invalid；文件保留且不覆盖。P200-030 只修正来源 schema 读取，不改变科学
  方法或 L 采用边界。

- 快速论文基线可迁移：device-time、acquisition/compile、end-to-end 使用不同环境
  authority；不能拿一个 Host 指标统一否决所有证据角色。
- 当前 SM89 可迁移：先由动作事实预测粗终态资源类 W，再由 whole-kernel service
  状态和 W 输出进入 H；但 L 只有在当前 artifact 证明相同稳定类后才允许采用这种
  两段结构。
- H20/S5000 可迁移：每个 predictor/机制域使用自己的 Q winner residual margin，
  保留 censored label；支持不足时 radius 保持 unavailable；E 只读一次且不反向
  修同代模型。
- 不迁移：S Strong/因果域、action identity、候选空间、W/H 特征与系数、support
  hull、Q margin/radius、阈值、域数量和任何计时标签。

## P200-031：环境证据角色分离

- 当前六角色是 compile/state/identity，不是性能角色；必须等待 S request 消失、
  保持 V6 哈希、正确设备和最多四编译进程，并保存 GPU/PID 前后快照，但不要求
  高精度性能独占。测试入口偶然产生的 benchmark 数字一律不可用。
- 历史时间只在 physical request、coordinate、current pre-C state 和 final SASS
  全部精确连接时复用，环境 authority 仍属于原冻结历史证据。
- final SASS 变化后的重测以及 Q/E 属于 paired device-time 角色，必须另行冻结严格
  环境门；Q 只校准 winner risk，E 一次只读且不修同代模型。
- GPU 利用率、显存、温度、功率只能是环境上下文，禁止进入 Strong/Weak/时间中心、
  残差或机制半径。

## P200-032：exact-prefix 后续生产接入设计冻结

- 在不触碰 V6 已绑定 `_loop_region.py` 的条件下，先冻结后续接入的唯一合法身份
  元组：因果 route 是 `full_unroll_logical_group`，子型是
  `predicated_exact_prefix_vectorization`，产物 route 是
  `predicated_exact_prefix_reduction`，factor=1 表示 exact-prefix adapter，而不是
  普通的“不优化”。因此它是完全展开＋向量化域的规则子型，不是第四因果域。
- 生产接入必须复用当前已经存在的 provider/factor ontology、D4 exact-prefix
  Strong、exact-prefix Weak 和独立窄域时间服务；禁止把它错误塞入 generic
  full-unroll Strong，也禁止恢复旧的 `selected_exact_prefix_reduction` 第四 route
  身份。
- 只有 provider 证明 exact active prefix、零值 predication、合法结合子和
  power-of-two carrier 等规则事实时才能自然生成一个子型候选；仓库、算子、
  kernel 和 PR 名称均禁止成为准入键。
- P41 第二 holdout 的 12/24 自动采用、零物质伤害和 `1.320999x`，以及 32 行 PR
  面板的 12 次采用、零伤害和 `1.309411x`，只作为现有局部 authority 的历史先验。
  当前生产接回后仍必须重验 contract、编译物化、正确性和 final SASS；只有 SASS
  精确相同才允许复用历史计时，漂移者另开预注册高精角色。

## P200-033：当前角色执行边界复核

- 一度把后续性能证据的“advisory lock＋持续 NVML 独占”误套到当前六个
  compile/state/identity 角色；按 P200-031 的证据角色分离复核后撤销该判断，没有
  修改 V6 runner，也没有因此升版。
- 当前角色的原子安全边界是一整个 kernel 角色：启动前必须看不到双向 request；
  若运行中对方提出 request，只允许完成当前角色，不得再启动下一角色。runner 在
  导入 Triton/torch 前检查 request，并保存角色前后 NVML，符合该语义。
- P200-033 证实 V6 implementation drift 为 0、六份报告和 sidecar 均尚不存在，
  但构建时 request 有效，所以只能声明 static-ready/runtime-deferred。request 消失
  后仍须在每个角色入口即时重查，不能用这份静态观察替代未来 admission。

## P200-034：route-only 最小模型选择预注册

- 在任何当前角色报告和新标签产生前冻结模型选择，防止看完赢家后按 kernel
  case-by-case 加 head。初始分区固定为 route 与规则式 Strong 终态子域，factor 只
  是剂量；拆分/合并都必须由事前 Strong 类型、typed backend path 与 grouped
  holdout 等价性共同支持。
- 候选模型被限制为 intercept-only 诊断、ridge、Huber 与有界浅层 boosting；特征
  只能逐层加入 pass 剂量、Strong 终态效应和可识别的 pass-invariant 服务环境。
  Candidate-C 产物、计时赢家、identity、温度、功率和利用率均禁止进入。
- 选择目标优先保证 `>=1.0175x` 物质机会的赢家恢复、零/低 false adoption 与较小
  排序 regret；不要求拟合阈值以下的微小正负。grouped log-error 相差不超过
  `0.005` 时强制选参数更少、字段更少者。
- 每个可导出 head 至少需要八个独立 physical request，条件允许时至少覆盖两个源码
  家族；不足就明确 unavailable 并扩真实 kernel，不能用结构补样伪造计时。D 只拟合
  中心，Q 只校准 winner risk，E 一次只读；结构化大残差回到外层，而非塞进残差或
  半径。

## P200-035：精确连接后的外层 owner 解析预注册

- 审查发现 P200-007 能正确分类缺失 request、缺失 graph coordinate、缺失 pre-C
  state、物化失败、SASS 漂移和精确复用，但其 `next_owner` 无条件写成模型选择。
  分类事实不需要推翻，固定流程指向却不能成为外层 authority。
- 为避免改动 V6 已绑定分类器，P200-035 在六份当前报告和 P200-007 均不存在时绑定
  后继 P200-036。它只读 P200-007，并按因果链从早到晚选择 owner；只在所有更早
  模块闭合后进入最小模型。它不重分类候选、不拟合、不产生 Q/E、半径或发布结论。

## P200-037：旧 FlagGems route-only 57 请求的后继执行协议

- 从 P200-020 的原始 L-lite/main-L exact-SASS 证据重新解析出六个入口及 57 个唯一
  physical request，而不是只保存总数。分布为 broadcast_to 4、renorm 3、
  searchsorted_out[dtype4] 1、softmax 7、softmax_backward 41、svd 1。
- 该协议明确位于 current29 之后：必须先有 P200-027 和 P200-036；若 current29 的
  最早 owner 仍是 graph/Strong、pre-C 或物化，runner 会拒绝扩大总体。只有全部闭合
  或仅剩 changed-SASS 精确重测清单时，才允许顺序执行六个 current57 角色。
- 六角色仍是 compile/state/identity 证据，不采纳运行中偶然 benchmark 数值。每个
  nodeid 是一个原子边界，900 秒超时、最多四编译进程，并在启动前检查双向 request。

## P200-038：current57 精确连接方法预注册

- 在六个 current57 输出均不存在时，冻结 P200-039 构建器。它只消费 P200-037
  预先列出的 57 个 request，逐候选连接 route coordinate、pre-C state、当前产物和
  历史 final SASS；历史时间只在字节身份一致时复用。
- P200-039 内置从 request/population、graph/Strong、pre-C、物化到 changed-SASS
  的最早 owner 顺序。全部通过后才形成完整 86 请求的最小模型输入；identity 与
  环境遥测只作证据来源，不能成为模型字段。

## P200-040：完整真实角色开发总体

- current29 六角色与 current57 六角色的物理 request 完全不重叠，合计 238 个请求；
  历史 L-lite 中共有 2618 个 route-only 候选坐标、536 个有限计时和 267 个
  `>=1.0175x` 物质候选。P200-040 原假设“至少含一个 route-only 物质候选的
  request 恰好是既有 86 目标集合”被检查失败关闭：实际是 84 个。
- 差出的两个请求不是丢失样本；其历史赢家坐标分别要求 `bridge_factor=2` 后再进入
  route，属于 `Bridge→route` 联合组成。把它们强塞进 route-only 模型会跨越因果
  owner。P200-041 因而冻结为 84 个 route-only 恢复目标加 2 个组合恢复目标，两者
  无交集且共同精确重构既有 86 个机会。
- 仅用 86 个已知正收益请求拟合会发生 positive-only selection bias：模型看不到
  同入口无收益 request 和候选，false adoption 指标会虚假变好。因此 238 个完整
  角色总体负责 D 拟合、support 和误选统计；84+2 只负责已披露机会恢复率，并按
  各自 owner 分开建模与核验。
- P200-041 仍是历史开发总体定义，不证明当前源码兼容；所有候选必须经过 current
  pre-C state、物化与 exact-SASS 连接后才能复用其时间标签。

## P200-042：完整笛卡尔积与第 87/88 个机会

- 对同一 12 个真实入口重新遍历完整 candidate ledger 后，确认总体由 238 个
  Original、2618 个 route-only 和 3322 个 `Bridge→route` 候选组成；没有作为发布
  坐标的 Bridge-only 候选。route-only 有 536 个有限计时、267 个物质候选；组合域
  有 1102 个有限计时、33 个物质候选。所有有限变换候选均带 final SASS identity。
- 84 个请求存在 route-only 物质候选，7 个请求存在组合物质候选，二者重叠 3 个，
  所以完整物质请求并集是 88，而不是先前清单中的 86。新增两个请求均来自
  FlagGems SVD，历史赢家分别是 Bridge 后的 phase-major 或 Pipeline route；这是
  population omission，不是预测误差，也不能按 kernel 名称打补丁。
- 后续 D 必须连接全部 238 请求与全部 6178 个提交坐标；88 个集合只作为开发收益
  恢复目标。组合候选先接受 Bridge Strong 转移，再接受所属 route Strong 转移；
  不把它声明成第四个优化 pass，也不把 Bridge 因果效应吞进 route-only 模型。

## P200-043：全 238×6178 current 连接方法

- 在十二个 current 角色仍未执行时冻结 P200-044 连接器。连接顺序严格按 request、
  candidate coordinate、Strong、Weak、pre-C、物化、final SASS、历史标签资格推进；
  有有限历史计时的变换候选在任一上游节点失败，都回到最早责任模块。
- 历史没有有限计时的候选只能说明“这个坐标曾被提交”，其当前闭合情况记入兼容性
  账本，但不能驱动回归或发布。当前 main-L 多出的坐标也只报告覆盖，除非另有预注册
  标签角色，否则不能静默进入 D。
- 历史 Original 候选没有 final SASS 哈希。P200-044 因此禁止直接复用旧 Original
  时间作为当前基线；最终收益比较必须测一次 paired Original。历史变换时间则只有
  current final SASS 字节相同时才能复用。

## P200-045：组合候选的最小因果模型协议

- pre-C 终态投影本来就包含 Bridge/route factor、二者交互、Bridge 构造 subject、
  原生已有 subject、main/tail 和服务机会等规则事实；不需要按算子增加局部模型。
  旧 P200-034 只因 `bridge_factor=1` 的范围限制而不足，P200-045 扩展的是数据与因果
  链范围，不修改三个 route pass 的身份。
- 组合候选依次经过 Bridge Strong/Weak 与 route Strong/Weak，最后进入所属 route 的
  窄域时间服务。初始模型 head 仍为 `route × strong_terminal_subdomain`，两个 factor
  是干预剂量，不是拆 head 的理由。各 head 第一目标是自己闭环；全部闭环后，只有
  Strong 类型、typed backend path 与 grouped holdout 响应同时等价，才允许有条件
  部分共享。
- 0.005 grouped log-error 等价带内选择拟合参数最少、再选择字段最少的模型；低于
  1.0175×的细小正负不要求全部判对。旧学习残差拟合的是 route-only 模型，不能转嫁
  到新的完整顺序模型，故从零开始，并且只在中心模型冻结后学习跨代系数小漂移。

## P200-046：Bridge 条件效应的可辨识边界

- 历史真实总体提供 197 个 request/route/route-factor 内配对，可比较 Bridge factor 1
  与更大 factor；覆盖 10 个 request、2 个来源，但只有 Pipeline 与 phase-major。
  这是 Bridge 条件边际时间效应的开发先验，仍须 current exact-SASS 连接。
- 4 个请求只在 `Bridge→route` 后存在物质收益，route-only 反事实本身不合法。这里
  能闭环的是“Bridge 构造 subject 是必要原因”和“组合产物相对 Original 的联合
  效果”，不能从结果中凭空拆出 Bridge 单独时间系数。
- 组合窄域模型可消费顺序 Strong/Weak 后的完整终态预测联合效果；不可分解的贡献不
  是小扰动，禁止塞进学习残差、时间残差或机制半径。若论文需要 logical-group 的
  独立 Bridge 边际声明，只能继续找自然配对真实 kernel；构造样本最多补结构闭环。

## P200-047：完整路径规则单元 census

- 旧 P200-026 只覆盖 `bridge_factor=1`，且容易把 graph 中的规则拒绝误称为 finite
  plan。P200-047 预注册的 P200-048 同时覆盖 route-only 与 `Bridge→route`，明确
  区分 typed rejection、Strong/Weak 闭合、pre-C 状态和 final SASS 物化见证。
- coverage cell 包含 path、route、Strong 子域和两个 factor；factor 只用于证明各剂量
  的合法性/物化能力，不自动成为回归 head。回归仍遵守 P200-045 的最小 head 协议。
- 已到达却无 final SASS 是工程物化失败，不能靠构造样本绕过。只有较早 owner 全部
  闭合后，规则可达但自然总体没有见证的结构边界才允许最多 12 个构造样本；任何
  构造计时、收益、Q/E 或发布结论仍被禁止。

## P200-049：带类型拒绝证据的最早 owner 修复

- `_loop_region.py` 的 graph candidate 已有 Strong/Weak/provider 各自的闭合布尔、
  typed reason 和投影 identity；P156 比较适配器原先只写出两个闭合布尔，导致报告
  不能证明“拒绝是规则给出的”而只能证明“没有闭合”。
- 修复仅新增序列化字段，不改变 graph 构造、合法性、物化、预测或执行。3 个 CPU
  单测分别证明合法候选理由为空、Strong 拒绝理由保留、Weak 与 provider 拒绝互不
  混淆。
- V6 已冻结但未执行；修改适配器后其 implementation hash 合法漂移。因此不得伪称
  V6 仍可执行，必须冻结 V7 并让下游 current join 绑定新证据 envelope。

## P200-050：current role V7 执行协议

- V7 保留 V6 的六个真实角色和 create-once 输出，不改变 population 或科学模型；
  只换用能保存 typed graph reason 的报告适配器与新 runner。
- 冻结门要求 V6 从未执行，且 V6 implementation drift 必须只有该适配器；任何其他
  源码漂移都会拒绝。runner 仍在 Triton/torch 初始化前检查双向 GPU request，编译
  并发最多四个，角色只生成 correctness/state/final-SASS evidence。
- 后续 current join 必须显式绑定 V7；旧 P200-043/P200-047 方法保留为未执行只读
  协议，不能作为新输出的执行 authority。

## P200-051：V7-aware 全量 current join

- P200-052 复用 P200-044 已冻结的 238×6178 坐标与 exact-SASS 分类，只把 current29
  输入切换为 V7，并保存 Strong/Weak/provider 各层 typed reason 和投影 identity。
- 对每个已观察 graph 坐标，`closed` 与对应 typed reason 必须恰好互斥；否则证据
  自身无效。投影 identity 用来追踪责任模块，永不进入回归特征。
- 历史 Original 仍无 SASS，正式收益必须 paired baseline；V7 不是性能标签来源，
  P200-052 也不产生模型、Q/E、半径或发布结论。

## P200-053：typed rejection 与缺失坐标分离

- P200-054 不再把 `current_coordinate_absent` 叫作 typed rejection；它只说明当前图
  没有该坐标，需结合历史有限标签判断是工程回归还是未测兼容性。
- `strong_transition_closed=false` 只有同时观察到非空 Strong typed reason 才是规则
  拒绝。V7 已提供该字段；若为空，返回报告/Strong 证据 owner，宁可证书失败也不
  猜理由。
- P200-052 的最早 owner 优先于 census 自己的后继判断。结构单元与补样上限不变，
  factor 仍只是物化剂量覆盖，不是按 case 扩大回归 head 的权限。

## P200-055：V7 execution-ready 审计

- V7 auditor 将 graph 中规则拒绝与已准入 plan 分开：只有 Strong/Weak 都闭合的 route
  坐标才必须有 pre-C state；每层 closure/reason 必须互斥，产物必须属于已准入集合
  且带 final SASS。
- 角色只产生 correctness、typed refusal、state 与 artifact identity，不产生性能
  标签，所以每个完整 kernel role 是原子安全边界；changed-SASS、Q/E 的 paired
  高精度环境门仍在后续独立 authority。
- 启动顺序为 3/1/2/4/5/0，每个角色前重新检查双向 request。当前 request 存在时
  即便静态检查全过也只能标记 runtime-deferred，不能把瞬时空闲当未来 admission。

## P200-056—P200-064：修正后的 29+59 执行链

- 发现并消除了旧分阶段协议中的循环依赖：current59 不再等待需要十二个角色输出的
  全量 join，而只等待独立的 current29 闭环 P200-057。
- P200-056 在角色执行前冻结 current29 的完整分母：29 request、414 坐标、295 个
  有限候选、154 个物质候选；它不是纯 route-only 总体，还含 66 个 Bridge→route
  坐标。P200-057 的最早 owner 顺序显式包含 Strong、Weak、provider 的 typed refusal
  与 evidence-envelope failure，防止把拒绝或证据缺失塞入时间残差。
- P200-058 只有一个源码扫描误报，保留为 invalid。P200-059 使用 `main()` 范围检查
  后冻结相同六个 FlagGems 入口，并把恢复目标从旧 route-only 57 修正为完整 59；
  SVD 的两个新增目标来自 Bridge→route，并未新增第四种 pass。
- P200-060/P200-061 是最终 238×6178 typed join 的当前 authority；P200-062/P200-063
  是其后的规则/物化 census authority。模型 head 仍由 route 与规则式 Strong 终态
  子域决定，factor 只是剂量，不能按算子或样本生成局部回归模型。
- P200-064：4/4 typed closure 单测通过，authoritative binding drift 为 0，当前 12
  个角色报告为 0。双向 S request 仍存在，所以两个 runner 都在 GPU 初始化前返回
  75；当前不是模型失败，也没有产生任何新性能标签。

## P200-065—P200-066：模型数据门

- P200-065 已在 current 输出前冻结，沿用 P200-045 的有界模型族、三层可解释特征
  阶梯、0.005 等价带和最小参数优先规则，但执行上只接受 P200-061/P200-063。
- P200-066 只导出 exact-SASS D 行；changed-SASS 单独排入 paired retime，上游失败
  不会以缺失值、残差或机制半径进入拟合。head 只取 route 与 Strong 终态子域，factor
  是剂量；request/source 只作 group/support/provenance。
- 数据解析和污染门 4/4 单测通过。当前因全量 join/census 尚未产生而 fail-closed，
  没有生成 P200-066 输出；这证明执行顺序有效，不是数据或模型失败。

## P200-067—P200-068：可执行最小模型选择

- 每个本地 head 使用 physical request 分组的 OOF 预测，禁止同 request 候选跨训练/
  验证泄漏。模型族严格有界为 Ridge、Huber、浅层 32-tree GBRT；intercept 仅诊断。
- D 选择优先级是物质机会恢复、有害采用、错误物质声明、winner regret；随后才在
  0.005 grouped log-error 等价带中选择参数最少和字段最少的模型。这落实了“可接受
  精度下最小参数组合”。
- 小于 8 个独立 request 的 head 只报告 typed unavailable。跨 head 共享尚未执行，
  只有本地闭环后才可按相同 Strong/backend path 与 holdout 等价证据尝试。
- 拟合实现 4/4 单测通过；P200-066 缺失时 P200-068 fail-closed，没有生成空模型或
  提前打开 Q/E、残差、半径与发布。

## P200-069：单 writer 提纯

- 唯一合法执行入口已列入 P200-069；旧 V6/current57/exact-join/census 脚本仍保存
  作为历史证据，但 execution forbidden。后续不能因文件仍可运行就把它当 authority。
- 当前 29+59、全量 join/census、D dataset、本地 head fit 各只有一个合法后继；十二
  角色输出仍为 0，P200-069 是计划提纯证据而不是完成声明。

## P200-070：低搜索 L-lite 恢复口径

- 恢复分母固定为 88 个已披露物质机会，误采用分母固定为全部 238 request；不再
  只报告正例而掩盖同入口无收益样本。
- 主 L 选择前候选计时数为 0，每 request 最多物化/实测一个候选，禁止 top-N 和失败
  后 autotune 式重试。L-lite 继续按完整合法笛卡尔积的编译、计时和 wall time 作为
  搜索成本对照。
- 本对比不设 acquisition 回本门，但仍只是 D；独立 Q/E、残差、机制半径与安全发布
  不会因此被跳过。

## P200-071：条件部分共享

- 本地 head 闭环优先。共享只针对 Strong 类、当前架构、typed backend service path
  和 service 特征 schema 相同的组；局部干预/Strong 块不共享，只测试 service-context
  系数块。
- 所有成员 head 必须同时保持恢复、伤害、错误声明与 regret，并在 0.005 error 带内
  严格减少参数。失败就保留本地模型，不新增按样本身份划分的 head。

## P200-072/P200-074：校准到封存

- P200-072 因 P200-031 状态后缀写错保留 invalid；P200-074 仅把预期从 FROZEN 修为
  实际 CLOSED，成为 authority。
- 环境残差由同批 Original/control anchor 识别整体漂移；route residual 只学习减去
  anchor 后的机制相对响应；radius 只覆盖剩余单侧过预测尾部。可观测结构性偏差、
  错分支和大残差必须外层回退，不能污染三类修正量。
- Q 是与 D 家族隔离的自然 kernel，中心冻结；每 head 至少 8 个独立 request 才能
  校准有限样本 q80。E 与 D/Q 隔离且只读打开一次。S 仅提供证据方法，不提供 L 的
  Strong/Weak、数值模型或半径。

## 当前完成度与成功率判断（P200-066 冻结后）

- 已完成：完整开发总体纠错、typed evidence envelope、29+59 分阶段 authority、
  238×6178 全量 join/census 方法、顺序 Bridge→route 最小模型协议和 exact-SASS 数据
  门；共 8 项新增 CPU 单测通过，authoritative binding drift 为 0。
- 未完成：十二个 current 真实编译角色、P200-057/P200-061/P200-063 实际分类、
  changed-SASS 重测、D 最小模型拟合、88 个机会恢复、Q/E、残差/机制半径和独立安全
  非零发布。因此当前绝不能称为全部成功或论文证书完成。
- 现阶段总体成功率估计 72%–80%。其中 current 合法性/物化与论文链闭环约 86%–91%，
  低搜索恢复全部 88 个已披露机会约 76%–83%，独立未参与拟合总体上的安全非零发布
  约 64%–73%。最大未知量已经从计划污染收敛为真实 current 产物兼容性、各规则
  head 的自然样本支持和 changed-SASS 数量。
- 当前唯一运行门是双向 S request 文件仍存在。GPU1 虽无 NVML compute PID，约定仍
  要求只在文件消失后恢复 L；不得以瞬时空闲代替 admission，也不得把等待状态记成
  L 模型失败。

## P200-075—P200-103：current29 实际执行与三次外层回退

- request 是设备级互斥：有 `gpu_uuid` 时只阻止该卡；无 UUID 才全局保守阻止。临时
  资源繁忙只轮询，不再把长期目标标成 blocked。
- V8 role 3 的方法前预运行已由 P200-076 永久排除。之后所有升版均使用新路径，旧
  输出只读，未删除或覆盖。
- 第一处真实内部失败是 81 组候选 graph 坐标重复。根因不是 case 数量，而是一个
  规范化干预同时拥有“笛卡尔积臂”和“Provider subject-set 臂”两个身份。修复是规则
  级证据合并并由 schema 强制坐标唯一；V11 实测重复为 0。
- 第二处真实内部失败是 Provider 把“无终态时间投影”当成“不合法”。route-only 修复
  使有限 Provider 拒绝 240→38，exact-SASS 复用 26→228；一般合法组合修复进一步使
  拒绝 38→15、复用 228→251。两次都只发布 execution-ready/no-prediction，未使用
  历史计时做准入，也未增加 material permission。
- 最后 17 个 helper-hidden 候选不是物化缺口：17/17 当前编译成功、17/17 有 artifact、
  17/17 final SASS 与历史一致。`bridge_pipeline_body_inline` 只能限制直接闭式时间投影，
  不能否定合法性；修复后仍需 route-local 回归头，不允许沿用不适用的闭式中心。
- V14 protocol=P200-101、closure method=P200-102，均已冻结且尚未执行。两卡忙时持续
  轮询；任一卡 request/UUID 准入通过即可执行 compile/state 角色，其计时不可用于性能
  声明。
- 当前总体成功率更新为 75%–82%。合法性/物化主链的未知量显著下降；主要剩余风险是
  FlagGems59 扩展后的真实 changed-SASS 数量、各 route×Strong 子域样本支持、最小参数
  回归头的排序误差，以及 Q/E 上一次实测能否安全恢复非零物质收益。

## P200-104—P200-202：D 模型和 runtime authority 已闭环

- 当前可训练证据是 163 request/1656 candidate rows/90 material opportunities；每个
  candidate 都与当前最终 SASS 精确连接。94 个规则单元全部有真实物化见证，构造样本
  没有提供性能标签。
- P180 的概率优先选择只有 81/90；P183 依据项目语义把唯一候选选择权交还相对时间
  中心，恢复 84/90。机会概率不再覆盖时间排序，只能用于不在当前对比范围内的未来
  acquisition gate。
- 剩余软件流水 factor 错误不是状态完全不可辨识：二次剂量和 pairwise 均被否证，
  但留一请求的因果状态邻域达到 49/51。四级字段组审计后，15 个 Provider+service
  字段的 3-neighbor 局部响应回归达到 50/51；加入全局后 grouped OOF 为 87/90，且
  仍然只选择并配对实测一个候选。
- 该局部回归只替换既有 `pipeline_existing_subject_set_runtime` 头的模型族，没有新增
  因果域、子域或按 kernel 分头。它保存 80 个真实请求原型及每个 factor=2/3/4 的
  响应；更深的 launch/category 字段会退化，因此未纳入最小模型。
- Runtime 接入出现两次可审计失败：P195 首次运行没有生成 P196，因为 authority
  缺 terminal path dispatch；P198 在不重拟合的前提下将 11 个统计头映射为 17 个
  execution dispatch 单元。P200 随后因空字符串类别被错误判无效而保留 INVALID；
  通用类别投影器允许空枚举值后，P202 达成 1656/1656 supported、17/17 exercised、
  163/163 单候选。
- P202 的 89/90 是训练内执行诊断，不能替代 87/90 grouped OOF，更不能称为 Q/E。
  P198 authority 状态仍是 `DEVELOPMENT_SHORTLIST_AUTHORITY_EXPORTED_NOT_RELEASEABLE`。
- 当前成功率估计 68%–78%。D 合法性、物化、模型导出和运行执行风险已显著下降；最大
  风险转为：能否取得与五个 D source family 不相交且每个关键 head 有足够支持的自然
  Q/E，以及 Q 校准后单侧半径是否仍允许至少一个安全非零物质发布。
