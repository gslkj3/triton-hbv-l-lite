# Main L 与 L-lite 已知正收益算子对比实验

> 本报告回答一个限定问题：在 L-lite 已经通过穷举式原生自动调优找到正收益的真实算子上，Main L 能否仅依靠冻结的语义模型、状态传播和窄域回归模型，在候选实测之前选择有收益的路线，并减少搜索开销？

## 结论

本轮达到预注册成功条件。

这里的“成功”是最低证据门：Main L 至少找到一个可信的物质正收益路线、没有采用已知伤害路线，并且搜索开销低于 L-lite。它不等于“已经恢复 L-lite 找到的大部分收益机会”。按后者衡量，本轮仍有明显覆盖缺口，尤其是软件流水和完全展开＋阶段重排路线尚未产生一次发布选择。

- 严格可比条目：70 / 75。
- Main L 选择非原始路线 7 次，其中历史同坐标证据显示正收益 6 次、达到 1.75% 物质收益门槛 6 次。
- 上述选择中 6 次能在冻结 L-lite 候选表中直接评分，1 次因旧版候选当时未得到有效计时而只能查看本轮算子级配对结果。
- 历史错误采用 0 次；达到至少 1% 实质退化的错误采用 0 次。
- Main L 在选择前实测替代候选 0 次；编译候选数相对 L-lite 减少 91.38%，采集时间减少 67.06%。

这里不要求 Main L 的首次优化成本必须立即回本。L-lite 自动调优也没有承担这个门槛；本实验只比较两种选择方法找到收益的能力和为此付出的搜索开销。

## 一、实验边界

本实验不是新的盲测。评测总体由历史 L-lite 结果筛出，因此只能证明 Main L 对已知真实机会的恢复能力，不能证明它对未知算子家族的泛化。L-lite 标签只用于确定评测条目和事后评分，未用于修改 Main L 的模型、系数、支持域、误差半径或决策。

每个条目使用相同仓库提交、相同 pytest 测试入口、输入构造、形状集合、原生 Triton 外层配置、候选坐标、计时协议、RTX 4090 架构和最多四个编译进程。唯一改变的是选择方式：L-lite 实测所有候选后取最快值；Main L 在候选计时之前根据冻结模型作出一次选择。

用户明确允许共享 GPU 环境中的测量噪声。因此当前 Main L 与 Original 的配对计时用于工程观察；最严格的选择质量评分采用历史 L-lite 同一轮里已经实测过的坐标：把 Main L 选择的坐标放回该次完整候选计时表，再与同表中的 L-lite 最优坐标比较。这样不会把跨时段 GPU 漂移误当作选择器能力。

Bridge-only 是候选图中的中间物化状态，不是提交给自动调优或 Main L 发布的独立路线，候选域对齐时不把它计作额外候选。

## 二、总体结果

| 指标 | 结果 |
|---|---:|
| 冻结正收益测试条目 | 75 |
| 完成双臂执行的条目 | 75 |
| 入口、请求、候选和基准点均严格对齐 | 70 |
| 严格对齐请求 | 1114 |
| 其中 L-lite 正收益请求 | 295 |
| 其中达到 1.75% 门槛的物质收益请求 | 228 |
| L-lite 正收益坐标可由当前物化器重放 | 295 |
| L-lite 正收益坐标进入 Main L 编译域 | 54 |
| L-lite 正收益坐标位于 Main L 模型支持域 | 3 |
| Main L 与 L-lite 最优坐标完全命中 | 1 |
| Main L 与物质收益最优坐标完全命中 | 1 |
| Main L 达到 L-lite 最优值 1.75% 以内 | 70 |
| Main L 在物质收益请求上达到最优值 1.75% 以内 | 3 |
| Main L 所选坐标的历史几何平均加速 | 1.0030× |
| Main L 在物质收益请求上的历史几何平均加速 | 1.0039× |
| 保留 L-lite 可得对数收益比例 | 2.23% |
| 在物质收益请求上保留的对数收益比例 | 2.26% |
| 当前共享环境配对几何平均加速 | 0.8915× |
| Main L 编译的替代候选 | 1369 |
| Main L 编译路线（含每请求一次 Original） | 2483 |
| L-lite 提交候选 | 28812 |
| Main L 选择采集时间 | 2661.326 秒 |
| L-lite 首次自动调优采集时间 | 8079.173 秒 |
| 含非原始选择的当前算子条目 | 3 |
| 其中当前无法计时判定 | 0 |
| 其中当前达到 1.75% 算子级收益 | 3 |
| 其中当前至少退化 1% | 0 |

“达到 L-lite 最优值 1.75% 以内”的 70 个请求里，大部分是 L-lite 本身只有很小收益、因此保留 Original 也落在 1.75% 范围内。真正更严格的指标是物质收益请求：228 个物质收益请求中只有 3 个达到 L-lite 最优值 1.75% 以内。这个数字不能用 70 来替代。

## 三、路线覆盖与没有命中的含义

L-lite 的最优路线分布与 Main L 实际选择分布如下。坐标可物化不等于模型已经拥有该路线的回归权威；例如软件流水、Bridge 组合路线或特殊前缀路线缺少回归支持时，应记为覆盖缺口，不能说成回归模型预测错误。

| 路线 | L-lite 正收益最优请求 | Main L 非原始选择 |
|---|---:|---:|
| 软件流水 | 125 | 0 |
| 完全展开＋逻辑向量化 | 42 | 7 |
| 完全展开＋阶段重排 | 128 | 0 |

## 四、逐测试条目结果

“当前加速”是本轮共享 GPU 环境中 Main L 对 Original 的算子级配对结果；只有严格对齐的条目才进入总体统计。冻结 L-lite 加速用于说明该条目为何进入已知正收益总体。

| 仓库 | 测试入口 | L-lite 冻结加速 | 当前 Main L 加速 | 请求（Main/冻结；额外） | Main 非原始选择 | 精确命中 | Main/L-lite 采集秒 | 状态 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| FlagGems | `benchmark/test_FLA/test_chunk_gated_delta_rule_fwd.py::test_perf_chunk_gated_delta_rule_fwd` | 1.0077× | — | 0/82；+0 | 0 | 0 | 0.000/0.000 | 排除：MAIN_L_ROLE_NOT_COMPLETE, FROZEN_REQUEST_NOT_OBSERVED, SELECTABLE_CANDIDATE_PARITY_NOT_CLOSED, BENCHMARK_POINT_PARITY_NOT_CLOSED |
| FlagGems | `benchmark/test_adaptive_avg_pool2d.py::test_perf_adaptive_avg_pool2d` | 1.0206× | 0.6355× | 106/105；+1 | 0 | 87 | 45.816/875.691 | 严格对齐 |
| FlagGems | `benchmark/test_amp_foreach_non_finite_check_and_unscale_.py::test_amp_foreach_non_finite_check_and_unscale_` | 1.0009× | 0.9239× | 57/49；+8 | 0 | 45 | 75.751/174.406 | 严格对齐 |
| FlagGems | `benchmark/test_bf16_paged_mqa_logits.py::test_bf16_paged_mqa_logits` | 1.0047× | 1.0073× | 7/7；+0 | 0 | 6 | 5.737/38.499 | 严格对齐 |
| FlagGems | `benchmark/test_broadcast_to.py::test_broadcast_to` | 1.1183× | 0.9405× | 6/6；+0 | 0 | 2 | 21.489/153.921 | 严格对齐 |
| FlagGems | `benchmark/test_conj_physical.py::test_conj_physical` | 1.0229× | 1.0935× | 36/35；+1 | 0 | 24 | 14.417/485.587 | 严格对齐 |
| FlagGems | `benchmark/test_conv_transpose2d.py::test_perf_conv_transpose2d` | 1.0491× | 1.3621× | 24/24；+0 | 0 | 15 | 23.160/149.149 | 严格对齐 |
| FlagGems | `benchmark/test_ctc_loss.py::test_ctc_loss` | 1.5333× | 1.2844× | 8/8；+0 | 2 | 0 | 104.249/103.895 | 严格对齐 |
| FlagGems | `benchmark/test_euclidean_dist.py::test_euclidean_dist` | 1.0949× | 1.7000× | 2/2；+0 | 0 | 0 | 16.032/39.537 | 严格对齐 |
| FlagGems | `benchmark/test_index_reduce.py::test_index_reduce_mean` | 1.0189× | 1.0230× | 11/11；+0 | 0 | 8 | 8.463/86.698 | 严格对齐 |
| FlagGems | `benchmark/test_linalg_cholesky.py::test_linalg_cholesky` | 1.0002× | 0.9036× | 90/90；+0 | 0 | 88 | 46.553/247.268 | 严格对齐 |
| FlagGems | `benchmark/test_linalg_eigvals.py::test_linalg_eigvals` | 1.1272× | 1.0351× | 5/5；+0 | 0 | 0 | 3.340/48.691 | 严格对齐 |
| FlagGems | `benchmark/test_linalg_ldl_factor.py::test_linalg_ldl_factor` | 1.0219× | 1.8037× | 24/24；+0 | 4 | 21 | 40.092/33.039 | 严格对齐 |
| FlagGems | `benchmark/test_mhc.py::test_mhc_pre` | 1.0097× | — | 0/40；+0 | 0 | 0 | 0.000/0.000 | 排除：MAIN_L_ROLE_NOT_COMPLETE, FROZEN_REQUEST_NOT_OBSERVED, SELECTABLE_CANDIDATE_PARITY_NOT_CLOSED, BENCHMARK_POINT_PARITY_NOT_CLOSED |
| FlagGems | `benchmark/test_pack_seq.py::test_pack_seq` | 1.0704× | 0.8840× | 18/18；+0 | 0 | 7 | 13.841/162.684 | 严格对齐 |
| FlagGems | `benchmark/test_pdist_backward.py::test_pdist_backward` | 1.0860× | 4.8984× | 6/6；+0 | 1 | 1 | 60.266/16.845 | 严格对齐 |
| FlagGems | `benchmark/test_prelu_kernel_backward.py::test_prelu_kernel_backward` | 1.0221× | 1.0559× | 9/9；+0 | 0 | 2 | 5.589/131.304 | 严格对齐 |
| FlagGems | `benchmark/test_reflection_pad1d.py::test_reflection_pad1d` | 1.0362× | 0.3427× | 12/12；+0 | 0 | 6 | 7.762/113.322 | 严格对齐 |
| FlagGems | `benchmark/test_reflection_pad1d.py::test_reflection_pad1d_out` | 1.0629× | 0.4626× | 12/12；+0 | 0 | 6 | 7.630/20.029 | 严格对齐 |
| FlagGems | `benchmark/test_reflection_pad1d_backward.py::test_reflection_pad1d_backward` | 1.0260× | 0.5234× | 18/15；+3 | 0 | 8 | 8.791/74.049 | 严格对齐 |
| FlagGems | `benchmark/test_reflection_pad2d.py::test_reflection_pad2d` | 1.0571× | 0.4111× | 15/15；+0 | 0 | 3 | 8.319/189.892 | 严格对齐 |
| FlagGems | `benchmark/test_reflection_pad2d.py::test_reflection_pad2d_out` | 1.0743× | 0.2699× | 15/15；+0 | 0 | 4 | 8.571/29.334 | 严格对齐 |
| FlagGems | `benchmark/test_reflection_pad3d.py::test_reflection_pad3d` | 1.0317× | 0.2819× | 12/12；+0 | 0 | 6 | 8.767/89.112 | 严格对齐 |
| FlagGems | `benchmark/test_reflection_pad3d.py::test_reflection_pad3d_out` | 1.0469× | 0.3292× | 12/12；+0 | 0 | 7 | 8.794/10.208 | 严格对齐 |
| FlagGems | `benchmark/test_reflection_pad3d_backward.py::test_reflection_pad3d_backward` | 1.0005× | 1.4699× | 48/48；+0 | 0 | 45 | 22.661/55.068 | 严格对齐 |
| FlagGems | `benchmark/test_renorm.py::test_renorm` | 1.0093× | 17.3277× | 72/72；+0 | 0 | 67 | 388.456/397.181 | 严格对齐 |
| FlagGems | `benchmark/test_replication_pad1d.py::test_replication_pad1d` | 1.0328× | 0.4533× | 12/12；+0 | 0 | 7 | 6.776/17.258 | 严格对齐 |
| FlagGems | `benchmark/test_scaled_mm.py::test_scaled_mm_benchmark[float8_e4m3fn-scalar-bfloat16-bias_True]` | 1.0175× | 0.9753× | 12/12；+0 | 0 | 8 | 47.041/56.495 | 严格对齐 |
| FlagGems | `benchmark/test_scaled_mm.py::test_scaled_mm_benchmark[float8_e4m3fn-scalar-float16-bias_True]` | 1.0097× | 0.9733× | 12/12；+0 | 0 | 8 | 48.802/55.838 | 严格对齐 |
| FlagGems | `benchmark/test_scaled_mm.py::test_scaled_mm_benchmark[float8_e4m3fn-scalar-float32-bias_False]` | 1.0065× | 1.0472× | 12/12；+0 | 0 | 9 | 47.545/54.121 | 严格对齐 |
| FlagGems | `benchmark/test_scaled_mm.py::test_scaled_mm_out_benchmark[float8_e4m3fn-rowwise-bfloat16-bias_True]` | 1.0201× | 0.6151× | 12/12；+0 | 0 | 9 | 49.270/61.170 | 严格对齐 |
| FlagGems | `benchmark/test_scaled_mm.py::test_scaled_mm_out_benchmark[float8_e4m3fn-scalar-bfloat16-bias_True]` | 1.0362× | 0.9355× | 12/12；+0 | 0 | 8 | 48.987/60.413 | 严格对齐 |
| FlagGems | `benchmark/test_scaled_mm.py::test_scaled_mm_out_benchmark[float8_e4m3fn-scalar-float16-bias_True]` | 1.0342× | 0.8699× | 12/12；+0 | 0 | 9 | 51.617/60.290 | 严格对齐 |
| FlagGems | `benchmark/test_scaled_mm.py::test_scaled_mm_out_benchmark[float8_e4m3fn-scalar-float32-bias_False]` | 1.0240× | 0.8419× | 12/12；+0 | 0 | 8 | 47.201/57.419 | 严格对齐 |
| FlagGems | `benchmark/test_searchsorted.py::test_searchsorted[dtype0]` | 1.0627× | 0.4260× | 3/3；+0 | 0 | 0 | 10.152/52.644 | 严格对齐 |
| FlagGems | `benchmark/test_searchsorted.py::test_searchsorted[dtype3]` | 1.0182× | 0.5443× | 3/3；+0 | 0 | 1 | 10.445/65.030 | 严格对齐 |
| FlagGems | `benchmark/test_searchsorted.py::test_searchsorted[dtype6]` | 1.0017× | 0.5782× | 3/3；+0 | 0 | 2 | 10.236/48.568 | 严格对齐 |
| FlagGems | `benchmark/test_searchsorted.py::test_searchsorted_out[dtype0]` | 1.0920× | 0.5621× | 3/3；+0 | 0 | 0 | 10.053/5.142 | 严格对齐 |
| FlagGems | `benchmark/test_searchsorted.py::test_searchsorted_out[dtype2]` | 1.0066× | 0.6853× | 3/3；+0 | 0 | 0 | 10.155/8.708 | 严格对齐 |
| FlagGems | `benchmark/test_searchsorted.py::test_searchsorted_out[dtype3]` | 1.0221× | 0.8481× | 3/3；+0 | 0 | 2 | 10.394/5.082 | 严格对齐 |
| FlagGems | `benchmark/test_searchsorted.py::test_searchsorted_out[dtype4]` | 1.0400× | 0.6402× | 3/3；+0 | 0 | 2 | 10.401/9.507 | 严格对齐 |
| FlagGems | `benchmark/test_searchsorted.py::test_searchsorted_out[dtype5]` | 1.0022× | 0.5018× | 3/3；+0 | 0 | 1 | 10.216/5.081 | 严格对齐 |
| FlagGems | `benchmark/test_searchsorted.py::test_searchsorted_out[dtype6]` | 1.1043× | 0.7426× | 3/3；+0 | 0 | 1 | 10.457/5.064 | 严格对齐 |
| FlagGems | `benchmark/test_searchsorted.py::test_searchsorted_scalar[dtype0]` | 1.0228× | 0.6062× | 4/4；+0 | 0 | 1 | 10.423/12.248 | 严格对齐 |
| FlagGems | `benchmark/test_searchsorted.py::test_searchsorted_scalar[dtype1]` | 1.0299× | 1.0120× | 4/4；+0 | 0 | 2 | 10.418/13.777 | 严格对齐 |
| FlagGems | `benchmark/test_searchsorted.py::test_searchsorted_scalar[dtype2]` | 1.0149× | 1.0219× | 4/4；+0 | 0 | 1 | 10.470/11.297 | 严格对齐 |
| FlagGems | `benchmark/test_searchsorted.py::test_searchsorted_scalar[dtype3]` | 1.0039× | 0.8104× | 4/4；+0 | 0 | 1 | 10.540/11.942 | 严格对齐 |
| FlagGems | `benchmark/test_searchsorted.py::test_searchsorted_scalar[dtype4]` | 1.0121× | 0.6283× | 4/4；+0 | 0 | 2 | 10.456/14.212 | 严格对齐 |
| FlagGems | `benchmark/test_searchsorted.py::test_searchsorted_scalar[dtype5]` | 1.0428× | 0.8774× | 4/4；+0 | 0 | 2 | 10.486/11.481 | 严格对齐 |
| FlagGems | `benchmark/test_searchsorted.py::test_searchsorted_scalar[dtype6]` | 1.0428× | 0.9235× | 4/4；+0 | 0 | 1 | 10.433/11.337 | 严格对齐 |
| FlagGems | `benchmark/test_searchsorted.py::test_searchsorted_scalar_out[dtype0]` | 1.0355× | 1.2027× | 4/4；+0 | 0 | 1 | 10.730/1.473 | 严格对齐 |
| FlagGems | `benchmark/test_searchsorted.py::test_searchsorted_scalar_out[dtype1]` | 1.0616× | 0.8684× | 4/4；+0 | 0 | 1 | 10.338/2.340 | 严格对齐 |
| FlagGems | `benchmark/test_searchsorted.py::test_searchsorted_scalar_out[dtype2]` | 1.0536× | 0.7766× | 4/4；+0 | 0 | 1 | 10.364/1.479 | 严格对齐 |
| FlagGems | `benchmark/test_searchsorted.py::test_searchsorted_scalar_out[dtype3]` | 1.0403× | 0.8605× | 4/4；+0 | 0 | 1 | 10.504/1.478 | 严格对齐 |
| FlagGems | `benchmark/test_searchsorted.py::test_searchsorted_scalar_out[dtype4]` | 1.0853× | 0.7847× | 4/4；+0 | 0 | 1 | 10.428/2.341 | 严格对齐 |
| FlagGems | `benchmark/test_searchsorted.py::test_searchsorted_scalar_out[dtype5]` | 1.0610× | 0.7941× | 4/4；+0 | 0 | 1 | 10.614/1.478 | 严格对齐 |
| FlagGems | `benchmark/test_searchsorted.py::test_searchsorted_scalar_out[dtype6]` | 1.0029× | 0.7903× | 4/4；+0 | 0 | 1 | 10.325/1.476 | 严格对齐 |
| FlagGems | `benchmark/test_select_backward.py::test_select_backward` | 1.0009× | 0.8769× | 45/42；+3 | 0 | 36 | 19.475/233.352 | 严格对齐 |
| FlagGems | `benchmark/test_smooth_l1_loss.py::test_smooth_l1_loss` | 1.0023× | 0.8166× | 18/18；+0 | 0 | 13 | 11.334/174.101 | 严格对齐 |
| FlagGems | `benchmark/test_smooth_l1_loss.py::test_smooth_l1_loss_backward` | 1.0021× | 0.9343× | 9/9；+0 | 0 | 3 | 5.868/158.271 | 严格对齐 |
| FlagGems | `benchmark/test_softmax.py::test_softmax` | 1.0171× | 0.8860× | 15/15；+0 | 0 | 5 | 269.995/591.160 | 严格对齐 |
| FlagGems | `benchmark/test_softmax.py::test_softmax_backward` | 1.1865× | 1.1914× | 75/75；+0 | 0 | 16 | 405.901/184.963 | 严格对齐 |
| FlagGems | `benchmark/test_softmax.py::test_softmax_out` | 1.1398× | 1.0606× | 15/3；+12 | 0 | 0 | 65.028/28.333 | 严格对齐 |
| FlagGems | `benchmark/test_svd.py::test_svd` | 1.0042× | 1.2574× | 52/38；+14 | 0 | 34 | 262.843/888.089 | 严格对齐 |
| FlagGems | `benchmark/test_thnn_fused_lstm_cell.py::test_thnn_fused_lstm_cell` | 1.0237× | 0.9772× | 15/15；+0 | 0 | 12 | 13.362/340.893 | 严格对齐 |
| FlagGems | `benchmark/test_tril.py::test_tril` | 1.0073× | 0.4937× | 24/24；+0 | 0 | 18 | 16.487/274.324 | 严格对齐 |
| FlagGems | `benchmark/test_tril.py::test_tril_inplace` | 1.0007× | 0.4394× | 15/15；+0 | 0 | 9 | 10.124/75.935 | 严格对齐 |
| FlagGems | `benchmark/test_tril.py::test_tril_out_sliced` | 1.1582× | 0.7175× | 15/15；+0 | 0 | 14 | 11.661/102.018 | 严格对齐 |
| FlagGems | `benchmark/test_unsafe_masked_index.py::test_unsafe_masked_index` | 1.0131× | 0.9718× | 30/30；+0 | 0 | 20 | 13.061/303.497 | 严格对齐 |
| FlagGems | `benchmark/test_upsample_nearest_exact1d.py::test_upsample_nearest_exact1d` | 1.0007× | 0.3537× | 12/12；+0 | 0 | 6 | 4.346/23.411 | 严格对齐 |
| FlagGems-vllm | `benchmark/test_cp_gather_indexer_k_quant_cache.py::test_cp_gather_indexer_k_quant_cache_benchmark` | 1.3398× | 0.7995× | 4/4；+0 | 0 | 0 | 4.912/111.098 | 严格对齐 |
| FlagGems-vllm | `benchmark/test_mhc.py::test_mhc_pre` | 1.0112× | — | 0/40；+0 | 0 | 0 | 0.000/0.000 | 排除：MAIN_L_ROLE_NOT_COMPLETE, FROZEN_REQUEST_NOT_OBSERVED, SELECTABLE_CANDIDATE_PARITY_NOT_CLOSED, BENCHMARK_POINT_PARITY_NOT_CLOSED |
| FlagGems-vllm | `benchmark/test_pack_seq.py::test_pack_seq` | 1.0303× | — | 18/18；+0 | 0 | 0 | 0.000/0.000 | 排除：SELECTABLE_CANDIDATE_PARITY_NOT_CLOSED |
| FlagGems-vllm | `benchmark/test_top_k_per_row_prefill.py::test_top_k_per_row_prefill` | 1.0099× | 0.8570× | 8/8；+0 | 0 | 6 | 6.106/138.133 | 严格对齐 |
| FlagGems-vllm | `benchmark/test_unpack_seq.py::test_unpack_seq` | 1.0291× | — | 18/18；+0 | 0 | 0 | 0.000/0.000 | 排除：SELECTABLE_CANDIDATE_PARITY_NOT_CLOSED |

## 五、Main L 的全部非原始选择

该表逐请求列出 Main L 真正采用的路线。历史坐标加速来自冻结 L-lite 同一候选计时表，不是把本轮算子级测量冒充成候选级测量。

| 仓库/测试 | Kernel | Main L 选择 | L-lite 最优 | 历史所选加速 | 历史最优加速 | 是否精确命中 |
|---|---|---|---|---:|---:|---|
| FlagGems<br>`benchmark/test_ctc_loss.py::test_ctc_loss` | `_ctc_loss_forward_full_length_reduce_kernel` | Bridge 1 × 完全展开＋逻辑向量化 16 | Bridge 1 × 完全展开＋逻辑向量化 4 | 1.2018× | 1.4984× | 否 |
| FlagGems<br>`benchmark/test_ctc_loss.py::test_ctc_loss` | `_ctc_loss_forward_full_length_reduce_kernel` | Bridge 1 × 完全展开＋逻辑向量化 16 | Bridge 1 × 完全展开＋逻辑向量化 4 | 1.2500× | 1.5000× | 否 |
| FlagGems<br>`benchmark/test_linalg_ldl_factor.py::test_linalg_ldl_factor` | `ldl_factor_kernel` | Bridge 1 × 完全展开＋逻辑向量化 16 | Bridge 1 × 完全展开＋逻辑向量化 8 | 1.1607× | 1.1607× | 否 |
| FlagGems<br>`benchmark/test_linalg_ldl_factor.py::test_linalg_ldl_factor` | `ldl_factor_kernel` | Bridge 1 × 完全展开＋逻辑向量化 16 | Bridge 1 × 软件流水 4 | 1.0733× | 1.1116× | 否 |
| FlagGems<br>`benchmark/test_linalg_ldl_factor.py::test_linalg_ldl_factor` | `ldl_factor_kernel` | Bridge 1 × 完全展开＋逻辑向量化 16 | Bridge 1 × 完全展开＋逻辑向量化 16 | 1.2451× | 1.2451× | 是 |
| FlagGems<br>`benchmark/test_linalg_ldl_factor.py::test_linalg_ldl_factor` | `ldl_factor_kernel` | Bridge 1 × 完全展开＋逻辑向量化 16 | Bridge 1 × 完全展开＋逻辑向量化 8 | 1.0476× | 1.0476× | 否 |
| FlagGems<br>`benchmark/test_pdist_backward.py::test_pdist_backward` | `_pdist_backward_p2_kernel` | Bridge 1 × 完全展开＋逻辑向量化 8 | Bridge 1 × 软件流水 3 | — | 1.0240× | 否 |

## 六、排除项与证据边界

排除不等于算子没有收益，也不等于模型已经被证伪。它只表示这一条没有满足本次严格比较所需的双臂对齐条件。

| 排除原因 | 条目数 |
|---|---:|
| `BENCHMARK_POINT_PARITY_NOT_CLOSED` | 3 |
| `FROZEN_REQUEST_NOT_OBSERVED` | 3 |
| `MAIN_L_ROLE_NOT_COMPLETE` | 3 |
| `SELECTABLE_CANDIDATE_PARITY_NOT_CLOSED` | 5 |

## 七、这次实验实际证明了什么

这次实验给出了两个同时成立、但不能混为一谈的结论。

第一，Main L 已经证明“可以工作”。它在完全不计时候选的前提下，实际选择了 7 个非 Original 路线。6 个路线在冻结的 L-lite 同一张候选计时表里有可评分记录，这 6 个全部是正收益，并且全部超过 1.75% 物质收益门槛；没有发现错误采用或至少 1% 的伤害性采用。本轮非 Original 选择分布在 `ctc_loss`、`linalg_ldl_factor` 和 `pdist_backward` 三个真实算子条目中。

第二，Main L 还没有达到“恢复 L-lite 大部分正收益机会”的目标。70 个严格可比条目包含 295 个 L-lite 正收益请求，其中 228 个属于物质收益请求；Main L 只在 7 个请求上选择非 Original，只在 3 个物质收益请求上达到 L-lite 最优值 1.75% 以内，保留的物质对数收益比例为 2.26%。L-lite 正收益最优坐标中，软件流水有 125 个、完全展开＋阶段重排有 128 个，而 Main L 对这两条路线的选择数都是 0。当前结果因此是“低搜索开销下找到了少量可信收益”，不是“收益机会恢复已经闭环”。

Main L 把严格对齐请求的编译路线从 L-lite 的 28812 个降到 2483 个，减少 91.38%；选择采集时间从 8079.173 秒降到 2661.326 秒，减少 67.06%；选择前替代候选计时为 0。这证明搜索开销优势真实存在。但这组 67.06% 只统计严格可比条目，不包含三个 900 秒超时，不能用来声称每个算子都更快。

## 八、V4 原始批次与工程补测账本

原始 V4 一次性完成 150 个原子角色：75 个 Original 全部通过；Main L 有 68 个完成、4 个类型化测试失败、3 个超时。V4 冻结后才修改执行边界，并以新的 P3L-0157 authority 只重放 4 个工程失败条目。强语义、弱语义传播、终态回归模型、系数、支持域、误差半径和 1.75% 决策门均未改变，也没有读取性能结果来决定修复方式。

| 条目 | V4 原始结果 | 规则级修复 | P157 补测结果 | 最终比较身份 |
|---|---|---|---|---|
| `conv_transpose2d` | 构造 Cut-B 计划时，多个嵌套 subject 无法唯一认证，异常逃逸 | 只拒绝无法唯一认证的候选；保留 Original 和其他可证明候选；累计记录 42 个候选级类型拒绝 | 正确性通过，24/24 冻结请求匹配，候选域和基准点对齐 | 严格可比 |
| `adaptive_avg_pool2d` | 同一类非唯一嵌套 subject 异常 | 同一条身份无关规则；累计记录 840 个候选级类型拒绝 | 正确性通过，105 个冻结请求全部匹配，另观察到 1 个当前额外请求；候选域和基准点对齐 | 严格可比 |
| FlagGems-vllm `pack_seq` | Provider 同时声称 route-capable，却没有对应 pass-service opportunity，异常逃逸 | 在 Provider 边界把矛盾事实变成类型化 Original；不构造、不编译、不执行无法证明的候选 | 正确性通过，18/18 请求匹配，全部保留 Original | 因候选域未形成而不进入严格选择比较 |
| FlagGems-vllm `unpack_seq` | 与 `pack_seq` 相同 | 与 `pack_seq` 相同 | 正确性通过，18/18 请求匹配，全部保留 Original | 因候选域未形成而不进入严格选择比较 |

这四项不是 case by case 修复。代码中没有算子名、源文件名或测试名判断：一个规则处理“候选无法唯一绑定 Provider subject”，另一个规则处理“能力证书与服务机会互相矛盾”。P157 使严格可比条目从 68 增至 70；`pack_seq` 和 `unpack_seq` 虽然不再功能失败，但没有可与 L-lite 对齐的 Main L 候选图，因此仍明确排除。

三个超时没有通过放宽预算或读取 L-lite 标签来“修好”，原始事实如下：

| 条目 | Main L | Original | 归因 |
|---|---:|---:|---|
| FlagGems `mhc_pre` | 900.556 秒后终止 | 53.796 秒完成 | 多循环入口的候选编译/分析规模超出本轮预算 |
| FlagGems-vllm `mhc_pre` | 900.619 秒后终止 | 54.259 秒完成 | 与 FlagGems 版本独立复现同一规模问题 |
| `chunk_gated_delta_rule_fwd` | 900.636 秒后终止 | 99.263 秒完成 | 复杂入口的候选编译/分析规模超出本轮预算 |

三个超时都是高 CPU 活跃直至协议终止，不是锁等待、GPU 死进程或执行器死循环。它们说明“无需候选计时”并不自动等于“采集一定便宜”：如果候选编译规模没有被语义支持域足够早地限制，Main L 仍可能花费很大。该问题属于后续覆盖与采集规模设计，不影响本轮已经选出的 6 个历史可评分正收益，但阻止我们声明全总体开销闭环。

## 九、如何解读当前配对计时

当前共享 GPU 环境下，70 个严格条目的所有基准点几何平均为 0.8915×，也就是整体上比独立 Original 臂慢。这一数字必须原样保留：它说明当前运行时接入和共享环境下的全条目稳态表现尚不能宣称整体加速。尤其是很多请求最终选择 Original，但两次独立角色的计时仍出现明显差异，不能把这些差异全部归因于所选 route。

为了判断“选择器是否选错”，本报告使用更严格的冻结同表坐标证据：Main L 所选坐标和 L-lite 最优坐标都来自同一次 L-lite 候选计时表。按这个口径，6 个可评分非 Original 选择全部为物质正收益、0 个伤害性采用。当前配对计时则用于补充检查运行时工程表现：包含非 Original 选择的 3 个算子条目本轮都达到至少 1.75% 的算子级收益，未出现当前伤害项。两类证据回答的问题不同，不能互相替代。

另外，三个超时消耗的约 2702 秒没有进入 2661.326 秒的严格可比 acquisition（采集）汇总。包含 pytest、正确性、基准点和超时在内，Main L 75 个角色的实际总墙钟约为 6522.228 秒；这个数字与 L-lite 的 10449.354 秒“首次候选采集”不是完全同范围，只能作为保守补充，不能替代严格行内的 67.06% 降幅。

## 十、可复核证据

- 正收益总体 SHA-256：`82a217ac76743213329bb9df4202475ba8ee8e587e42f84ac1830b50f7f28573`。
- 精确重放清单 SHA-256：`1c9eabbf280d358b48859e21f860e41e2f400b2dfb8b6f3c83a421854d91be8b`。
- 全量执行状态 SHA-256：`dac9a016ad4621130d249fd6480ce2b35df0250ec90eb9b2a2e0ddfd5cae799e`。
- P156 V4 原始执行状态 SHA-256：`d8ebc8f69b175885ef433b3ae1e54d8cb08c9e1a4d74e1acd258d1b48cec9ff3`。
- P157 工程补测状态 SHA-256：`861219f1ba1654923852aa72677192bc78912ee431b7056602d3ec0c2d8d00d9`。
- P157 合并状态 SHA-256：`dac9a016ad4621130d249fd6480ce2b35df0250ec90eb9b2a2e0ddfd5cae799e`。
- 最终比较 JSON SHA-256：`80a000c107bb867ad13bf6a086fbc82d38c781fe20d7867986c05f6fa5db8eb2`。
- 完整机器可读逐条结果：[main-l-vs-l-lite-positive-operators.v1.json](evidence/main-l-vs-l-lite-positive-operators.v1.json)。

最终 JSON 保留所有请求、候选坐标、物化身份、模型决定、历史坐标计时、当前基准点计时和排除原因。Markdown 是面向评审的摘要，任何汇总数字均可从 JSON 重新计算。
