# 迁移后295入口本轮执行结果与旧L-lite对照

## 结论：执行遍历完成，核心目标没有完成

本轮295个入口均已有明确执行处置，没有仍在队列中的入口：280个正确并计时，9个未通过数值参考，2个因夹具内存预算不足未执行，4个因原始代码越界隔离。后两类不是GPU测试成功。
在280个正确计时入口中，对旧L-lite加速比的恢复率达到90%的有177个（63.2%），达到95%的有139个（49.6%）。未成功的15个入口不从总目标中删除，也不计达标。
更关键的是机会覆盖：旧L-lite在本清单有228个物质收益入口，本轮其中214个可正确计时，但只恢复27个物质收益机会；占全部旧机会11.8%，占已正确计时旧机会12.6%。不能用部分入口达到90%来掩盖大量漏选。

## 本轮实际测试了什么

当前迁移代码尚未完成完整生产预测体系。这轮先固定反射填充开发数据训练的诊断均值模型，对295入口产生34个非Original选择和261个Original选择，再执行对应产物。34个优化选择全部正确，27个达到1.0175×门槛，7个略慢；其中24个属于训练入口。
Original选择也已真实执行和检查；其相对自身的加速比按1×定义，不把两次计时的噪声当收益。广播输入历史形状存在歧义的入口测试两个匹配形状，保留限制。
比较分母为旧L-lite实测最优加速比，输入探测值、批次及部分候选实现可能不同。这是用户要求的开发期直接对照，不是同时同环境严格配对，更不是独立泛化证书。没有重新跑L-lite。

## 未达成原因和已经完成的局部修复

1. 预测输入不完整：1393个候选缺字段。已修复Python丢失编译器现有启动循环次数的公共传播链，857个恢复可评分，536个仍缺字段；原2960个可评分保持。66项单元测试及3项真实编译器测试通过。此修复不改本轮冻结选择，不能提前声称改善收益。
2. 模型迁移未完成：反射训练的诊断模型不能代替所有后端响应模型。需继续恢复可解释的完整状态与响应模型、检查漏选和7个误采用。不得因未训练factor直接拒绝或按算子增补模型。
3. 原始实现/夹具责任：pooling 0/6/12、scaled-MM 143/151/153/154/155、SVD269参考失败；softmax267/268夹具内存不足；LDL60–63已知原始越界。均不得直接称为HBV服务边界或新Pass错误。
4. 剩余字段中存在条件调用和被过粗标记的不透明区域，先检查是否已有信息被丢弃，再决定公共规则修复，不能补零猜值。

## 搜索开销尚不能宣称成功

本轮均值推断累计约1.693秒，拟合约0.183秒。这不含此前全部候选编译、发现、投影、进程启动、正确性与测量，不能把它们称为总搜索开销或证明优于autotune。CTC/原地tril单独采用每次执行前重置、事件计时协议；重置不计kernel时间，但包含在诊断墙钟耗时中。

## 下一步

保留本轮不变，继续CPU修复公共预测状态及后端响应；已发现的问题做GPU局部补测，并与本轮结果分列。只有局部修复完成后再跑下一轮完整比较。共享Pass/执行链变更同步L-lite，预测变更不混入Lite。

## 全入口结果

| 入口 | kernel | GPU状态 | 旧Lite加速 | 当前加速 | 恢复率 |
|---|---|---|---:|---:|---:|
| 0 | adaptive_avg_pool2d_kernel | failed | 1.1693× | — | — |
| 1 | adaptive_avg_pool2d_kernel | measured_correct | 1.1146× | 1.0000× | 89.7% |
| 2 | adaptive_avg_pool2d_kernel | measured_correct | 1.0357× | 1.0000× | 96.6% |
| 3 | adaptive_avg_pool2d_kernel | measured_correct | 1.0097× | 1.0000× | 99.0% |
| 4 | adaptive_avg_pool2d_kernel | measured_correct | 1.3143× | 1.0000× | 76.1% |
| 5 | adaptive_avg_pool2d_kernel | measured_correct | 1.1928× | 1.0000× | 83.8% |
| 6 | adaptive_avg_pool2d_kernel | failed | 1.1667× | — | — |
| 7 | adaptive_avg_pool2d_kernel | measured_correct | 1.1667× | 1.0000× | 85.7% |
| 8 | adaptive_avg_pool2d_kernel | measured_correct | 1.0279× | 1.0000× | 97.3% |
| 9 | adaptive_avg_pool2d_kernel | measured_correct | 1.0093× | 1.0000× | 99.1% |
| 10 | adaptive_avg_pool2d_kernel | measured_correct | 1.2115× | 1.0000× | 82.5% |
| 11 | adaptive_avg_pool2d_kernel | measured_correct | 1.0828× | 1.0000× | 92.4% |
| 12 | adaptive_avg_pool2d_kernel | failed | 1.1784× | — | — |
| 13 | adaptive_avg_pool2d_kernel | measured_correct | 1.1739× | 1.0000× | 85.2% |
| 14 | adaptive_avg_pool2d_kernel | measured_correct | 1.0101× | 1.0000× | 99.0% |
| 15 | adaptive_avg_pool2d_kernel | measured_correct | 1.0050× | 1.0000× | 99.5% |
| 16 | adaptive_avg_pool2d_kernel | measured_correct | 1.3085× | 1.0000× | 76.4% |
| 17 | adaptive_avg_pool2d_kernel | measured_correct | 1.1667× | 1.0000× | 85.7% |
| 18 | _amp_foreach_non_finite_check_and_unscale_kernel | measured_correct | 1.0027× | 1.0000× | 99.7% |
| 19 | _amp_foreach_non_finite_check_and_unscale_kernel | measured_correct | 1.0325× | 1.0000× | 96.9% |
| 20 | _amp_foreach_non_finite_check_and_unscale_kernel | measured_correct | 1.0072× | 0.9979× | 99.1% |
| 21 | _amp_foreach_non_finite_check_and_unscale_kernel | measured_correct | 1.0016× | 0.9896× | 98.8% |
| 22 | _kernel_H32_D128 | measured_correct | 1.0333× | 1.0000× | 96.8% |
| 23 | broadcast_to_kernel | measured_correct | 1.0741× | 1.0000× | 93.1% |
| 24 | broadcast_to_kernel | measured_correct | 1.1541× | 1.0000× | 86.6% |
| 25 | broadcast_to_kernel | measured_correct | 1.2522× | 1.0000× | 79.9% |
| 26 | broadcast_to_kernel | measured_correct | 1.2603× | 1.0000× | 79.3% |
| 27 | conj_physical_kernel | measured_correct | 1.1486× | 1.4169× | 123.4% |
| 28 | conj_physical_kernel | measured_correct | 1.0599× | 1.0197× | 96.2% |
| 29 | conj_physical_kernel | measured_correct | 1.2800× | 1.0000× | 78.1% |
| 30 | conj_physical_kernel | measured_correct | 1.1667× | 1.2269× | 105.2% |
| 31 | conj_physical_kernel | measured_correct | 1.0833× | 1.0000× | 92.3% |
| 32 | conj_physical_kernel | measured_correct | 1.0569× | 1.0000× | 94.6% |
| 33 | conj_physical_kernel | measured_correct | 1.0608× | 1.0000× | 94.3% |
| 34 | _conv_transpose2d_direct_kernel | measured_correct | 1.0313× | 1.0000× | 97.0% |
| 35 | _conv_transpose2d_direct_kernel | measured_correct | 1.0002× | 1.0000× | 100.0% |
| 36 | _conv_transpose2d_direct_kernel | measured_correct | 1.0313× | 1.0000× | 97.0% |
| 37 | _conv_transpose2d_stride2_pad1_3x3_kernel | measured_correct | 1.1429× | 1.0000× | 87.5% |
| 38 | _conv_transpose2d_stride2_pad1_3x3_kernel | measured_correct | 1.1416× | 1.0000× | 87.6% |
| 39 | _conv_transpose2d_stride2_pad1_3x3_kernel | measured_correct | 1.2778× | 1.0000× | 78.3% |
| 40 | _conv_transpose2d_stride2_pad1_3x3_kernel | measured_correct | 1.2667× | 1.0000× | 78.9% |
| 41 | _conv_transpose2d_stride2_pad1_3x3_kernel | measured_correct | 1.3295× | 1.0000× | 75.2% |
| 42 | _conv_transpose2d_stride2_pad1_3x3_kernel | measured_correct | 1.0588× | 1.0000× | 94.4% |
| 43 | _ctc_loss_forward_full_length_reduce_kernel | measured_correct | 1.4984× | 1.0000× | 66.7% |
| 44 | _ctc_loss_forward_full_length_reduce_kernel | measured_correct | 1.5000× | 1.0000× | 66.7% |
| 45 | _ctc_loss_forward_full_length_reduce_kernel | measured_correct | 1.7077× | 1.0000× | 58.6% |
| 46 | _ctc_loss_forward_full_length_reduce_kernel | measured_correct | 1.6218× | 1.0000× | 61.7% |
| 47 | _ctc_loss_forward_full_length_reduce_kernel | measured_correct | 1.6736× | 1.0000× | 59.8% |
| 48 | _ctc_loss_forward_full_length_reduce_kernel | measured_correct | 1.6562× | 1.0000× | 60.4% |
| 49 | _ctc_loss_forward_full_length_reduce_kernel | measured_correct | 1.3317× | 1.0000× | 75.1% |
| 50 | _ctc_loss_forward_full_length_reduce_kernel | measured_correct | 1.3293× | 1.0000× | 75.2% |
| 51 | _euclidean_dist_kernel | measured_correct | 1.1989× | 1.0000× | 83.4% |
| 52 | _index_reduce_mean_finalize_kernel | measured_correct | 1.0826× | 1.0000× | 92.4% |
| 53 | _index_reduce_mean_finalize_kernel | measured_correct | 1.0588× | 0.9598× | 90.6% |
| 54 | _index_reduce_mean_finalize_kernel | measured_correct | 1.0714× | 1.0000× | 93.3% |
| 55 | mul_scalar_kernel | measured_correct | 1.0195× | 1.0000× | 98.1% |
| 56 | _linalg_eigvals_proxy_kernel | measured_correct | 1.3333× | 1.0000× | 75.0% |
| 57 | _linalg_eigvals_proxy_kernel | measured_correct | 1.3333× | 1.0000× | 75.0% |
| 58 | _linalg_eigvals_proxy_kernel | measured_correct | 1.0159× | 1.0000× | 98.4% |
| 59 | _linalg_eigvals_proxy_kernel | measured_correct | 1.0078× | 1.0000× | 99.2% |
| 60 | ldl_factor_kernel | unsafe_original_isolated | 1.1607× | — | — |
| 61 | ldl_factor_kernel | unsafe_original_isolated | 1.1116× | — | — |
| 62 | ldl_factor_kernel | unsafe_original_isolated | 1.2451× | — | — |
| 63 | ldl_factor_kernel | unsafe_original_isolated | 1.0476× | — | — |
| 64 | _pack_seq_kernel | measured_correct | 1.2461× | 1.0000× | 80.3% |
| 65 | _pack_seq_kernel | measured_correct | 1.1202× | 1.0000× | 89.3% |
| 66 | _pack_seq_kernel | measured_correct | 1.2031× | 1.0000× | 83.1% |
| 67 | _pack_seq_kernel | measured_correct | 1.1250× | 1.0000× | 88.9% |
| 68 | _pack_seq_kernel | measured_correct | 1.0647× | 1.0000× | 93.9% |
| 69 | _pack_seq_kernel | measured_correct | 1.0516× | 1.0000× | 95.1% |
| 70 | _pack_seq_kernel | measured_correct | 1.1125× | 1.0000× | 89.9% |
| 71 | _pack_seq_kernel | measured_correct | 1.2857× | 1.0000× | 77.8% |
| 72 | _pack_seq_kernel | measured_correct | 1.1250× | 1.0000× | 88.9% |
| 73 | _pdist_backward_p2_kernel | measured_correct | 1.0240× | 1.0000× | 97.7% |
| 74 | _pdist_backward_p2_kernel | measured_correct | 1.6021× | 1.0000× | 62.4% |
| 75 | _prelu_kernel_backward_kernel | measured_correct | 1.0057× | 1.0000× | 99.4% |
| 76 | _prelu_kernel_backward_kernel | measured_correct | 1.0136× | 1.0000× | 98.7% |
| 77 | _prelu_kernel_backward_kernel | measured_correct | 1.0168× | 1.0000× | 98.3% |
| 78 | _prelu_kernel_backward_kernel | measured_correct | 1.0064× | 1.0000× | 99.4% |
| 79 | _prelu_kernel_backward_kernel | measured_correct | 1.1667× | 1.0000× | 85.7% |
| 80 | reflection_pad1d_kernel | measured_correct | 1.1071× | 1.4514× | 131.1% |
| 81 | reflection_pad1d_kernel | measured_correct | 1.0039× | 1.0000× | 99.6% |
| 82 | reflection_pad1d_kernel | measured_correct | 1.0417× | 1.0453× | 100.3% |
| 83 | reflection_pad1d_kernel | measured_correct | 1.0937× | 1.0000× | 91.4% |
| 84 | reflection_pad1d_kernel | measured_correct | 1.0935× | 1.0000× | 91.5% |
| 85 | reflection_pad1d_kernel | measured_correct | 1.1071× | 1.4574× | 131.6% |
| 86 | reflection_pad1d_kernel | measured_correct | 1.3333× | 1.0000× | 75.0% |
| 87 | reflection_pad1d_kernel | measured_correct | 1.0972× | 1.0000× | 91.1% |
| 88 | reflection_pad1d_kernel | measured_correct | 1.0787× | 1.4601× | 135.4% |
| 89 | reflection_pad1d_kernel | measured_correct | 1.1429× | 1.0000× | 87.5% |
| 90 | reflection_pad1d_kernel | measured_correct | 1.0417× | 1.0619× | 101.9% |
| 91 | reflection_pad1d_kernel | measured_correct | 1.1071× | 1.4568× | 131.6% |
| 92 | reflection_pad1d_backward_kernel | measured_correct | 1.3333× | 1.0000× | 75.0% |
| 93 | reflection_pad1d_backward_kernel | measured_correct | 1.0104× | 1.0000× | 99.0% |
| 94 | reflection_pad1d_backward_kernel | measured_correct | 1.0469× | 1.0000× | 95.5% |
| 95 | reflection_pad1d_backward_kernel | measured_correct | 1.0104× | 1.0000× | 99.0% |
| 96 | reflection_pad1d_backward_kernel | measured_correct | 1.0104× | 1.0000× | 99.0% |
| 97 | reflection_pad1d_backward_kernel | measured_correct | 1.0103× | 1.0000× | 99.0% |
| 98 | reflection_pad1d_backward_kernel | measured_correct | 1.0104× | 1.0000× | 99.0% |
| 99 | reflection_pad2d_kernel | measured_correct | 1.1285× | 1.1841× | 104.9% |
| 100 | reflection_pad2d_kernel | measured_correct | 1.0180× | 1.0000× | 98.2% |
| 101 | reflection_pad2d_kernel | measured_correct | 1.0959× | 1.3902× | 126.9% |
| 102 | reflection_pad2d_kernel | measured_correct | 1.1442× | 1.0000× | 87.4% |
| 103 | reflection_pad2d_kernel | measured_correct | 1.1250× | 1.0340× | 91.9% |
| 104 | reflection_pad2d_kernel | measured_correct | 1.0165× | 1.0000× | 98.4% |
| 105 | reflection_pad2d_kernel | measured_correct | 1.0440× | 1.1188× | 107.2% |
| 106 | reflection_pad2d_kernel | measured_correct | 1.0407× | 1.0000× | 96.1% |
| 107 | reflection_pad2d_kernel | measured_correct | 1.1429× | 1.1875× | 103.9% |
| 108 | reflection_pad2d_kernel | measured_correct | 1.0186× | 1.0000× | 98.2% |
| 109 | reflection_pad2d_kernel | measured_correct | 1.1034× | 1.3887× | 125.8% |
| 110 | reflection_pad2d_kernel | measured_correct | 1.2190× | 1.0000× | 82.0% |
| 111 | reflection_pad2d_kernel | measured_correct | 1.1667× | 1.1908× | 102.1% |
| 112 | reflection_pad2d_kernel | measured_correct | 1.0093× | 1.0000× | 99.1% |
| 113 | reflection_pad2d_kernel | measured_correct | 1.0787× | 1.3902× | 128.9% |
| 114 | reflection_pad2d_kernel | measured_correct | 1.2075× | 1.0000× | 82.8% |
| 115 | reflection_pad2d_kernel | measured_correct | 1.1250× | 1.0364× | 92.1% |
| 116 | reflection_pad2d_kernel | measured_correct | 1.0072× | 1.0000× | 99.3% |
| 117 | reflection_pad2d_kernel | measured_correct | 1.0400× | 1.1306× | 108.7% |
| 118 | reflection_pad2d_kernel | measured_correct | 1.1667× | 1.1667× | 100.0% |
| 119 | reflection_pad2d_kernel | measured_correct | 1.0124× | 1.0000× | 98.8% |
| 120 | reflection_pad2d_kernel | measured_correct | 1.1254× | 1.3782× | 122.5% |
| 121 | reflection_pad3d_kernel | measured_correct | 1.1198× | 1.1757× | 105.0% |
| 122 | reflection_pad3d_kernel | measured_correct | 1.1487× | 1.1702× | 101.9% |
| 123 | reflection_pad3d_kernel | measured_correct | 1.1302× | 1.1702× | 103.5% |
| 124 | reflection_pad3d_kernel | measured_correct | 1.1939× | 1.0000× | 83.8% |
| 125 | reflection_pad3d_kernel | measured_correct | 1.0726× | 1.1733× | 109.4% |
| 126 | reflection_pad3d_kernel | measured_correct | 1.0588× | 1.0000× | 94.4% |
| 127 | reflection_pad3d_kernel | measured_correct | 1.1313× | 1.1971× | 105.8% |
| 128 | reflection_pad3d_kernel | measured_correct | 1.1302× | 1.1718× | 103.7% |
| 129 | zeros_kernel | measured_correct | 1.0260× | 1.0000× | 97.5% |
| 130 | renorm_kernel_scale | measured_correct | 1.1215× | 1.0000× | 89.2% |
| 131 | renorm_kernel_scale | measured_correct | 1.0079× | 1.0000× | 99.2% |
| 132 | renorm_kernel_scale | measured_correct | 1.2387× | 1.0000× | 80.7% |
| 133 | renorm_kernel_scale | measured_correct | 1.3955× | 1.0000× | 71.7% |
| 134 | replication_pad1d_kernel | measured_correct | 1.2050× | 1.0000× | 83.0% |
| 135 | replication_pad1d_kernel | measured_correct | 1.1679× | 1.0000× | 85.6% |
| 136 | replication_pad1d_kernel | measured_correct | 1.0469× | 1.0000× | 95.5% |
| 137 | scaled_mm_kernel | measured_correct | 1.0076× | 1.0000× | 99.2% |
| 138 | scaled_mm_kernel | measured_correct | 1.0438× | 1.0000× | 95.8% |
| 139 | scaled_mm_kernel | measured_correct | 1.1508× | 1.0000× | 86.9% |
| 140 | scaled_mm_kernel | measured_correct | 1.0179× | 1.0000× | 98.2% |
| 141 | scaled_mm_kernel | measured_correct | 1.1034× | 1.0000× | 90.6% |
| 142 | scaled_mm_kernel | measured_correct | 1.0174× | 1.0000× | 98.3% |
| 143 | scaled_mm_kernel | baseline_reference_failed | 1.0812× | — | — |
| 144 | scaled_mm_kernel | measured_correct | 1.1111× | 1.0000× | 90.0% |
| 145 | scaled_mm_kernel | measured_correct | 1.1429× | 1.0000× | 87.5% |
| 146 | scaled_mm_kernel | measured_correct | 1.2000× | 1.0000× | 83.3% |
| 147 | scaled_mm_kernel | measured_correct | 1.0059× | 1.0000× | 99.4% |
| 148 | scaled_mm_kernel | measured_correct | 1.1111× | 1.0000× | 90.0% |
| 149 | scaled_mm_kernel | measured_correct | 1.1429× | 1.0000× | 87.5% |
| 150 | scaled_mm_kernel | measured_correct | 1.2000× | 1.0000× | 83.3% |
| 151 | scaled_mm_kernel | baseline_reference_failed | 1.1111× | — | — |
| 152 | scaled_mm_kernel | measured_correct | 1.1228× | 1.0000× | 89.1% |
| 153 | scaled_mm_kernel | baseline_reference_failed | 1.0174× | — | — |
| 154 | scaled_mm_kernel | baseline_reference_failed | 1.1429× | — | — |
| 155 | scaled_mm_kernel | baseline_reference_failed | 1.1429× | — | — |
| 156 | _searchsorted_kernel | measured_correct | 1.2000× | 1.0000× | 83.3% |
| 157 | _searchsorted_kernel | measured_correct | 1.0556× | 1.0000× | 94.7% |
| 158 | _searchsorted_kernel | measured_correct | 1.0052× | 1.0000× | 99.5% |
| 159 | _searchsorted_kernel | measured_correct | 1.1750× | 1.0000× | 85.1% |
| 160 | _searchsorted_kernel | measured_correct | 1.1082× | 1.0000× | 90.2% |
| 161 | _searchsorted_kernel | measured_correct | 1.0199× | 1.0000× | 98.0% |
| 162 | _searchsorted_kernel | measured_correct | 1.0678× | 1.0000× | 93.7% |
| 163 | _searchsorted_kernel | measured_correct | 1.1250× | 1.0000× | 88.9% |
| 164 | _searchsorted_kernel | measured_correct | 1.0066× | 1.0000× | 99.3% |
| 165 | _searchsorted_kernel | measured_correct | 1.1594× | 1.0000× | 86.3% |
| 166 | _searchsorted_kernel | measured_correct | 1.1615× | 1.0000× | 86.1% |
| 167 | _searchsorted_kernel | measured_correct | 1.0375× | 1.0000× | 96.4% |
| 168 | _searchsorted_kernel | measured_correct | 1.0549× | 1.0000× | 94.8% |
| 169 | _searchsorted_kernel | measured_correct | 1.1250× | 1.0000× | 88.9% |
| 170 | _searchsorted_kernel | measured_correct | 1.0608× | 1.0000× | 94.3% |
| 171 | _searchsorted_kernel | measured_correct | 1.0159× | 1.0000× | 98.4% |
| 172 | _searchsorted_kernel | measured_correct | 1.0492× | 1.0000× | 95.3% |
| 173 | _searchsorted_kernel | measured_correct | 1.0438× | 1.0000× | 95.8% |
| 174 | _searchsorted_kernel | measured_correct | 1.1327× | 1.0000× | 88.3% |
| 175 | _searchsorted_kernel | measured_correct | 1.1827× | 1.0000× | 84.6% |
| 176 | _searchsorted_kernel | measured_correct | 1.1497× | 1.0000× | 87.0% |
| 177 | _searchsorted_kernel | measured_correct | 1.1375× | 1.0000× | 87.9% |
| 178 | _searchsorted_kernel | measured_correct | 1.1166× | 1.0000× | 89.6% |
| 179 | _searchsorted_kernel | measured_correct | 1.2000× | 1.0000× | 83.3% |
| 180 | _searchsorted_kernel | measured_correct | 1.0267× | 1.0000× | 97.4% |
| 181 | _searchsorted_kernel | measured_correct | 1.0250× | 1.0000× | 97.6% |
| 182 | _searchsorted_kernel | measured_correct | 1.1429× | 1.0000× | 87.5% |
| 183 | _searchsorted_kernel | measured_correct | 1.1563× | 1.0000× | 86.5% |
| 184 | _searchsorted_kernel | measured_correct | 1.2000× | 1.0000× | 83.3% |
| 185 | _searchsorted_kernel | measured_correct | 1.0561× | 1.0000× | 94.7% |
| 186 | _searchsorted_kernel | measured_correct | 1.2000× | 1.0000× | 83.3% |
| 187 | _searchsorted_kernel | measured_correct | 1.0063× | 1.0000× | 99.4% |
| 188 | _searchsorted_kernel | measured_correct | 1.0052× | 1.0000× | 99.5% |
| 189 | _select_backward_kernel | measured_correct | 1.0286× | 0.9819× | 95.5% |
| 190 | _select_backward_kernel | measured_correct | 1.0059× | 0.9993× | 99.3% |
| 191 | _select_backward_kernel | measured_correct | 1.0058× | 0.9801× | 97.4% |
| 192 | _smooth_l1_loss_partial_sum_kernel | measured_correct | 1.0173× | 1.0000× | 98.3% |
| 193 | _smooth_l1_loss_partial_sum_kernel | measured_correct | 1.0019× | 1.0000× | 99.8% |
| 194 | _smooth_l1_loss_partial_sum_kernel | measured_correct | 1.0048× | 1.0000× | 99.5% |
| 195 | _smooth_l1_loss_partial_sum_kernel | measured_correct | 1.0174× | 1.0000× | 98.3% |
| 196 | _smooth_l1_loss_backward_kernel | measured_correct | 1.0018× | 1.0000× | 99.8% |
| 197 | _smooth_l1_loss_backward_kernel | measured_correct | 1.0140× | 1.0000× | 98.6% |
| 198 | _smooth_l1_loss_backward_kernel | measured_correct | 1.0031× | 0.9960× | 99.3% |
| 199 | softmax_kernel_inner | measured_correct | 1.0800× | 1.0000× | 92.6% |
| 200 | softmax_kernel_inner | measured_correct | 1.0013× | 1.0000× | 99.9% |
| 201 | softmax_kernel_inner | measured_correct | 1.0300× | 1.0000× | 97.1% |
| 202 | softmax_kernel_non_inner | measured_correct | 1.0215× | 1.0000× | 97.9% |
| 203 | softmax_kernel_non_inner | measured_correct | 1.0310× | 1.0000× | 97.0% |
| 204 | softmax_kernel_non_inner | measured_correct | 1.0245× | 1.0000× | 97.6% |
| 205 | softmax_kernel_non_inner | measured_correct | 1.0165× | 1.0000× | 98.4% |
| 206 | softmax_kernel_non_inner | measured_correct | 1.0213× | 1.0000× | 97.9% |
| 207 | softmax_kernel_non_inner | measured_correct | 1.0330× | 1.0000× | 96.8% |
| 208 | softmax_backward_kernel_inner | measured_correct | 2.1689× | 1.0000× | 46.1% |
| 209 | softmax_backward_kernel_inner | measured_correct | 2.0211× | 1.0000× | 49.5% |
| 210 | softmax_backward_kernel_inner | measured_correct | 2.0208× | 1.0000× | 49.5% |
| 211 | softmax_backward_kernel_inner | measured_correct | 2.0353× | 1.0000× | 49.1% |
| 212 | softmax_backward_kernel_inner | measured_correct | 1.9423× | 1.0000× | 51.5% |
| 213 | softmax_backward_kernel_inner | measured_correct | 1.2531× | 1.0000× | 79.8% |
| 214 | softmax_backward_kernel_inner | measured_correct | 1.1447× | 1.0000× | 87.4% |
| 215 | softmax_backward_kernel_inner | measured_correct | 1.0750× | 1.0000× | 93.0% |
| 216 | softmax_backward_kernel_inner | measured_correct | 1.1039× | 1.0000× | 90.6% |
| 217 | softmax_backward_kernel_inner | measured_correct | 1.0893× | 1.0000× | 91.8% |
| 218 | softmax_backward_kernel_inner | measured_correct | 1.4241× | 1.0000× | 70.2% |
| 219 | softmax_backward_kernel_inner | measured_correct | 1.4485× | 1.0000× | 69.0% |
| 220 | softmax_backward_kernel_inner | measured_correct | 1.5217× | 1.0000× | 65.7% |
| 221 | softmax_backward_kernel_inner | measured_correct | 1.3811× | 1.0000× | 72.4% |
| 222 | softmax_backward_kernel_inner | measured_correct | 1.9614× | 1.0000× | 51.0% |
| 223 | softmax_backward_kernel_inner | measured_correct | 1.1787× | 1.0000× | 84.8% |
| 224 | softmax_backward_kernel_inner | measured_correct | 1.0414× | 1.0000× | 96.0% |
| 225 | softmax_backward_kernel_inner | measured_correct | 1.0302× | 1.0000× | 97.1% |
| 226 | softmax_backward_kernel_inner | measured_correct | 1.0610× | 1.0000× | 94.3% |
| 227 | softmax_backward_kernel_inner | measured_correct | 1.2277× | 1.0000× | 81.5% |
| 228 | softmax_backward_kernel_inner | measured_correct | 2.1422× | 1.0000× | 46.7% |
| 229 | softmax_backward_kernel_inner | measured_correct | 2.1452× | 1.0000× | 46.6% |
| 230 | softmax_backward_kernel_inner | measured_correct | 2.1519× | 1.0000× | 46.5% |
| 231 | softmax_backward_kernel_inner | measured_correct | 2.0284× | 1.0000× | 49.3% |
| 232 | softmax_backward_kernel_inner | measured_correct | 2.3223× | 1.0000× | 43.1% |
| 233 | softmax_backward_kernel_inner | measured_correct | 1.3268× | 1.0000× | 75.4% |
| 234 | softmax_backward_kernel_inner | measured_correct | 1.1520× | 1.0000× | 86.8% |
| 235 | softmax_backward_kernel_inner | measured_correct | 1.0688× | 1.0000× | 93.6% |
| 236 | softmax_backward_kernel_inner | measured_correct | 1.0922× | 1.0000× | 91.6% |
| 237 | softmax_backward_kernel_inner | measured_correct | 1.1284× | 1.0000× | 88.6% |
| 238 | softmax_backward_kernel_non_inner | measured_correct | 1.0282× | 1.0000× | 97.3% |
| 239 | softmax_backward_kernel_non_inner | measured_correct | 1.0101× | 1.0000× | 99.0% |
| 240 | softmax_backward_kernel_non_inner | measured_correct | 1.0104× | 1.0000× | 99.0% |
| 241 | softmax_backward_kernel_non_inner | measured_correct | 1.0363× | 1.0000× | 96.5% |
| 242 | softmax_backward_kernel_non_inner | measured_correct | 1.4413× | 1.0000× | 69.4% |
| 243 | softmax_backward_kernel_non_inner | measured_correct | 1.0219× | 1.0000× | 97.9% |
| 244 | softmax_backward_kernel_non_inner | measured_correct | 1.0141× | 1.0000× | 98.6% |
| 245 | softmax_backward_kernel_non_inner | measured_correct | 1.0061× | 1.0000× | 99.4% |
| 246 | softmax_backward_kernel_non_inner | measured_correct | 1.0061× | 1.0000× | 99.4% |
| 247 | softmax_backward_kernel_non_inner | measured_correct | 1.0571× | 1.0000× | 94.6% |
| 248 | softmax_backward_kernel_non_inner | measured_correct | 1.0111× | 1.0000× | 98.9% |
| 249 | softmax_backward_kernel_non_inner | measured_correct | 1.0082× | 1.0000× | 99.2% |
| 250 | softmax_backward_kernel_non_inner | measured_correct | 1.0275× | 1.0000× | 97.3% |
| 251 | softmax_backward_kernel_non_inner | measured_correct | 1.2121× | 1.0000× | 82.5% |
| 252 | softmax_backward_kernel_non_inner | measured_correct | 1.0294× | 1.0000× | 97.1% |
| 253 | softmax_backward_kernel_non_inner | measured_correct | 1.0076× | 1.0000× | 99.2% |
| 254 | softmax_backward_kernel_non_inner | measured_correct | 1.0022× | 1.0000× | 99.8% |
| 255 | softmax_backward_kernel_non_inner | measured_correct | 1.0013× | 1.0000× | 99.9% |
| 256 | softmax_backward_kernel_non_inner | measured_correct | 1.0051× | 1.0000× | 99.5% |
| 257 | softmax_backward_kernel_non_inner | measured_correct | 1.0094× | 1.0000× | 99.1% |
| 258 | softmax_backward_kernel_non_inner | measured_correct | 1.0052× | 1.0000× | 99.5% |
| 259 | softmax_backward_kernel_non_inner | measured_correct | 1.0105× | 1.0000× | 99.0% |
| 260 | softmax_backward_kernel_non_inner | measured_correct | 1.0365× | 1.0000× | 96.5% |
| 261 | softmax_backward_kernel_non_inner | measured_correct | 1.6936× | 1.0000× | 59.0% |
| 262 | softmax_backward_kernel_non_inner | measured_correct | 1.0202× | 1.0000× | 98.0% |
| 263 | softmax_backward_kernel_non_inner | measured_correct | 1.0105× | 1.0000× | 99.0% |
| 264 | softmax_backward_kernel_non_inner | measured_correct | 1.0040× | 1.0000× | 99.6% |
| 265 | softmax_backward_kernel_non_inner | measured_correct | 1.0141× | 1.0000× | 98.6% |
| 266 | softmax_kernel_inner | measured_correct | 1.2044× | 1.0000× | 83.0% |
| 267 | softmax_kernel_inner | fixture_memory_budget_exceeded | 1.0258× | — | — |
| 268 | softmax_kernel_inner | fixture_memory_budget_exceeded | 1.1985× | — | — |
| 269 | _rank2_svd_tiny_kernel | failed | 1.0323× | — | — |
| 270 | _cyclic_jacobi_init_kernel | measured_correct | 1.0417× | 1.0000× | 96.0% |
| 271 | _triton_bmm_kernel | measured_correct | 1.0894× | 1.0000× | 91.8% |
| 272 | _thnn_fused_lstm_cell_kernel | measured_correct | 1.0469× | 1.0000× | 95.5% |
| 273 | _thnn_fused_lstm_cell_kernel | measured_correct | 1.0859× | 1.0000× | 92.1% |
| 274 | _thnn_fused_lstm_cell_kernel | measured_correct | 1.2500× | 1.0000× | 80.0% |
| 275 | _tril_rows_kernel | measured_correct | 1.0039× | 1.0000× | 99.6% |
| 276 | _tril_rows_kernel | measured_correct | 1.1371× | 1.0000× | 87.9% |
| 277 | _tril_exact_row_kernel | measured_correct | 1.0435× | 1.0000× | 95.8% |
| 278 | _tril_inplace_zero_tile_kernel | measured_correct | 1.0076× | 1.0000× | 99.2% |
| 279 | _tril_inplace_zero_tile_kernel | measured_correct | 1.0032× | 1.0000× | 99.7% |
| 280 | _tril_strided_out_tile_kernel | measured_correct | 9.0547× | 1.0000× | 11.0% |
| 281 | _unsafe_masked_index_kernel | measured_correct | 1.0356× | 1.0000× | 96.6% |
| 282 | _unsafe_masked_index_kernel | measured_correct | 1.0079× | 1.0000× | 99.2% |
| 283 | _unsafe_masked_index_kernel | measured_correct | 1.1000× | 1.0000× | 90.9% |
| 284 | _unsafe_masked_index_kernel | measured_correct | 1.0357× | 1.0000× | 96.6% |
| 285 | _unsafe_masked_index_kernel | measured_correct | 1.0323× | 1.0000× | 96.9% |
| 286 | _unsafe_masked_index_kernel | measured_correct | 1.0940× | 1.0000× | 91.4% |
| 287 | _unsafe_masked_index_kernel | measured_correct | 1.1000× | 1.0000× | 90.9% |
| 288 | _upsample_nearest_exact1d_kernel | measured_correct | 1.0079× | 1.0000× | 99.2% |
| 289 | _cp_gather_indexer_quant_cache_kernel | measured_correct | 1.0458× | 1.0000× | 95.6% |
| 290 | _cp_gather_indexer_quant_cache_kernel | measured_correct | 1.1429× | 1.0000× | 87.5% |
| 291 | _cp_gather_indexer_quant_cache_kernel | measured_correct | 1.5625× | 1.0000× | 64.0% |
| 292 | _cp_gather_indexer_quant_cache_kernel | measured_correct | 1.7255× | 1.0000× | 58.0% |
| 293 | _fused_postprocess_kernel | measured_correct | 1.0458× | 1.0000× | 95.6% |
| 294 | _fused_postprocess_kernel | measured_correct | 1.0344× | 1.0000× | 96.7% |
