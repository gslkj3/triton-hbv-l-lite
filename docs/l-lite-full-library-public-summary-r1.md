# L-lite 全量自然算子测试公开摘要

> V47（第 47 版全量证据）SHA256（内容哈希）：`900fce65cda334c59cec009f5780072d2efc429ea4eb0f9eef2c87b1ee612049`  
> V48（第 48 版完整性审计）SHA256：`060b425ca4494c35a69bb14f30f47aeba6b5b233dfea73abc2233d4a5f5ba625`

本摘要覆盖 279 个冻结 pytest（测试框架）算子实例，并且已经通过 V48 逐项 inventory（冻结测试清单）、Original（原始编译路径）对照、kernel（内核）请求身份、收益计时身份和超时证据审计。数据来自允许共享负载的开发级 GPU（图形处理器）环境，适合评估功能覆盖、收益机会和 autotune（自动实测选优）获取开销，不作为论文高精度性能结论。

## 1. 分仓库总览

| 仓库 | 算子项 | 通过项 | 优化获胜项 | 收益合格 kernel 请求/全部已观测请求 | 非 Original 获胜请求 | 收益合格请求几何平均 | 合格获胜请求几何平均 | 已闭合 acquisition（候选获取，秒） | 获取超时项 | 相对 Original 超时增量下界（秒） | 实际提交候选 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FlagGems | 239 | 170 | 83 | 3144/3202 | 419 | 1.0127 | 1.0996 | 13483.387 | 39 | 28937.014 | 59027 |
| FlagGems-vllm | 40 | 20 | 5 | 160/194 | 40 | 1.0173 | 1.0711 | 1864.580 | 6 | 4861.287 | 3827 |

收益合格请求的几何平均来自同一 native（原生）autotune 请求内 Original 与候选的直接计时，并且所在测试最终通过、角色原子闭合。失败测试中的局部计时只保留为诊断证据。超时项没有完整计时表，因此只报告获取成本下界，不产生收益。文件级超时墙钟可能已经包含该文件中此前闭合请求的 acquisition，故已闭合 acquisition 与超时下界不能直接相加。

## 2. 获胜请求几何平均大于 1× 的自然算子

| 仓库 | pytest 算子项 | 请求 | 非 Original 获胜请求 | 获胜请求几何平均 | 请求加速比最小 | 请求加速比最大 | 获胜 route（优化路线）/factor（变换因子） | 候选获取耗时（秒） |
|---|---|---|---|---|---|---|---|---|
| FlagGems | benchmark/test_tril.py::test_tril_out_sliced | 15 | 1 | 9.0547 | 1.0000 | 9.0547 | B2:l.ttir.full_unroll_phase_major.v1:F2=1 | 102.018 |
| FlagGems | benchmark/test_ctc_loss.py::test_ctc_loss | 8 | 8 | 1.5333 | 1.3293 | 1.7077 | B1:l.ttir.full_unroll_logical_group.v1:F4=6; B1:l.ttir.full_unroll_logical_group.v1:F8=2 | 103.895 |
| FlagGems-vllm | benchmark/test_cp_gather_indexer_k_quant_cache.py::test_cp_gather_indexer_k_quant_cache_benchmark | 4 | 4 | 1.3398 | 1.0458 | 1.7255 | B4:l.ttir.full_unroll_phase_major.v1:F4=2; B8:l.ttir.full_unroll_phase_major.v1:F8=2 | 111.098 |
| FlagGems | benchmark/test_softmax.py::test_softmax_backward | 75 | 59 | 1.2428 | 1.0000 | 2.3223 | B1:l.nvidia.software_pipeline.v1:F2=6; B1:l.nvidia.software_pipeline.v1:F3=8; B1:l.nvidia.software_pipeline.v1:F4=45 | 184.963 |
| FlagGems | benchmark/test_broadcast_to.py::test_broadcast_to | 6 | 4 | 1.1827 | 1.0000 | 1.2603 | B1:l.ttir.full_unroll_phase_major.v1:F16=4 | 153.921 |
| FlagGems | benchmark/test_searchsorted.py::test_searchsorted_out[dtype6] | 3 | 2 | 1.1604 | 1.0000 | 1.1615 | B1:l.ttir.full_unroll_phase_major.v1:F2=1; B1:l.ttir.full_unroll_phase_major.v1:F8=1 | 5.064 |
| FlagGems | benchmark/test_scaled_mm.py::test_scaled_mm_out_benchmark[float8_e4m3fn-scalar-float16-bias_True] | 12 | 3 | 1.1440 | 1.0000 | 1.2000 | B1:l.nvidia.software_pipeline.v1:F3=1; B1:l.nvidia.software_pipeline.v1:F4=1; B1:l.ttir.full_unroll_phase_major.v1:F2=1 | 60.290 |
| FlagGems | benchmark/test_renorm.py::test_renorm | 72 | 5 | 1.1434 | 1.0000 | 1.3955 | B1:l.nvidia.software_pipeline.v1:F3=2; B1:l.nvidia.software_pipeline.v1:F4=3 | 397.181 |
| FlagGems | benchmark/test_softmax.py::test_softmax_out | 3 | 3 | 1.1398 | 1.0258 | 1.2044 | B1:l.nvidia.software_pipeline.v1:F3=1; B1:l.nvidia.software_pipeline.v1:F4=2 | 28.333 |
| FlagGems | benchmark/test_linalg_ldl_factor.py::test_linalg_ldl_factor | 24 | 4 | 1.1390 | 1.0000 | 1.2451 | B1:l.nvidia.software_pipeline.v1:F4=1; B1:l.ttir.full_unroll_logical_group.v1:F8=2; B1:l.ttir.full_unroll_logical_group.v1:F16=1 | 33.039 |
| FlagGems | benchmark/test_conv_transpose2d.py::test_perf_conv_transpose2d | 24 | 9 | 1.1365 | 1.0000 | 1.3295 | B1:l.nvidia.software_pipeline.v1:F2=5; B1:l.nvidia.software_pipeline.v1:F3=1; B1:l.ttir.full_unroll_phase_major.v1:F2=3 | 149.149 |
| FlagGems | benchmark/test_reflection_pad1d.py::test_reflection_pad1d_out | 12 | 6 | 1.1298 | 1.0000 | 1.3333 | B2:l.ttir.full_unroll_phase_major.v1:F2=1; B4:l.ttir.full_unroll_phase_major.v1:F4=1; B8:l.ttir.full_unroll_phase_major.v1:F8=1; B16:l.ttir.full_unroll_phase_major.v1:F16=3 | 20.029 |
| FlagGems | benchmark/test_linalg_eigvals.py::test_linalg_eigvals | 5 | 5 | 1.1272 | 1.0000 | 1.3333 | B2:l.ttir.full_unroll_logical_group.v1:F2=1; B2:l.ttir.full_unroll_phase_major.v1:F2=1; B4:l.ttir.full_unroll_logical_group.v1:F4=1; B4:l.ttir.full_unroll_phase_major.v1:F4=1; B16:l.ttir.full_unroll_phase_major.v1:F16=1 | 48.691 |
| FlagGems | benchmark/test_adaptive_avg_pool2d.py::test_perf_adaptive_avg_pool2d | 105 | 18 | 1.1261 | 1.0000 | 1.3143 | B1:l.ttir.full_unroll_logical_group.v1:F2=3; B1:l.ttir.full_unroll_phase_major.v1:F2=15 | 875.691 |
| FlagGems | benchmark/test_searchsorted.py::test_searchsorted_out[dtype4] | 3 | 1 | 1.1250 | 1.0000 | 1.1250 | B1:l.ttir.full_unroll_phase_major.v1:F2=1 | 9.507 |
| FlagGems | benchmark/test_thnn_fused_lstm_cell.py::test_thnn_fused_lstm_cell | 15 | 3 | 1.1243 | 1.0000 | 1.2500 | B2:l.nvidia.software_pipeline.v1:F3=1; B2:l.nvidia.software_pipeline.v1:F4=2 | 340.893 |
| FlagGems | benchmark/test_pack_seq.py::test_pack_seq | 18 | 11 | 1.1178 | 1.0000 | 1.2857 | B2:l.nvidia.software_pipeline.v1:F3=3; B4:l.nvidia.software_pipeline.v1:F3=3; B4:l.nvidia.software_pipeline.v1:F4=2; B8:l.nvidia.software_pipeline.v1:F3=1; B8:l.nvidia.software_pipeline.v1:F4=2 | 162.684 |
| FlagGems | benchmark/test_reflection_pad3d.py::test_reflection_pad3d_out | 12 | 5 | 1.1163 | 1.0000 | 1.1939 | B2:l.ttir.full_unroll_logical_group.v1:F2=1; B2:l.ttir.full_unroll_phase_major.v1:F2=1; B8:l.ttir.full_unroll_logical_group.v1:F8=2; B8:l.ttir.full_unroll_phase_major.v1:F8=1 | 10.208 |
| FlagGems | benchmark/test_searchsorted.py::test_searchsorted_scalar_out[dtype4] | 4 | 3 | 1.1153 | 1.0000 | 1.2000 | B1:l.ttir.full_unroll_phase_major.v1:F2=1; B1:l.ttir.full_unroll_phase_major.v1:F8=2 | 2.341 |
| FlagGems | benchmark/test_scaled_mm.py::test_scaled_mm_out_benchmark[float8_e4m3fn-scalar-bfloat16-bias_True] | 12 | 4 | 1.1127 | 1.0000 | 1.2000 | B1:l.nvidia.software_pipeline.v1:F4=2; B1:l.ttir.full_unroll_phase_major.v1:F2=2 | 60.413 |
| FlagGems | benchmark/test_pdist_backward.py::test_pdist_backward | 6 | 5 | 1.1041 | 1.0000 | 1.6021 | B1:l.nvidia.software_pipeline.v1:F3=2; B1:l.nvidia.software_pipeline.v1:F4=3 | 16.845 |
| FlagGems | benchmark/test_reflection_pad2d.py::test_reflection_pad2d_out | 15 | 11 | 1.1026 | 1.0000 | 1.2190 | B2:l.nvidia.software_pipeline.v1:F2=1; B2:l.ttir.full_unroll_phase_major.v1:F2=1; B4:l.ttir.full_unroll_phase_major.v1:F4=1; B8:l.ttir.full_unroll_logical_group.v1:F8=2; B8:l.ttir.full_unroll_phase_major.v1:F2=1; B8:l.ttir.full_unroll_phase_major.v1:F4=1; B16:l.ttir.full_unroll_phase_major.v1:F2=1; B16:l.ttir.full_unroll_phase_major.v1:F16=3 | 29.334 |
| FlagGems | benchmark/test_euclidean_dist.py::test_euclidean_dist | 2 | 2 | 1.0949 | 1.0000 | 1.1989 | B2:l.ttir.full_unroll_phase_major.v1:F2=1; B4:l.ttir.full_unroll_phase_major.v1:F4=1 | 39.537 |
| FlagGems | benchmark/test_searchsorted.py::test_searchsorted_out[dtype0] | 3 | 3 | 1.0920 | 1.0000 | 1.1750 | B1:l.ttir.full_unroll_phase_major.v1:F4=2; B1:l.ttir.full_unroll_phase_major.v1:F8=1 | 5.142 |
| FlagGems | benchmark/test_searchsorted.py::test_searchsorted_scalar[dtype5] | 4 | 2 | 1.0873 | 1.0000 | 1.1327 | B1:l.ttir.full_unroll_phase_major.v1:F2=2 | 11.481 |
| FlagGems | benchmark/test_searchsorted.py::test_searchsorted_scalar_out[dtype1] | 4 | 3 | 1.0830 | 1.0000 | 1.1375 | B1:l.ttir.full_unroll_phase_major.v1:F4=3 | 2.340 |
| FlagGems | benchmark/test_scaled_mm.py::test_scaled_mm_out_benchmark[float8_e4m3fn-rowwise-bfloat16-bias_True] | 12 | 3 | 1.0829 | 1.0000 | 1.1429 | B1:l.nvidia.software_pipeline.v1:F4=1; B1:l.ttir.full_unroll_phase_major.v1:F2=2 | 61.170 |
| FlagGems | benchmark/test_searchsorted.py::test_searchsorted_scalar_out[dtype5] | 4 | 3 | 1.0822 | 1.0000 | 1.2000 | B1:l.ttir.full_unroll_phase_major.v1:F2=1; B1:l.ttir.full_unroll_phase_major.v1:F8=2 | 1.478 |
| FlagGems | benchmark/test_replication_pad1d.py::test_replication_pad1d | 12 | 5 | 1.0806 | 1.0000 | 1.2050 | B2:l.nvidia.software_pipeline.v1:F3=1; B2:l.ttir.full_unroll_logical_group.v1:F2=3; B2:l.ttir.full_unroll_phase_major.v1:F2=1 | 17.258 |
| FlagGems-vllm | benchmark/test_unpack_seq.py::test_unpack_seq | 18 | 7 | 1.0764 | 1.0000 | 1.2800 | B1:l.ttir.predicated_exact_prefix_reduction.v1:F1=7 | 18.485 |
| FlagGems | benchmark/test_conj_physical.py::test_conj_physical | 35 | 11 | 1.0747 | 1.0000 | 1.2800 | B2:l.nvidia.software_pipeline.v1:F3=5; B2:l.ttir.full_unroll_logical_group.v1:F2=1; B2:l.ttir.full_unroll_phase_major.v1:F2=1; B4:l.nvidia.software_pipeline.v1:F3=1; B4:l.ttir.full_unroll_phase_major.v1:F4=2; B8:l.nvidia.software_pipeline.v1:F4=1 | 485.587 |
| FlagGems | benchmark/test_reflection_pad1d.py::test_reflection_pad1d | 12 | 6 | 1.0738 | 1.0000 | 1.1071 | B2:l.ttir.full_unroll_logical_group.v1:F2=2; B4:l.ttir.full_unroll_logical_group.v1:F4=1; B16:l.ttir.full_unroll_phase_major.v1:F16=3 | 113.322 |
| FlagGems | benchmark/test_scaled_mm.py::test_scaled_mm_out_benchmark[float8_e4m3fn-scalar-float32-bias_False] | 12 | 4 | 1.0737 | 1.0000 | 1.1429 | B1:l.nvidia.software_pipeline.v1:F3=1; B1:l.nvidia.software_pipeline.v1:F4=2; B1:l.ttir.full_unroll_phase_major.v1:F2=1 | 57.419 |
| FlagGems | benchmark/test_searchsorted.py::test_searchsorted_scalar_out[dtype2] | 4 | 3 | 1.0720 | 1.0000 | 1.2000 | B1:l.ttir.full_unroll_phase_major.v1:F2=1; B1:l.ttir.full_unroll_phase_major.v1:F4=2 | 1.479 |
| FlagGems | benchmark/test_reflection_pad2d.py::test_reflection_pad2d | 15 | 12 | 1.0718 | 1.0000 | 1.1442 | B2:l.ttir.full_unroll_logical_group.v1:F2=1; B4:l.nvidia.software_pipeline.v1:F2=1; B4:l.ttir.full_unroll_logical_group.v1:F4=2; B4:l.ttir.full_unroll_phase_major.v1:F2=1; B8:l.ttir.full_unroll_phase_major.v1:F2=1; B8:l.ttir.full_unroll_phase_major.v1:F8=1; B16:l.ttir.full_unroll_logical_group.v1:F16=1; B16:l.ttir.full_unroll_phase_major.v1:F8=1; B16:l.ttir.full_unroll_phase_major.v1:F16=3 | 189.892 |
| FlagGems | benchmark/test_index_reduce.py::test_index_reduce_mean | 11 | 3 | 1.0709 | 1.0000 | 1.0826 | B2:l.ttir.full_unroll_phase_major.v1:F2=2; B4:l.ttir.full_unroll_phase_major.v1:F4=1 | 86.698 |
| FlagGems | benchmark/test_searchsorted.py::test_searchsorted_out[dtype3] | 3 | 1 | 1.0678 | 1.0000 | 1.0678 | B1:l.ttir.full_unroll_phase_major.v1:F8=1 | 5.082 |
| FlagGems | benchmark/test_reflection_pad3d.py::test_reflection_pad3d | 12 | 6 | 1.0644 | 1.0000 | 1.1487 | B2:l.ttir.full_unroll_logical_group.v1:F2=1; B2:l.ttir.full_unroll_phase_major.v1:F2=1; B4:l.ttir.full_unroll_logical_group.v1:F4=1; B8:l.ttir.full_unroll_logical_group.v1:F8=2; B8:l.ttir.full_unroll_phase_major.v1:F8=1 | 89.112 |
| FlagGems | benchmark/test_searchsorted.py::test_searchsorted[dtype0] | 3 | 3 | 1.0627 | 1.0000 | 1.2000 | B1:l.ttir.full_unroll_phase_major.v1:F2=1; B1:l.ttir.full_unroll_phase_major.v1:F4=1; B1:l.ttir.full_unroll_phase_major.v1:F8=1 | 52.644 |
| FlagGems | benchmark/test_searchsorted.py::test_searchsorted_scalar[dtype1] | 4 | 2 | 1.0607 | 1.0000 | 1.1250 | B1:l.ttir.full_unroll_phase_major.v1:F8=2 | 13.777 |
| FlagGems | benchmark/test_searchsorted.py::test_searchsorted_scalar[dtype6] | 4 | 3 | 1.0575 | 1.0000 | 1.1827 | B1:l.ttir.full_unroll_phase_major.v1:F2=1; B1:l.ttir.full_unroll_phase_major.v1:F4=1; B1:l.ttir.full_unroll_phase_major.v1:F8=1 | 11.337 |
| FlagGems | benchmark/test_reflection_pad1d_backward.py::test_reflection_pad1d_backward | 15 | 7 | 1.0566 | 1.0000 | 1.3333 | B2:l.ttir.full_unroll_logical_group.v1:F2=3; B2:l.ttir.full_unroll_phase_major.v1:F2=2; B4:l.ttir.full_unroll_logical_group.v1:F4=1; B4:l.ttir.full_unroll_phase_major.v1:F4=1 | 74.049 |
| FlagGems | benchmark/test_searchsorted.py::test_searchsorted_scalar_out[dtype3] | 4 | 3 | 1.0542 | 1.0000 | 1.1429 | B1:l.ttir.full_unroll_phase_major.v1:F2=1; B1:l.ttir.full_unroll_phase_major.v1:F4=1; B1:l.ttir.full_unroll_phase_major.v1:F8=1 | 1.478 |
| FlagGems | benchmark/test_scaled_mm.py::test_scaled_mm_benchmark[float8_e4m3fn-scalar-bfloat16-bias_True] | 12 | 4 | 1.0535 | 1.0000 | 1.1508 | B1:l.nvidia.software_pipeline.v1:F2=1; B1:l.nvidia.software_pipeline.v1:F4=2; B1:l.ttir.full_unroll_phase_major.v1:F2=1 | 56.495 |
| FlagGems | benchmark/test_searchsorted.py::test_searchsorted_scalar_out[dtype0] | 4 | 3 | 1.0476 | 1.0000 | 1.1497 | B1:l.ttir.full_unroll_phase_major.v1:F4=2; B1:l.ttir.full_unroll_phase_major.v1:F8=1 | 1.473 |
| FlagGems | benchmark/test_svd.py::test_svd | 38 | 4 | 1.0403 | 1.0000 | 1.0894 | B1:l.nvidia.software_pipeline.v1:F4=1; B2:l.ttir.full_unroll_phase_major.v1:F2=1; B4:l.ttir.full_unroll_logical_group.v1:F2=1; B4:l.ttir.full_unroll_logical_group.v1:F4=1 | 888.089 |
| FlagGems | benchmark/test_FLA/test_chunk_gated_delta_rule_fwd.py::test_perf_chunk_gated_delta_rule_fwd | 82 | 16 | 1.0401 | 1.0000 | 1.1429 | B1:l.nvidia.software_pipeline.v1:F3=2; B1:l.nvidia.software_pipeline.v1:F4=14 | 692.354 |
| FlagGems-vllm | benchmark/test_top_k_per_row_prefill.py::test_top_k_per_row_prefill | 8 | 2 | 1.0400 | 1.0000 | 1.0458 | B4:l.ttir.full_unroll_logical_group.v1:F4=1; B8:l.ttir.full_unroll_logical_group.v1:F8=1 | 138.133 |
| FlagGems | benchmark/test_unsafe_masked_index.py::test_unsafe_masked_index | 30 | 10 | 1.0398 | 1.0000 | 1.1000 | B2:l.ttir.full_unroll_phase_major.v1:F2=4; B4:l.ttir.full_unroll_logical_group.v1:F4=1; B4:l.ttir.full_unroll_phase_major.v1:F4=3; B16:l.ttir.full_unroll_phase_major.v1:F2=2 | 303.497 |
| FlagGems-vllm | benchmark/test_pack_seq.py::test_pack_seq | 18 | 14 | 1.0391 | 1.0000 | 1.1667 | B1:l.ttir.predicated_exact_prefix_reduction.v1:F1=7; B2:l.nvidia.software_pipeline.v1:F2=4; B2:l.nvidia.software_pipeline.v1:F4=2; B4:l.ttir.full_unroll_phase_major.v1:F2=1 | 157.496 |
| FlagGems-vllm | benchmark/test_mhc.py::test_mhc_pre | 40 | 13 | 1.0348 | 1.0000 | 1.1176 | B1:l.nvidia.software_pipeline.v1:F2=1; B1:l.nvidia.software_pipeline.v1:F3=4; B1:l.nvidia.software_pipeline.v1:F4=4; B2:l.ttir.full_unroll_phase_major.v1:F2=4 | 724.581 |
| FlagGems | benchmark/test_bf16_paged_mqa_logits.py::test_bf16_paged_mqa_logits | 7 | 1 | 1.0333 | 1.0000 | 1.0333 | B2:l.ttir.full_unroll_phase_major.v1:F2=1 | 38.499 |
| FlagGems | benchmark/test_mhc.py::test_mhc_pre | 40 | 12 | 1.0326 | 1.0000 | 1.0625 | B1:l.nvidia.software_pipeline.v1:F2=1; B1:l.nvidia.software_pipeline.v1:F3=5; B1:l.nvidia.software_pipeline.v1:F4=3; B2:l.ttir.full_unroll_phase_major.v1:F2=3 | 777.266 |
| FlagGems | benchmark/test_searchsorted.py::test_searchsorted_scalar[dtype0] | 4 | 3 | 1.0306 | 1.0000 | 1.0549 | B1:l.ttir.full_unroll_phase_major.v1:F8=3 | 12.248 |
| FlagGems | benchmark/test_tril.py::test_tril | 24 | 6 | 1.0296 | 1.0000 | 1.1371 | B2:l.nvidia.software_pipeline.v1:F4=1; B2:l.ttir.full_unroll_phase_major.v1:F2=4; B4:l.ttir.full_unroll_phase_major.v1:F4=1 | 274.324 |
| FlagGems | benchmark/test_scaled_mm.py::test_scaled_mm_benchmark[float8_e4m3fn-scalar-float16-bias_True] | 12 | 4 | 1.0293 | 1.0000 | 1.1034 | B1:l.nvidia.software_pipeline.v1:F4=2; B1:l.ttir.full_unroll_phase_major.v1:F2=2 | 55.838 |
| FlagGems | benchmark/test_prelu_kernel_backward.py::test_prelu_kernel_backward | 9 | 7 | 1.0285 | 1.0000 | 1.1667 | B2:l.nvidia.software_pipeline.v1:F3=1; B2:l.nvidia.software_pipeline.v1:F4=5; B8:l.ttir.full_unroll_phase_major.v1:F8=1 | 131.304 |
| FlagGems | benchmark/test_searchsorted.py::test_searchsorted[dtype3] | 3 | 2 | 1.0274 | 1.0000 | 1.0556 | B1:l.ttir.full_unroll_phase_major.v1:F2=1; B1:l.ttir.full_unroll_phase_major.v1:F8=1 | 65.030 |
| FlagGems | benchmark/test_scaled_mm.py::test_scaled_mm_benchmark[float8_e4m3fn-scalar-float32-bias_False] | 12 | 3 | 1.0264 | 1.0000 | 1.0812 | B1:l.nvidia.software_pipeline.v1:F4=1; B1:l.ttir.full_unroll_phase_major.v1:F2=2 | 54.121 |
| FlagGems | benchmark/test_softmax.py::test_softmax | 15 | 10 | 1.0257 | 1.0000 | 1.0800 | B1:l.nvidia.software_pipeline.v1:F2=2; B1:l.nvidia.software_pipeline.v1:F4=6; B2:l.ttir.full_unroll_phase_major.v1:F2=2 | 591.160 |
| FlagGems | benchmark/test_searchsorted.py::test_searchsorted_scalar[dtype4] | 4 | 2 | 1.0243 | 1.0000 | 1.0492 | B1:l.ttir.full_unroll_phase_major.v1:F8=2 | 14.212 |
| FlagGems | benchmark/test_searchsorted.py::test_searchsorted_scalar[dtype2] | 4 | 3 | 1.0199 | 1.0000 | 1.0608 | B1:l.ttir.full_unroll_phase_major.v1:F2=1; B1:l.ttir.full_unroll_phase_major.v1:F8=2 | 11.297 |
| FlagGems | benchmark/test_amp_foreach_non_finite_check_and_unscale_.py::test_amp_foreach_non_finite_check_and_unscale_ | 49 | 4 | 1.0109 | 1.0000 | 1.0325 | B2:l.nvidia.software_pipeline.v1:F4=2; B2:l.ttir.full_unroll_phase_major.v1:F2=1; B4:l.ttir.full_unroll_logical_group.v1:F4=1 | 174.406 |
| FlagGems | benchmark/test_linalg_cholesky.py::test_linalg_cholesky | 90 | 2 | 1.0097 | 1.0000 | 1.0195 | B2:l.ttir.full_unroll_phase_major.v1:F2=2 | 247.268 |
| FlagGems | benchmark/test_reflection_pad3d_backward.py::test_reflection_pad3d_backward | 48 | 3 | 1.0086 | 1.0000 | 1.0260 | B2:l.ttir.full_unroll_logical_group.v1:F2=1; B2:l.ttir.full_unroll_phase_major.v1:F2=1; B4:l.ttir.full_unroll_phase_major.v1:F2=1 | 55.068 |
| FlagGems | benchmark/test_smooth_l1_loss.py::test_smooth_l1_loss | 18 | 5 | 1.0083 | 1.0000 | 1.0174 | B2:l.ttir.full_unroll_logical_group.v1:F2=1; B2:l.ttir.full_unroll_phase_major.v1:F2=3; B4:l.nvidia.software_pipeline.v1:F4=1 | 174.101 |
| FlagGems | benchmark/test_select_backward.py::test_select_backward | 42 | 6 | 1.0067 | 1.0000 | 1.0286 | B2:l.nvidia.software_pipeline.v1:F4=1; B2:l.ttir.full_unroll_phase_major.v1:F2=1; B4:l.ttir.full_unroll_phase_major.v1:F2=1; B8:l.ttir.full_unroll_logical_group.v1:F8=1; B8:l.ttir.full_unroll_phase_major.v1:F8=2 | 233.352 |
| FlagGems | benchmark/test_searchsorted.py::test_searchsorted_out[dtype2] | 3 | 3 | 1.0066 | 1.0000 | 1.0199 | B1:l.ttir.full_unroll_phase_major.v1:F4=1; B1:l.ttir.full_unroll_phase_major.v1:F8=2 | 8.708 |
| FlagGems | benchmark/test_searchsorted.py::test_searchsorted_scalar[dtype3] | 4 | 3 | 1.0053 | 1.0000 | 1.0159 | B1:l.ttir.full_unroll_phase_major.v1:F2=1; B1:l.ttir.full_unroll_phase_major.v1:F8=2 | 11.942 |
| FlagGems | benchmark/test_searchsorted.py::test_searchsorted[dtype6] | 3 | 1 | 1.0052 | 1.0000 | 1.0052 | B1:l.ttir.full_unroll_phase_major.v1:F2=1 | 48.568 |
| FlagGems | benchmark/test_searchsorted.py::test_searchsorted_scalar_out[dtype6] | 4 | 3 | 1.0038 | 1.0000 | 1.0063 | B1:l.ttir.full_unroll_phase_major.v1:F2=2; B1:l.ttir.full_unroll_phase_major.v1:F8=1 | 1.476 |
| FlagGems | benchmark/test_searchsorted.py::test_searchsorted_out[dtype5] | 3 | 2 | 1.0033 | 1.0000 | 1.0066 | B1:l.ttir.full_unroll_phase_major.v1:F4=1; B1:l.ttir.full_unroll_phase_major.v1:F8=1 | 5.081 |
| FlagGems | benchmark/test_smooth_l1_loss.py::test_smooth_l1_loss_backward | 9 | 6 | 1.0031 | 1.0000 | 1.0140 | B2:l.nvidia.software_pipeline.v1:F3=2; B2:l.nvidia.software_pipeline.v1:F4=3; B2:l.ttir.full_unroll_phase_major.v1:F2=1 | 158.271 |
| FlagGems | benchmark/test_tril.py::test_tril_inplace | 15 | 6 | 1.0018 | 1.0000 | 1.0076 | B2:l.ttir.full_unroll_phase_major.v1:F2=3; B4:l.ttir.full_unroll_phase_major.v1:F4=1; B8:l.ttir.full_unroll_phase_major.v1:F4=1; B8:l.ttir.full_unroll_phase_major.v1:F8=1 | 75.935 |
| FlagGems | benchmark/test_upsample_nearest_exact1d.py::test_upsample_nearest_exact1d | 12 | 6 | 1.0013 | 1.0000 | 1.0079 | B2:l.nvidia.software_pipeline.v1:F3=1; B2:l.ttir.full_unroll_phase_major.v1:F2=3; B4:l.ttir.full_unroll_logical_group.v1:F4=1; B4:l.ttir.full_unroll_phase_major.v1:F4=1 | 23.411 |

## 3. 候选获取超时逐项账本

| 仓库 | pytest 算子项 | Original 耗时（秒） | L-lite 边界（秒） | 增量下界（秒） | 冷缓存 cubin（CUDA（GPU 计算平台）二进制）/文件/字节 | 归因 |
|---|---|---|---|---|---|---|
| FlagGems | benchmark/test_avg_pool2d.py::test_avg_pool2d | 79.684 | 900.313 | 820.629 | 1331/10658/393149071 | Original 在边界内终止，L-lite 因候选编译/实测未在 900 秒内原子完成；这是获取开销证据，不产生收益标签。 |
| FlagGems | benchmark/test_avg_pool2d.py::test_avg_pool2d_backward | 73.875 | 900.302 | 826.427 | 1392/11147/410943711 | Original 在边界内终止，L-lite 因候选编译/实测未在 900 秒内原子完成；这是获取开销证据，不产生收益标签。 |
| FlagGems | benchmark/test_baddbmm.py::test_baddbmm | 105.063 | 900.368 | 795.305 | 818/6935/365147678 | Original 在边界内终止，L-lite 因候选编译/实测未在 900 秒内原子完成；这是获取开销证据，不产生收益标签。 |
| FlagGems | benchmark/test_baddbmm.py::test_baddbmm_out | 94.836 | 900.275 | 805.439 | 891/7546/389495474 | Original 在边界内终止，L-lite 因候选编译/实测未在 900 秒内原子完成；这是获取开销证据，不产生收益标签。 |
| FlagGems | benchmark/test_col2im.py::test_col2im | 92.017 | 900.299 | 808.282 | 561/4496/339038621 | Original 在边界内终止，L-lite 因候选编译/实测未在 900 秒内原子完成；这是获取开销证据，不产生收益标签。 |
| FlagGems | benchmark/test_conv1d.py::test_conv1d | 627.917 | 900.276 | 272.359 | 804/6438/460212946 | Original 在边界内终止，L-lite 因候选编译/实测未在 900 秒内原子完成；这是获取开销证据，不产生收益标签。 |
| FlagGems | benchmark/test_conv1d.py::test_conv1d_padding | 440.126 | 900.336 | 460.210 | 813/6508/478856530 | Original 在边界内终止，L-lite 因候选编译/实测未在 900 秒内原子完成；这是获取开销证据，不产生收益标签。 |
| FlagGems | benchmark/test_conv2d.py::test_conv2d | 578.654 | 900.251 | 321.597 | 385/3084/337253746 | Original 在边界内终止，L-lite 因候选编译/实测未在 900 秒内原子完成；这是获取开销证据，不产生收益标签。 |
| FlagGems | benchmark/test_conv2d.py::test_conv2d_padding | 417.294 | 900.351 | 483.057 | 413/3309/365669328 | Original 在边界内终止，L-lite 因候选编译/实测未在 900 秒内原子完成；这是获取开销证据，不产生收益标签。 |
| FlagGems | benchmark/test_conv3d.py::test_conv3d | 266.429 | 900.161 | 633.731 | 278/2255/213546021 | Original 在边界内终止，L-lite 因候选编译/实测未在 900 秒内原子完成；这是获取开销证据，不产生收益标签。 |
| FlagGems | benchmark/test_conv3d.py::test_conv3d_padding | 148.385 | 900.322 | 751.937 | 379/3081/274785611 | Original 在边界内终止，L-lite 因候选编译/实测未在 900 秒内原子完成；这是获取开销证据，不产生收益标签。 |
| FlagGems | benchmark/test_conv_depthwise2d.py::test_conv_depthwise2d | 318.560 | 900.347 | 581.787 | 530/4244/386438277 | Original 在边界内终止，L-lite 因候选编译/实测未在 900 秒内原子完成；这是获取开销证据，不产生收益标签。 |
| FlagGems | benchmark/test_conv_transpose1d.py::test_conv_transpose1d | 200.470 | 900.346 | 699.876 | 294/2358/375195827 | Original 在边界内终止，L-lite 因候选编译/实测未在 900 秒内原子完成；这是获取开销证据，不产生收益标签。 |
| FlagGems | benchmark/test_cp_gather_indexer_k_quant_cache.py::test_cp_gather_indexer_k_quant_cache_benchmark | 22.227 | 900.296 | 878.069 | 574/4595/227578961 | Original 在边界内终止，L-lite 因候选编译/实测未在 900 秒内原子完成；这是获取开销证据，不产生收益标签。 |
| FlagGems | benchmark/test_cudnn_convolution.py::test_cudnn_convolution | 317.995 | 900.302 | 582.307 | 309/2483/301500397 | Original 在边界内终止，L-lite 因候选编译/实测未在 900 秒内原子完成；这是获取开销证据，不产生收益标签。 |
| FlagGems | benchmark/test_deepseek_v4_attention_compute_global_topk_indices_and_lens.py::test_compute_global_topk_indices_and_lens_benchmark | 64.920 | 900.291 | 835.371 | 872/7209/231604651 | Original 在边界内终止，L-lite 因候选编译/实测未在 900 秒内原子完成；这是获取开销证据，不产生收益标签。 |
| FlagGems | benchmark/test_grid_sample.py::test_grid_sample[dtype0] | 184.531 | 900.293 | 715.762 | 733/5871/323183025 | Original 在边界内终止，L-lite 因候选编译/实测未在 900 秒内原子完成；这是获取开销证据，不产生收益标签。 |
| FlagGems | benchmark/test_grid_sample.py::test_grid_sample[dtype1] | 64.356 | 900.308 | 835.952 | 671/5375/310955820 | Original 在边界内终止，L-lite 因候选编译/实测未在 900 秒内原子完成；这是获取开销证据，不产生收益标签。 |
| FlagGems | benchmark/test_grid_sample.py::test_grid_sample[dtype2] | 64.367 | 900.302 | 835.936 | 705/5647/313350056 | Original 在边界内终止，L-lite 因候选编译/实测未在 900 秒内原子完成；这是获取开销证据，不产生收益标签。 |
| FlagGems | benchmark/test_group_norm.py::test_group_norm | 28.512 | 900.181 | 871.669 | 171/1376/154174591 | Original 在边界内终止，L-lite 因候选编译/实测未在 900 秒内原子完成；这是获取开销证据，不产生收益标签。 |
| FlagGems | benchmark/test_im2col.py::test_im2col | 92.534 | 900.358 | 807.824 | 665/5335/314542441 | Original 在边界内终止，L-lite 因候选编译/实测未在 900 秒内原子完成；这是获取开销证据，不产生收益标签。 |
| FlagGems | benchmark/test_instance_norm.py::test_instance_norm | 38.557 | 900.305 | 861.748 | 810/6591/292540185 | Original 在边界内终止，L-lite 因候选编译/实测未在 900 秒内原子完成；这是获取开销证据，不产生收益标签。 |
| FlagGems | benchmark/test_layer_norm.py::test_layer_norm | 153.589 | 900.355 | 746.765 | 553/4589/283718011 | Original 在边界内终止，L-lite 因候选编译/实测未在 900 秒内原子完成；这是获取开销证据，不产生收益标签。 |
| FlagGems | benchmark/test_layer_norm.py::test_layer_norm_backward | 95.052 | 900.360 | 805.308 | 793/6597/309596055 | Original 在边界内终止，L-lite 因候选编译/实测未在 900 秒内原子完成；这是获取开销证据，不产生收益标签。 |
| FlagGems | benchmark/test_median.py::test_median_dim | 71.130 | 900.341 | 829.211 | 148/1194/169030487 | Original 在边界内终止，L-lite 因候选编译/实测未在 900 秒内原子完成；这是获取开销证据，不产生收益标签。 |
| FlagGems | benchmark/test_mhc.py::test_mhc_post | 73.062 | 900.260 | 827.197 | 678/5428/302175293 | Original 在边界内终止，L-lite 因候选编译/实测未在 900 秒内原子完成；这是获取开销证据，不产生收益标签。 |
| FlagGems | benchmark/test_mhc.py::test_hc_split_sinkhorn_forward | 13.699 | 900.325 | 886.625 | 34/303/49960735 | Original 在边界内终止，L-lite 因候选编译/实测未在 900 秒内原子完成；这是获取开销证据，不产生收益标签。 |
| FlagGems | benchmark/test_mm.py::test_mm_self_transpose_benchmark | 831.100 | 900.373 | 69.273 | 882/8784/653156973 | Original 在边界内终止，L-lite 因候选编译/实测未在 900 秒内原子完成；这是获取开销证据，不产生收益标签。 |
| FlagGems | benchmark/test_nanmedian.py::test_nanmedian_dim | 90.872 | 900.344 | 809.472 | 230/1859/189328617 | Original 在边界内终止，L-lite 因候选编译/实测未在 900 秒内原子完成；这是获取开销证据，不产生收益标签。 |
| FlagGems | benchmark/test_nanmedian.py::test_nanmedian_dim_values | 50.507 | 900.281 | 849.774 | 68/552/152194065 | Original 在边界内终止，L-lite 因候选编译/实测未在 900 秒内原子完成；这是获取开销证据，不产生收益标签。 |
| FlagGems | benchmark/test_pixel_shuffle.py::test_pixel_shuffle | 33.906 | 900.276 | 866.370 | 929/7441/240787209 | Original 在边界内终止，L-lite 因候选编译/实测未在 900 秒内原子完成；这是获取开销证据，不产生收益标签。 |
| FlagGems | benchmark/test_pixel_shuffle.py::test_pixel_shuffle_out | 33.410 | 900.348 | 866.938 | 908/7279/229730903 | Original 在边界内终止，L-lite 因候选编译/实测未在 900 秒内原子完成；这是获取开销证据，不产生收益标签。 |
| FlagGems | benchmark/test_rot90.py::test_rot90 | 25.689 | 900.289 | 874.601 | 1702/13626/292214161 | Original 在边界内终止，L-lite 因候选编译/实测未在 900 秒内原子完成；这是获取开销证据，不产生收益标签。 |
| FlagGems | benchmark/test_segment_reduce.py::test_segment_reduce | 88.980 | 900.298 | 811.318 | 991/8499/268465377 | Original 在边界内终止，L-lite 因候选编译/实测未在 900 秒内原子完成；这是获取开销证据，不产生收益标签。 |
| FlagGems | benchmark/test_segment_reduce.py::test_segment_reduce_out | 41.842 | 900.322 | 858.480 | 978/8407/262253836 | Original 在边界内终止，L-lite 因候选编译/实测未在 900 秒内原子完成；这是获取开销证据，不产生收益标签。 |
| FlagGems | benchmark/test_segment_reduce.py::test_segment_reduce_backward | 57.386 | 900.357 | 842.971 | 1082/8772/259654113 | Original 在边界内终止，L-lite 因候选编译/实测未在 900 秒内原子完成；这是获取开销证据，不产生收益标签。 |
| FlagGems | benchmark/test_segment_reduce.py::test_segment_reduce_backward_out | 63.649 | 900.299 | 836.650 | 1077/8730/258358420 | Original 在边界内终止，L-lite 因候选编译/实测未在 900 秒内原子完成；这是获取开销证据，不产生收益标签。 |
| FlagGems | benchmark/test_topk.py::test_topk | 63.592 | 900.327 | 836.734 | 140/1201/134544408 | Original 在边界内终止，L-lite 因候选编译/实测未在 900 秒内原子完成；这是获取开销证据，不产生收益标签。 |
| FlagGems | benchmark/test_tril.py::test_tril_extreme_diagonal | 66.245 | 900.294 | 834.049 | 1121/9909/298570541 | Original 在边界内终止，L-lite 因候选编译/实测未在 900 秒内原子完成；这是获取开销证据，不产生收益标签。 |
| FlagGems-vllm | benchmark/test_FLA/test_chunk_gated_delta_rule_fwd.py::test_perf_chunk_gated_delta_rule_fwd | 349.298 | 900.303 | 551.005 | 397/3275/432309363 | Original 在边界内终止，L-lite 因候选编译/实测未在 900 秒内原子完成；这是获取开销证据，不产生收益标签。 |
| FlagGems-vllm | benchmark/test_instance_norm.py::test_instance_norm | 35.432 | 900.252 | 864.820 | 845/6871/301601652 | Original 在边界内终止，L-lite 因候选编译/实测未在 900 秒内原子完成；这是获取开销证据，不产生收益标签。 |
| FlagGems-vllm | benchmark/test_mhc.py::test_mhc_post | 74.580 | 900.309 | 825.728 | 684/5474/307730240 | Original 在边界内终止，L-lite 因候选编译/实测未在 900 秒内原子完成；这是获取开销证据，不产生收益标签。 |
| FlagGems-vllm | benchmark/test_mhc.py::test_hc_split_sinkhorn_forward | 12.351 | 900.323 | 887.972 | 35/311/54063177 | Original 在边界内终止，L-lite 因候选编译/实测未在 900 秒内原子完成；这是获取开销证据，不产生收益标签。 |
| FlagGems-vllm | benchmark/test_parallel_nsa.py::test_perf_parallel_nsa | 33.011 | 900.293 | 867.282 | 421/3407/336236046 | Original 在边界内终止，L-lite 因候选编译/实测未在 900 秒内原子完成；这是获取开销证据，不产生收益标签。 |
| FlagGems-vllm | benchmark/test_parallel_nsa_compression.py::test_perf_parallel_nsa_compression | 35.868 | 900.349 | 864.481 | 306/2477/313598752 | Original 在边界内终止，L-lite 因候选编译/实测未在 900 秒内原子完成；这是获取开销证据，不产生收益标签。 |

### 3.1 已完成但没有非 Original 获胜的高获取成本项

下表按候选获取耗时降序列出前 10 项。它们已经正确完成，因而不是超时；但 L-lite 付出了候选生成、编译和原生 autotune 实测成本后，所有请求仍由 Original 获胜。这是 autotune 对照版的已实现成本，而不是未来估算。完整无获胜项保留在 V47 逐项报告中。

| 仓库 | pytest 算子项 | 请求 | 候选获取耗时（秒） | 原生 autotune 计时（秒） | 实际提交候选 | 调优后无效候选 | 成对根因 |
|---|---|---|---|---|---|---|---|
| FlagGems | benchmark/test_mhc.py::test_mhc_bwd | 6 | 622.546 | 618.168 | 314 | 255 | LEGAL_L_LITE_CANDIDATE_BUT_ORIGINAL_WON |
| FlagGems-vllm | benchmark/test_mhc.py::test_mhc_bwd | 6 | 601.270 | 596.621 | 314 | 255 | LEGAL_L_LITE_CANDIDATE_BUT_ORIGINAL_WON |
| FlagGems | benchmark/test_special_logsumexp.py::test_special_logsumexp | 9 | 240.116 | 237.335 | 504 | 207 | LEGAL_L_LITE_CANDIDATE_BUT_ORIGINAL_WON |
| FlagGems | benchmark/test_mm.py::test_mm_out | 160 | 200.166 | 70.282 | 1920 | 1760 | NO_EXECUTABLE_NON_ORIGINAL_CANDIDATE |
| FlagGems | benchmark/test_mm.py::test_mm | 160 | 199.466 | 70.467 | 1920 | 1760 | NO_EXECUTABLE_NON_ORIGINAL_CANDIDATE |
| FlagGems | benchmark/test_vdot.py::test_vdot | 35 | 172.014 | 166.251 | 860 | 568 | LEGAL_L_LITE_CANDIDATE_BUT_ORIGINAL_WON |
| FlagGems | benchmark/test_per_token_group_quant_fp8.py::test_per_token_group_quant_fp8 | 5 | 170.624 | 167.653 | 159 | 85 | LEGAL_L_LITE_CANDIDATE_BUT_ORIGINAL_WON |
| FlagGems | benchmark/test_pixel_unshuffle.py::test_pixel_unshuffle | 9 | 127.670 | 124.363 | 240 | 159 | LEGAL_L_LITE_CANDIDATE_BUT_ORIGINAL_WON |
| FlagGems | benchmark/test_linear_backward.py::test_linear_backward | 40 | 122.607 | 26.129 | 480 | 440 | NO_EXECUTABLE_NON_ORIGINAL_CANDIDATE |
| FlagGems | benchmark/test_renorm_.py::test_renorm_ | 126 | 116.569 | 30.328 | 1512 | 1386 | NO_EXECUTABLE_NON_ORIGINAL_CANDIDATE |

## 4. 全量成对根因分布

| 成对根因 | 算子项数 |
|---|---|
| BOTH_ARMS_FAILED_WITH_DIFFERENT_CAUSES | 7 |
| FINAL_JIT_REACHED_WITHOUT_CANDIDATE_REQUEST | 9 |
| LEGAL_L_LITE_CANDIDATE_BUT_ORIGINAL_WON | 11 |
| L_LITE_OPTIMIZED_ROUTE_SELECTED | 88 |
| L_LITE_ORIGINAL_REPLAY_STATE_PROTOCOL_GAP | 1 |
| L_LITE_PROVIDER_SCHEMA_ACTIVE_EXTENT_GAP | 4 |
| L_LITE_SPECIFIC_CANDIDATE_ACQUISITION_TIMEOUT | 45 |
| L_LITE_SPECIFIC_TEST_OR_RUNTIME_FAILURE | 3 |
| NO_EXECUTABLE_NON_ORIGINAL_CANDIDATE | 71 |
| NO_FINAL_TRITON_JIT_IN_ITEM_PATH | 11 |
| PAIRED_ACQUISITION_COMPARISON_INDETERMINATE | 1 |
| PAIRED_DEPENDENCY_UNAVAILABLE | 2 |
| PAIRED_ENVIRONMENT_GPU_MEMORY_EXHAUSTION | 1 |
| PAIRED_FROZEN_VLLM_FLASH_ATTN_BINARY_UNSUPPORTED_ON_CURRENT_GPU | 1 |
| PAIRED_HARDWARE_CAPABILITY_MISMATCH | 1 |
| PAIRED_HARDWARE_SHARED_MEMORY_LIMIT | 1 |
| PAIRED_WORKLOAD_DYNAMIC_FUNCTION_ARGUMENT_GAP | 1 |
| TEST_NOT_APPLICABLE_ON_FROZEN_ENVIRONMENT | 21 |

完整的 279 项逐算子、逐 kernel、逐循环、候选选择和类型化拒绝证据保存在 V47 全量报告中；本摘要不以聚合表替代逐项归因。
