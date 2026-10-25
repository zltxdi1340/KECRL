# LIBERO CNN-only 阶段均衡跨任务复核（2026-10-25）

## 目的与边界

前一轮 task 3/7、单 seed 的容量匹配表示诊断中，CNN-only + phase-balanced 的离线 query MAE 低于 CNN+GRU。为判断 CNN-only 训练路径在 source 任务间是否稳定，本次固定 CNN-only、HDF5 causal observation、phase-balanced loss 和 500-update 预算，扩展到全部 5 个 source-SPI 任务及两个 seed。

本次共 10 个 task-seed 训练单元。每个 run 使用对应任务的 30 条 `meta_support` 训练、20 条 `meta_query` 离线评估；不执行 Policy 闭环动作，不访问 validation/held-out/qualification，不选择 checkpoint，也不更新 SPT/Knowledge 或注册 Module。

- 配置：[CNN-only phase replication diagnostic](../configs/libero_cnn_only_phase_replication_diagnostic_v1.json)，SHA-256 `b8a1a16e886fb9e09fbfc4c7a9fc6263569e4c61464a1aeee91932a2545ab959`。
- 结果：`/tmp/libero-cnn-only-phase-replication-20261025-v1.json`，SHA-256 `0be80d87bb4a9410da8bde635666d631c11a277e2002d061a3dbf9710c693b14`。
- 运行时记录提交：`5e4b5fca90491c670da36adf56c636bd52ff7cca`；CUDA `cuda:1` / RTX 4090；每个任务 500 updates。

## 汇总结果

| 聚合方式 | Query 步数 | action MAE | translation MAE | gripper MAE | gripper sign accuracy |
|---|---:|---:|---:|---:|---:|
| 全部任务、seed 加权 | 25,238 | 0.12295 | 0.18016 | 0.21472 | 0.9228 |
| seed 20261021 | 12,619 | 0.12329 | 0.18243 | 0.20950 | 0.9255 |
| seed 20261022 | 12,619 | 0.12261 | 0.17789 | 0.21993 | 0.9200 |

按 task 汇总两个 seed：

| task | Query 步数 | action MAE | translation MAE | gripper MAE | gripper sign accuracy |
|---:|---:|---:|---:|---:|---:|
| 0 | 4,286 | 0.15848 | 0.22317 | 0.31093 | 0.8773 |
| 1 | 5,236 | 0.12464 | 0.19199 | 0.19074 | 0.9360 |
| 3 | 3,916 | 0.11133 | 0.15312 | 0.23101 | 0.9211 |
| 4 | 6,014 | 0.11582 | 0.17628 | 0.17963 | 0.9343 |
| 7 | 5,786 | 0.11038 | 0.15993 | 0.19058 | 0.9336 |

两个 seed 的汇总 action MAE 相差 `0.00068`，表明本配置总体离线误差对这两个训练 seed 较稳定；task 间差异远大于 seed 汇总差异，task 0 是明显的高误差条件。

## 诊断结论

1. **CNN-only + phase-balanced 能在五个 source task 上完成固定预算训练。** 本轮 10 个 run 均完成，query 步数和评估协议一致，没有 validation/held-out 数据泄漏。
2. **seed 汇总稳定不代表每个 task 都稳定或已学会 SPI。** task 0 的 action MAE `0.15848`，明显高于 task 3/4/7；任务关系和几何条件仍造成较大差异。
3. **本轮没有检验 phase-balanced 是否优于普通 uniform loss。** 它只有一个损失臂；前一轮两 task/一 seed 的 uniform 比较不能覆盖全部 source 任务与两个 seed。因此不能把 `0.12295` 解释为阶段均衡带来的收益。
4. **结果是离线动作拟合诊断，不是闭环 SPI 能力。** 不能据此声称任务完成、跨任务迁移成功、SPT 有效或 Module 合格。

## 下一步

运行同结构的 CNN-only + uniform action MSE 对照，覆盖完全相同的 5 task × 2 seed、相同初始化、batch 顺序和 500 updates。与本结果配对比较 overall/phase action MAE，判断 phase-balanced 是否改善少样本 release 阶段而不损害其他阶段。之后才决定是否在 SPT/SPI A/B/C pilot 中保留该损失定义。
