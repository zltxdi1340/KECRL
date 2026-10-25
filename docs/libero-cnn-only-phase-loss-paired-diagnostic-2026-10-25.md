# LIBERO CNN-only 阶段损失配对诊断（2026-10-25）

## 目的与边界

此前 CNN-only + phase-balanced 在 5 个 source task、2 个 seed 上获得整体 action MAE `0.12295`，但缺少同结构 uniform-loss 对照，不能把该误差归因于 phase weighting。本次在完全相同的 task/seed/data/init/batch order 下配对比较 uniform action MSE 与 phase-balanced loss，隔离阶段权重对离线动作误差的影响。

- Policy：容量匹配 CNN-only，逐帧 MLP 参数量与单层 GRU 相差 1 个参数；两臂均采用相同 Policy 架构。
- 数据：5 个 source-SPI task `0/1/3/4/7`；`meta_support` 训练、`meta_query` 离线评估；HDF5 causal pre-action 视图。
- 预算：seed `20261021/20261022`，每个 task-seed 500 次 Adam update，共 20 个训练 arm。
- 控制：每个 task-seed 内两臂初始 Policy SHA 完全匹配；batch 顺序和训练随机数重置一致。唯一实验变量是 loss 是否按 phase 逆频率加权。两臂都对七个动作维度等权；配置中的 `gripper_action_weight=6.0` 不会被这两种 loss mode 使用，只在单独的 gripper-weighted arm 生效。
- 边界：没有执行 Policy 闭环动作，没有访问 validation/held-out/qualification，没有选择 checkpoint，也没有更新 SPT/Knowledge 或注册 Module。

配置：[paired phase-loss diagnostic](../configs/libero_cnn_only_phase_loss_paired_diagnostic_v1.json)，SHA-256 `0b4af76d6e6dc0d80b7493880b097e7a8700882aa7f61b89c904ba48d9bbb63d`。

结果：`/tmp/libero-cnn-phase-loss-paired-20261025-v1.json`，SHA-256 `249fe27c13909dce98a17c5edbc6aa68c5ac3cb7cf527ac86fa1c0435e668c03`。运行时记录提交 `1830179c0a27a780235faaf8e6f5214430e2c4fd`；CUDA `cuda:1` / RTX 4090。

## 总体结果

两臂均在相同的 25,238 个 query step 上计算：

| loss | action MAE | translation MAE | rotation MAE | gripper MAE | gripper sign accuracy |
|---|---:|---:|---:|---:|---:|
| Uniform | 0.11708 | 0.17228 | 0.03484 | 0.19823 | 0.9278 |
| Phase-balanced | 0.12295 | 0.18016 | 0.03516 | 0.21472 | 0.9228 |

Phase-balanced 相对 uniform 的总体 action MAE 增加 `0.00587`，translation MAE 增加 `0.00788`，gripper MAE 增加 `0.01649`。两个 seed 都呈相同方向：

| seed | Uniform action MAE | Phase-balanced action MAE | 差值 |
|---:|---:|---:|---:|
| 20261021 | 0.11697 | 0.12329 | +0.00632 |
| 20261022 | 0.11720 | 0.12261 | +0.00541 |

## 阶段分层

| phase | Query steps | Uniform action MAE | Phase-balanced action MAE | Uniform gripper MAE | Phase-balanced gripper MAE | Uniform / balanced sign accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Approach | 8,754 | 0.08386 | 0.08398 | 0.07511 | 0.06664 | 0.9845 / 0.9880 |
| Transport | 13,118 | 0.12186 | 0.13935 | 0.18762 | 0.26065 | 0.9379 / 0.8989 |
| Release | 3,366 | 0.18487 | 0.16041 | 0.55979 | 0.42079 | 0.7409 / 0.8461 |

阶段均衡确实改善 release：action MAE 减少 `0.02447`，gripper MAE 减少 `0.13900`，gripper sign accuracy 提升约 `10.5` 个百分点；同时 transport action MAE 增加 `0.01749`，gripper MAE 增加 `0.07304`，sign accuracy 下降约 `3.9` 个百分点。总体损失说明这种 reweighting 把拟合能力从高频 transport 转向 release，但在当前任务混合中，release 改善不足以抵消 transport 恶化。

## 任务分布与结论

任务级 action MAE 中，phase-balanced 在 8/10 个 task-seed pair 上更差。按两 seed 聚合后，五个 task 的 action MAE 也都高于 uniform；task 0 仍是最高误差条件（uniform `0.14985`，phase-balanced `0.15848`）。

1. **不保留 phase-balanced 作为通用默认 loss。** 其 release 误差改善是真实、可重复的，但总体和 transport 指标变差，两个 seed 方向一致。
2. **release/gripper 是局部误差问题，不等于应提高整段 release 的总训练权重。** 当前 phase 标记仅由 gripper action 符号定义，phase-balanced 会连带改变该阶段全部 7 个动作维度的梯度权重。
3. **后续基线采用 CNN-only + phase-uniform、七维等权的 action MSE。** 若研究 release，需要另设单变量、受限到 gripper 维度或明确边界动作的 loss 干预，不能继续使用全动作 phase 权重并把效果叫作 memory/SPT 收益。
4. **这仍是离线诊断。** 结果不能推出当前 runtime 闭环成功率、跨任务适应效率、SPT 更新效应或 Module qualification。

## 下一步

固定 CNN-only + phase-uniform、七维等权 action MSE 为离线基线，先测试该 Policy 在当前 LIBERO runtime 的 candidate-nonregression 初始状态上是否能完成闭环动作链，并按 task 报告成功率和终止步数。该闭环结果与 HDF5 query action MAE 分开呈现；若仍然失败，再基于任务 0 的高误差与 transport/release 分层确定下一个单变量诊断。只有基础策略出现可重复的闭环成功，才进入 SPT 更新前的 A/B/C 迁移准备。
