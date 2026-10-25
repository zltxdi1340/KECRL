# LIBERO CNN-only 与 CNN+GRU 表示诊断（2026-10-25）

## 目的与边界

阶段/夹爪损失诊断显示，阶段逆频率加权有小幅离线收益，固定 6 倍夹爪权重没有收益。本次在相同 HDF5 causal 离线协议和相同 phase-balanced loss 下，比较是否需要跨时间 hidden：

- `cnn_only`：共享双视角 CNN 和 fusion 后使用逐帧 MLP，不携带 recurrent hidden；
- `cnn_gru`：保留现有连续 GRU hidden。

为了避免把容量差异误称为记忆收益，CNN-only 的逐帧 MLP 与单层 GRU 的参数量匹配在 10 个参数以内。两臂使用 task 3、task 7、seed `20261021`、相同 support/query 数据、相同 batch 顺序和 500 次 Adam update；两臂的 `visual_encoder+fusion` 初始参数摘要相同。没有执行 Policy 闭环动作，没有访问 validation/held-out/qualification，没有选择 checkpoint，也没有更新 SPT/Knowledge 或注册 Module。

- 配置：[memory representation diagnostic](../configs/libero_memory_representation_diagnostic_v1.json)，SHA-256 `a61f703ceda4f446a9d440867108a38534ed6cf05f3f41eccd1806548b4b469e`。
- 结果：`/tmp/libero-memory-representation-20261025-v2.json`，SHA-256 `edff7d0271abc633f22a5cea38d77e199450ecb022f21e76f8091ee766089cfc`。
- 运行时记录提交：`5a17dfd7c43e730a286dd7b55e6252e603fa7e19`；CUDA `cuda:1` / RTX 4090。
- CNN-only 参数量 `821704`，CNN+GRU 参数量 `821703`；两臂共享 CNN/fusion 初始摘要均为 `75f9604ecd8aa5d07eb781dec3fe67d42484f4eec5158b0482d2d2f5301a3f2e`。

## Query 结果

| task | arm | action MAE | translation MAE | gripper MAE | gripper sign accuracy |
|---:|---|---:|---:|---:|---:|
| 3 | CNN-only | 0.10806 | 0.15216 | 0.21286 | 0.9280 |
| 3 | CNN+GRU | 0.11256 | 0.15979 | 0.20518 | 0.9132 |
| 7 | CNN-only | 0.10640 | 0.15436 | 0.17595 | 0.9319 |
| 7 | CNN+GRU | 0.11371 | 0.17323 | 0.16600 | 0.9405 |

两任务 query 步数加权（4,851 步）：

- CNN-only：action MAE `0.10707`，translation MAE `0.15347`，rotation MAE `0.03274`，gripper MAE `0.19085`，gripper sign accuracy `0.9303`；
- CNN+GRU：action MAE `0.11324`，translation MAE `0.16781`，rotation MAE `0.03582`，gripper MAE `0.18181`，gripper sign accuracy `0.9295`。

phase 加权结果：

| phase | CNN-only action MAE | CNN+GRU action MAE | CNN-only gripper MAE | CNN+GRU gripper MAE |
|---|---:|---:|---:|---:|
| approach | 0.08124 | 0.08618 | 0.06145 | 0.06696 |
| transport | 0.11229 | 0.11840 | 0.22364 | 0.19762 |
| release | 0.14250 | 0.15160 | 0.34279 | 0.36797 |

CNN-only 在两个任务的整体 action/translation MAE 都低于 CNN+GRU；CNN+GRU 只在 transport gripper MAE 上较低。release 阶段是两臂共同的高误差区域，GRU 没有稳定改善该阶段。

## 诊断结论

1. **当前 targeted 数据不支持“GRU 记忆稳定改善动作拟合”。** 在容量匹配、同一 phase-balanced loss、同一数据顺序下，CNN-only 的加权 action MAE `0.10707` 低于 CNN+GRU 的 `0.11324`。
2. **记忆收益不是全局缺口的主要解释。** CNN+GRU 在 task 7 transport gripper 上有局部收益，但在整体 action、translation 和 release 上没有形成稳定优势；这更像条件/阶段特定差异，而不是已验证的记忆效应。
3. **目前应保留 CNN-only 作为更简单的离线 baseline。** 这不是说 GRU 在所有任务都无用；只是在 task 3/7、一个 seed、500 update 的机制诊断中，没有理由优先增加 recurrent 复杂度。
4. **结果仍不能推出闭环成功率、SPT 更新或 Module 资格。** 本轮只测 HDF5 causal query 的离线 action fit；当前 runtime 观测协议仍受 17/150 support 示教完整前缀覆盖限制，只用于闭环/对齐诊断。

## 下一步

先固定 CNN-only 作为基础 Policy，扩大到全部 5 个 source-SPI 任务和至少两个 replicate seed，保持 phase-balanced loss、HDF5 causal 协议和同一评估指标；只在离线结果稳定后，读取当前 runtime 闭环作为独立诊断。随后再把 CNN-only Policy 接入 A/B/C 的 SPT/SPI 迁移小规模 pilot，仍需先冻结适应预算、资格门槛和 Module 合同，不能把本轮离线误差直接当作迁移收益。
