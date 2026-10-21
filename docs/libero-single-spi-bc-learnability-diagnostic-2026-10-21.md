# LIBERO 单 Source SPI 行为克隆可学习性诊断（2026-10-21）

本诊断原本检查候选 RGB+proprioception+GRU Policy 在进入多任务 SPT 更新前，能否拟合一个 source SPI 的独立示教配对，以及这种离线改善是否转化为闭环任务执行。观察结果是：**保存配对上的离线误差明显下降，官方初态闭环仍未成功**；后续时序勘误进一步限制了该离线指标的含义。

本次不是学习 pilot、架构对照、SPT 更新、Module qualification 或正式结果。

> **后续勘误（2026-10-21）：** 闭环归因发现 LIBERO HDF5 的 `obs[t]` 是执行 `action[t]` 后记录的 observation，当前 loader 却把二者按同索引监督。因此本页的 MSE 下降只能说明模型拟合了独立轨迹上的“post-action observation 到同一 action”映射，不能解释为已经学会因果 expert-state 控制映射。专家动作重放 6/6 成功，而 Policy 在 demo 初态和官方初态均不稳定。完整证据见 [`libero-bc-closed-loop-attribution-2026-10-21.md`](libero-bc-closed-loop-attribution-2026-10-21.md)。

## 固定边界与 provenance

- 运行代码 commit：`57c447a`（`增加LIBERO单SPI固定预算学习诊断`）。
- 固定配置：[`configs/libero_spatial_single_spi_bc_diagnostic_v1.json`](../configs/libero_spatial_single_spi_bc_diagnostic_v1.json)。
- 入口：[`experiments/run_libero_single_spi_bc_diagnostic.py`](../experiments/run_libero_single_spi_bc_diagnostic.py)。
- task 0：`pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate`。
- 训练：source `meta_support` demonstration 0--29，共 2,925 个有效时间步、306 个窗口。
- 离线 query：同一 source SPI 的 `meta_query` demonstration 30--49，共 2,143 个有效时间步、223 个窗口。
- 闭环：source `candidate_nonregression` 初态 0、1、2；每个 episode 最多 300 步。
- 单 seed `20261021`，`cuda:1`，500 个 Adam 更新，batch 8，序列长度 10，`lr=3e-4`，梯度裁剪 1.0。
- 500 更新共处理 37,534 个有效 action-step；checkpoint 只记录，不做选择，固定运行到 500。
- validation、held-out、source qualification 均未访问；SPT 保持不变。
- 提交后结果文件：`/tmp/libero-single-spi-bc-diagnostic-20261021-v2.json`，SHA-256 `e061eaae69db725be0b94e234474db7e00559929e1ce2094e5bd9983666e8c48`。

同一代码提交前做过一次预运行。除共享 GPU 导致的 wall time 外，两次输出逐项一致；以下数字来自提交后的 v2。

## 离线动作结果

| update | support MSE | query MSE | query translation MSE | query gripper MSE | query gripper sign accuracy |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.248368 | 0.248540 | 0.239320 | 1.000131 | 0.5100 |
| 50 | 0.219477 | 0.222314 | 0.208968 | 0.917098 | 0.5198 |
| 150 | 0.176679 | 0.182320 | 0.190225 | 0.693913 | 0.7830 |
| 300 | 0.104540 | 0.111637 | 0.141781 | 0.345321 | 0.8903 |
| 500 | 0.079568 | 0.088300 | 0.112195 | 0.270151 | 0.9127 |

query MSE 下降 64.5%，最终 support/query 差值约 0.0087。该曲线说明模型不只是记住训练窗口：在未用于更新的同任务 expert trajectories 上，平移和夹爪预测也同步改善。rotation MSE 从一开始就很小，最终 query rotation MSE 为 `0.003789`；总体 MSE 因而不能替代各动作分量报告。

首 batch loss 为 `0.243631`，末 batch loss 为 `0.075395`，500 次更新的平均 batch loss 为 `0.141713`。运行耗时 50.22 秒，峰值 CUDA allocation 为 285,952,000 bytes。该耗时包含多次完整离线评估和 6 个闭环 episode，不是纯训练吞吐。

## 闭环结果

| update | successes | 初态 0 | 初态 1 | 初态 2 |
|---:|---:|---|---|---|
| 0 | 0/3 | 300 步超时 | 300 步超时 | 300 步超时 |
| 500 | 0/3 | 300 步超时 | 300 步超时 | 300 步超时 |

所有 episode reward 都为 0，未触发环境成功谓词。500 次更新不能让这个 SPI 通过最小闭环可行性检查，因此不能形成 Module，也不能作为后续 SPT 迁移的可靠反馈来源。

## 已得到的诊断结论

1. **训练链路和基础表示不是完全失效，但原指标不是因果控制证据。** 两路 RGB、proprioception、GRU 和连续动作头能拟合独立轨迹上的已保存配对，说明优化器在工作；后续时序审计表明 observation 位于同索引 action 之后，因此不能再把该结果称为因果 expert-state 动作学习。
2. **离线 MSE 仍不能代表闭环控制。** 训练和 query 都来自专家访问过的状态；Policy 自己执行后会进入不同状态，误差会累积。当前 0/3 与下降的 query loss 同时出现，正是需要单独检查 covariate shift 和动作执行动态的证据。
3. **不能立刻归因为预算不足。** 曲线到 500 更新仍在改善，更大预算可能继续降低误差；但在不知道闭环动作幅值、移动范围和夹爪时序前直接扩大预算，无法区分“尚未拟合够”与“监督目标/时序接口不适合闭环”。
4. **当前只涉及一个 source SPI。** 结果不能推广到其余 source、validation 或 held-out，也没有说明 structured context 或 SPT generator 有效。

## 下一步

以下闭环失败归因已经完成：专家重放 6/6，三种隐藏态模式在 demo 初态仅有两个单例成功，官方初态全部失败，并发现 HDF5 同索引 observation/action 的因果错位。下一步保持同 task、seed、模型和 500-update 预算：

1. 从每条 `init_state` 重建首个 pre-action observation；
2. 将其与后续 stored observation 组成等长的 causal `pre_action_obs[t] -> action[t]` 数据视图；
3. 与现有 same-index 数据视图做固定预算诊断，先检验 causal query loss 和 demo-start 闭环，再检查官方初态；
4. 因果对齐后若仍失败，再分别检查 MSE 动作平均化、长度 10 窗口/隐藏状态训练和 covariate shift。

当前不进入 FOMAML 或 A/B/C，因为基础 SPI 尚不能闭环执行；此时比较 SPT 更新只会把 Policy 执行失败混入迁移结论。
