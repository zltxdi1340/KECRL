# LIBERO BC 因果时序对齐诊断（2026-10-22）

**状态：v1 单 seed 诊断已完成。** 结果用于排查数据时序与闭环失败，不是正式实验、架构结论或 Module 资格结果。

## 要回答的问题

LIBERO HDF5 将 `obs[t]` 记录为执行 `action[t]` 后的观测。现有 BC loader 用 `obs[t] -> action[t]` 训练，与在线控制所需的 pre-action state/action 配对错位。本轮只检验这个数据时序错误是否足以解释“离线 MSE 下降、闭环执行不稳定”。

因果视图从每条授权 demonstration 的 `init_state` 重建首个 pre-action observation，再用 `[initial_observation, stored_obs[:-1]] -> actions[:]`。它与历史同索引视图保持序列长度、动作标签和窗口数一致；原始 HDF5 不修改，默认 loader 行为也不变。

## 固定对照

| 项目 | 固定设置 |
|---|---|
| 任务 | LIBERO-Spatial task 0，`PickUp(Bowl) -> Place(Plate)` |
| 训练 seed | `20261021` |
| 训练数据 | source `meta_support` demonstrations 0--29 |
| 对照臂 | `stored_same_index` 与 `causal_pre_action` |
| 每臂预算 | Adam 500 updates，batch 8，`lr=3e-4`，gradient clip 1.0，序列长 10、stride 10 |
| 配对约束 | 同一 Policy 初始化哈希、同一 shuffle generator seed/批次顺序；检查两臂有效动作步数相同 |
| 离线评估 | 两个训练后 Policy 均在 support 与 query demonstrations 上，用两种 observation/action 视图计算指标 |
| 闭环评估 | 连续 GRU；support 初态 0/1/2、query 初态 30/31/32、官方 candidate 初态 0/1/2；最多 300 步 |
| 稳定步 | demonstration 初态 0 步，便于与专家轨迹一致；官方初态 5 个零动作稳定步 |
| 前置门槛 | support/query 各 3 条专家动作重放必须全部成功，否则不训练 |

validation、held-out、qualification 初态均不访问；不选 checkpoint、不更新 SPT、不创建 Module、不更新 Knowledge，也不用于完整架构比较。该协议只允许诊断单 task、单 seed 下 observation/action 时序的影响。

## 文件与 provenance

- 代码 commit：`96d37f7bd8d694cd6792786d1267924cde9ddc83`，GitHub `main` 已推送。
- 配置：[因果对齐诊断配置](../configs/libero_spatial_bc_causal_alignment_diagnostic_v1.json)，SHA-256 `95c800594719063ed2a5b02e0e206584817b527b263fecbe3f50cc13b2a1af48`。
- 入口：[run_libero_bc_causal_alignment_diagnostic.py](../experiments/run_libero_bc_causal_alignment_diagnostic.py)。
- 基础单 SPI 配置 SHA-256：`bbc4b28c4d2d606f2595642ee0f2936bb3fbcded3e5cbd9449e78b337f134e53`。
- 回归：LIBERO 定向 `14 passed`；全量 `526 passed, 2 skipped`。
- 环境：`libero-kecrl`、Python 3.10.22、PyTorch 2.4.1+cu124、2 张 RTX 4090 可见。
- 结果文件：`/tmp/libero-bc-causal-alignment-20261022-v1.json`，SHA-256 `5fbf8e1ed56e8c5f052f75c9a3303aff80eb3fc65337501ca0b5c459c66a34ed`。
- 运行记录的代码 commit：`2e735b85f8e63bc3c559c9e44fe23b1647dcf461`；两臂初始 Policy 哈希相同，均为 `80c75b88ac2721d62949dfbe2c8bd5adb3be74303aa66ab4007ab54f2bfb78b4`。
- 每臂处理 `37,534` 个有效动作步；结果记录 valid-step 数相同，SPT 未改变。

## v1 结果

专家轨迹从 30 条 support 和 20 条 query demo 中各抽 3 条重放，结果均为 `3/3`。数据和官方初始状态的 pinned SHA-256 校验通过；validation、held-out 和 qualification 均未访问。

| 训练数据视图 | 评估数据视图 | support MSE | query MSE |
|---|---|---:|---:|
| stored same-index | stored same-index | 0.07957 | 0.08830 |
| stored same-index | causal pre-action | 0.08314 | 0.09318 |
| causal pre-action | stored same-index | 0.08840 | 0.09578 |
| causal pre-action | causal pre-action | 0.08769 | 0.09682 |

这次单 seed、固定 500-update 对照中，因果视图训练没有降低因果 query MSE；same-index 模型在因果 query view 的 MSE `0.09318`，略低于因果模型的 `0.09682`。这不是“same-index 更好”的统计证据，只表示修正配对时序单独没有修复这次训练的闭环失败。

| 训练数据视图 | support demo rollout | query demo rollout | official candidate rollout |
|---|---:|---:|---:|
| stored same-index | 1/3 | 0/3 | 0/3 |
| causal pre-action | 0/3 | 0/3 | 0/3 |

所有未成功 Policy episode 都运行到 300 步上限。因果臂的 9 个 rollout 中，夹爪动作全程为负且没有符号切换；其 causal query gripper sign accuracy 仍为 `0.886`。同一 task 0 的 support/query 专家标签中，夹爪正负命令约为 `48%/52%`，不是单一负类占绝大多数。该反差说明离线标签拟合不能保证 Policy 离开 expert-state 分布后仍产生正确的开合时序。

运行耗时 `187.85s`，训练与 Policy 评估部分 `125.68s`，峰值 CUDA allocation `292,003,840 bytes`。单 seed 结果不用于否定因果 loader，也不说明具体是视觉、动作头、GRU 还是状态覆盖导致失败。

## 下一步诊断

隐藏态诊断、冻结 causal Policy 的六初态逐步动作/状态偏差追踪以及局部单步反事实追踪均已完成，结果见 [`libero-bc-hidden-state-diagnostic-2026-10-22.md`](libero-bc-hidden-state-diagnostic-2026-10-22.md)。单步反事实确认两次专家前缀重放完全一致，并测到动作差异产生的即时 joint/gripper/RGB 响应；其量级小于自由 rollout 的长程漂移，因此下一步检查离开专家状态后 Policy 输出是否额外偏移。当前不进行 SPT/FOMAML、Knowledge 更新或 Module qualification。

结果可由已提交配置和 runner 重现；输出文件已存在时请使用新的 `--output` 路径：

```bash
MUJOCO_GL=egl LIBERO_CONFIG_PATH=/home/zl202621/.libero PYTHONPATH=. \
  /opt/anaconda3/bin/conda run --no-capture-output -n libero-kecrl \
  python -u experiments/run_libero_bc_causal_alignment_diagnostic.py \
  --dataset-root /home/zl202621/third_party/LIBERO/datasets \
  --device cuda:1 \
  --output /tmp/libero-bc-causal-alignment-20261022-rerun.json
```
