# LIBERO 当前运行时专家轨迹审计（2026-10-25）

## 目的和边界

回放语义冻结审计确认 controller 配置、动作边界和 control frequency 没有解释 HDF5 首步偏差。本次固定历史 HDF5 的初态与动作序列，在当前 pinned LIBERO/robosuite/MuJoCo runtime 中重新生成 observation/state 轨迹摘要，检查当前 runtime 是否自身确定。

每条 case 新建 adapter 3 次，`seed=20261021`、`stabilization=0`。完整轨迹的 digest 同时包含实际执行动作、sim state、agentview/eye-in-hand RGB、joint 和 gripper observation。官方 HDF5 只读，不写回 regenerated observation，也不用于 checkpoint、Policy、Knowledge 或 Module 选择。

- 配置：[current runtime replay audit](../configs/libero_current_runtime_replay_audit_v1.json)，SHA-256 `5f2dbb50a10d5fa5b702f9e0860ecdb7016fb8b4ffe0d4b1c5f72d1d7ae83622`。
- 结果：`/tmp/libero-current-runtime-replay-20261025-v2.json`，SHA-256 `13fbb4cec452ac3bf7d88674e24168e757d6ba5c92983619ecbc64c9a86c6f45`。
- 运行提交：`024d75b8a7f9fe6b47019f4ae8fbbc00a51fe5c7`；CUDA `cuda:1` / RTX 4090；耗时约 `38.87s`。
- 当前 runtime：Python `3.10.22`、NumPy `1.22.4`、robosuite `1.4.0`、MuJoCo `3.2.3`。

## 结果

| case | HDF5 reward/done | 当前 runtime 三次成功 | 三次 steps | trace digest | sim-state digest | observation digest |
|---|---|---:|---|---:|---:|---:|
| task 3 demo 1 | `1` / `[100]` | `0/3` | `101,101,101` | 1 个 | 1 个 | 1 个 |
| task 3 demo 0 | `1` / `[98]` | `3/3` | `88,88,88` | 1 个 | 1 个 | 1 个 |
| task 7 demo 0 | `1` / `[154]` | `0/3` | `155,155,155` | 1 个 | 1 个 | 1 个 |
| task 7 demo 1 | `1` / `[124]` | `3/3` | `112,112,112` | 1 个 | 1 个 | 1 个 |

四条 case 的初态都与 HDF5 `init_state` 精确一致。当前 runtime 的失败/成功边界在重复创建 adapter、完整 sim-state 和完整 observation digest 上都稳定，没有出现重复之间的随机轨迹分歧。

## 与 reset 路径的补充检查

为排除 adapter 的 reset 顺序，task 3 demo 1 的第一条动作又在以下路径下执行：

1. adapter 的 `reset()` + `set_init_state()`；
2. `reset()` 后直接 `sim.set_state_from_flattened()`；
3. 先 `reset_from_xml_string()` 再 `sim.set_state_from_flattened()`；
4. `reset()`、保存的 episode XML、再 `sim.set_state_from_flattened()`；
5. 直接 sim set 后显式 `_post_process/_update_observables`。

五条路径的首步完整 state L∞ 都约为 `0.098517`，qvel L∞ 都约为 `0.098517`，qpos L∞ 都约为 `0.005424`。因此当前观测到的首步偏差不能由 adapter 的 reset 顺序单独解释；保存 XML 也没有消除偏差。这是同一环境 checkout 中的补充取证，不是新的训练协议。

## 诊断结论

1. 当前 runtime 对历史动作序列的轨迹是确定的，前面 task 3/7 的 replay 结果可以稳定复现。
2. HDF5 成功标签与当前 runtime 成功判定仍不一致，但这种不一致不能再归因于重复噪声、reset 顺序、controller 配置或 adapter 动作裁剪。
3. 历史 HDF5 observation/state 与当前 runtime 重新生成的 observation/state 需要被视作两种数据语义。当前 runtime 轨迹可用于诊断当前闭环，但不能直接覆盖官方 HDF5。
4. 在 HDF5 生成时 runtime provenance 缺失的情况下，继续调 Policy loss 不能解决这条数据—环境边界。后续 source-support 全量 replay 显示 150 条 meta-support 示教只有 17 条能完整执行到 HDF5 action 序列末端，因此当前 runtime 不能直接替代完整离线训练视图；本仓库冻结为“离线历史观测 + 当前 runtime 闭环”的双协议。

## 下一步

冻结当前 runtime 的 regenerated trajectory 只作为诊断缓存。历史 HDF5 与当前 runtime 输入对齐审计、固定 Policy 输入替换敏感性审计和 source-support 全量 replay 可行性审计均已完成，结果见[输入对齐审计](libero-replay-input-alignment-audit-2026-10-25.md)、[输入替换敏感性审计](libero-policy-input-replacement-sensitivity-audit-2026-10-25.md)和[source-support replay 可行性审计](libero-source-support-replay-feasibility-audit-2026-10-25.md)。当前 runtime 只能作为闭环/对齐诊断，HDF5 causal view 作为完整离线 BC 协议。

phase-balanced、gripper-loss 和 A/B/C 正式候选实验继续暂停。
