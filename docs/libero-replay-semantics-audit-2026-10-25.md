# LIBERO 回放语义冻结审计（2026-10-25）

## 目的和范围

状态对齐审计已经确认四条示教都在首个动作后偏离 HDF5。本次只检查首动作边界是否被 controller 配置、control frequency、仿真时间步、版本差异或 adapter 的动作检查改变。它是环境兼容性诊断，不训练 Policy，也不访问 validation、held-out 或 qualification 状态。

- 配置：[replay semantics audit](../configs/libero_replay_semantics_audit_v1.json)，SHA-256 `85c84bc93ad889ae0219f95e9353aae9342f59cbc2138ec09942d8f5b1bd8e3c`。
- 结果：`/tmp/libero-replay-semantics-20261025-v2.json`，SHA-256 `561ac3c655bfbf5d73d8fbafff306bf9686606bab859251a602fdf3544580886`。
- 运行提交：`30200c2a13c6953752427c09661c0fe6a6680c5e`；CUDA `cuda:1` / RTX 4090；耗时约 `10.84s`。
- 四条 case 与状态对齐审计相同：task 3 demo 1/0、task 7 demo 0/1；固定 seed `20261021`，`stabilization=0`。

## 检查结果

### Controller、频率和动作接口

四条示教的 HDF5 `env_args.env_kwargs.controller_configs` 与当前 adapter 的相关字段逐项相同：`OSC_POSE`、输入/输出范围、`kp=150`、阻尼比、`control_delta`、`uncouple_pos_ori`、插值和 `ramp_ratio` 均没有差异。四条 case 的 `controller_config_differences` 都为空。

当前环境与数据配置都使用 control frequency `20`；当前模型时间步为 `0.002`，state 维度为 `nq=48/nv=43`。四条首动作都有限且在 adapter 的 `[-1,1]` 边界内，动作原样通过 `_validate_action`，没有发现越界或额外裁剪。

| case | 首步 state L∞ | 首步 qpos L∞ | 首步 qvel L∞ | 首步 joint L∞ | 首步 gripper L∞ | 首步 eye-in-hand MAE |
|---|---:|---:|---:|---:|---:|---:|
| task 3 demo 1 | .0985 | .00542 | .0985 | .00267 | .00529 | 6.75 |
| task 3 demo 0 | .1228 | .00542 | .1228 | .00438 | .00529 | 7.53 |
| task 7 demo 0 | .1328 | .00540 | .1328 | .00381 | .00527 | 7.31 |
| task 7 demo 1 | .1328 | .00540 | .1328 | .00302 | .00527 | 7.61 |

四条轨迹的初态 `init_state` 与 HDF5 `states[0]` 仍精确一致，但首步后都不满足 `1e-5` 状态对齐阈值。qvel 偏差是完整 state 偏差的主要部分；qpos、joint、gripper 和 RGB 也同时有非零差异。这个模式在 HDF5 标签为成功的控制示教中同样出现。

### 当前运行时信息

语义审计记录的当前运行时为：Python `3.10.22`、NumPy `1.22.4`、PyTorch `2.4.1+cu124`、robosuite `1.4.0`、MuJoCo `3.2.3`，LIBERO checkout `8f1084e3132a39270c3a13ebe37270a43ece2a01`。HDF5 数据和 `model_file` 属性没有记录生成时的仿真器版本；当前专用环境也没有安装 `mujoco_py`。因此可以确认“当前运行时无法逐步复现历史 HDF5”，但不能仅凭本审计断言差异来自哪一个版本或物理参数。

## 结论

1. 已检查的 controller 配置、control frequency、动作范围和 adapter 动作校验不是首步差异的来源。
2. HDF5 与当前环境的差异发生在动力学响应和观测生成层，qvel 偏差最大，并伴随可见的 qpos/gripper/RGB 偏差。
3. 当前 replay 的成功/失败仍是当前环境中的真实行为，但不能把它当成历史 HDF5 标签的直接复现，也不能把某一条 replay 失败归因给 Policy。
4. 这一步没有解决历史数据的生成版本问题；它把问题收窄到“缺少生成时 runtime provenance 或存在仿真/控制器内部状态差异”。

## 下一步

当前 runtime 的可复现专家轨迹审计已经完成，见 [LIBERO 当前运行时专家轨迹审计](libero-current-runtime-replay-audit-2026-10-25.md)。四条 case 的完整 sim-state/observation digest 在三次重复中都稳定，但成功边界仍与 HDF5 标签不同。下一项比较历史 HDF5 causal observation 与当前 regenerated pre-action observation 对 Policy 输入的影响；原始 HDF5 继续只作为离线 BC 数据，当前 runtime 轨迹只作为诊断数据，不混合两种 observation 语义。

在这一步完成前，不启动 phase-balanced、gripper-loss 或 A/B/C 正式候选实验，也不把 replay 成功率写成 SPI/Module 结果。
