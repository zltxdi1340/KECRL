# LIBERO task 4/7 Full Policy observation/state replacement 诊断结果（2026-10-31）

## 目的与协议

本项执行了已获授权的 v2 短窗口诊断：对 task 4/7、三个已有 training seed、candidate nonregression 初态 `0..19`，使用 source 闭环结果中固定的六个 Full Policy checkpoint，每条 episode 只采集 20 个 pre-action observation。自然分支执行环境动作；public joint/gripper reset replacement 只在离线 Policy 前向中使用，替换动作不送回环境。

本项没有训练、没有 checkpoint 选择、没有完整 300-step rollout，也没有访问 qualification、validation、held-out 或 final evaluation。120 条短窗口 episode 的 `time_limit` 是 capture horizon 截断，不是新的任务成功率。

## Provenance 与范围

- v2 配置：[observation-state-replacement-diagnostic_v2.json](../configs/libero-task4-task7-full-policy-observation-state-replacement-diagnostic_v2.json)，SHA-256 `376bf80852e9d0a144acabab41e04e3c787ea553c0b1abc9183163870ae95e55`。
- runner：[run_libero_task4_task7_full_policy_observation_state_replacement_diagnostic.py](../experiments/run_libero_task4_task7_full_policy_observation_state_replacement_diagnostic.py)，SHA-256 `75e7ce37261d5f99ae56af08b30e03bdee2ed4df0671482725fdb0b119a683f8`。
- 结果：`/tmp/libero-task4-task7-full-policy-observation-state-replacement-20261031-v2.json`，SHA-256 `1f225135c3dd1da796660cbb470fc055551a6d4edb03e0794647afc1b96404a9`。
- observation archive：`/tmp/libero-task4-task7-full-policy-observation-state-replacement-archive-20261031-v2`，120 个 `.npz`，manifest SHA-256 `14cc497119acb76d5c2e817b3a513172b5d72b8b0ac6c7b232fd4a6bd220a23f`。
- Runtime：Python `3.10.22`、PyTorch `2.4.1+cu124`、`cuda:0` / RTX 4090；耗时约 `230.20s`。
- 采集规模：120 episodes、2400 natural environment actions；replacement action 全部离线计算。

六个 checkpoint 的 file SHA 与 weight SHA 均重新核对并与 source 闭环结果一致；natural action 在归档后重新前向复现，逐步 action 通过 `1e-6` 容差核验。每个 archive 文件的 shape、dtype、SHA 和 manifest 均通过检查。

## 结果

### Public state reset replacement 的 action 差异

下表为每个 episode 先在前 `h` 步求 action 差异均值，再在分组内求均值。replacement 保持两路 RGB 不变，只把 joint/gripper 替换为该 episode 第 0 步公开 reset 值。Full Policy 是 CNN-only，因此 shared/independent 两种 hidden mode 数值完全相同，不能解释为 recurrent hidden 对照。

| horizon | action L2 | translation L2 | rotation L2 | gripper abs |
|---:|---:|---:|---:|---:|
| 1 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 3 | 0.003391 | 0.003119 | 0.000169 | 0.000656 |
| 5 | 0.008324 | 0.008038 | 0.000478 | 0.000802 |
| 10 | 0.025073 | 0.024858 | 0.001562 | 0.000646 |
| 20 | 0.076515 | 0.076158 | 0.005264 | 0.000620 |

第 20 步 public reset-relative state drift 的均值为 joint `0.22169`、gripper `0.01990`；对应 action 差异主要来自 translation，gripper action 变化较小。

### Task 差异

| task | 第 20 步 action L2 | translation L2 | rotation L2 | gripper abs | joint reset drift |
|---:|---:|---:|---:|---:|---:|
| 4 | 0.065864 | 0.065359 | 0.008086 | 0.000081 | 0.301590 |
| 7 | 0.087166 | 0.086958 | 0.002442 | 0.001158 | 0.141796 |

task 7 的 joint drift 较小，但 action 对 public state replacement 的敏感性较高；task 4 的 joint drift 较大，但 action 差异较低。这说明“状态漂移大小”与“动作映射敏感性”不是同一个量，不能单独将 joint drift 写成根因。

### 首次 close 与 source outcome 分层

20 步窗口内，117/120 条 natural trace 没有 close；replacement 没有提前产生 close。其余 3 条是 source full-run 中的失败轨迹，replacement 也没有把 close 提前到 natural 之前。7 条 source full-run 成功轨迹全部在前 20 步没有 close，因此本短窗口没有新的成功/失败判定。

按 source full-run outcome 分层，第 20 步 action L2 为：source success `0.078606`，source time-limit `0.076386`。该分层只是复用已有 300-step 诊断标签，不能解释为 replacement 后的成功率或因果效应。

## 解释边界

这项结果支持一个有限的局部描述：固定 Full Policy 在前 20 步对 public joint/gripper reset replacement 有可测 action sensitivity，且 task 7 的 action sensitivity 高于 task 4；差异主要在 translation action。它不能区分视觉 encoder、fusion、数据覆盖、控制器或状态反馈的相对因果贡献，也不能预测长程闭环成功率。

所有初态都是此前使用过的 candidate nonregression `0..19`。本项不构成 unseen generalization、正式 SPI、Knowledge Evolution 或 Module qualification；不更新 SPT/Knowledge，不注册 Module。

## 下一步

当前可执行的短窗口 observation/state replacement 已完成。若要把局部 action sensitivity 连接到长程控制，必须另行冻结更严格的干预协议，明确是否允许新的 episode budget、替换对象和成功判定；不能直接把本项的 action difference 当作训练变量收益或闭环成功预测。
