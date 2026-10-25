# LIBERO CNN-only phase-uniform 闭环诊断（2026-10-25）

## 目的与边界

离线配对诊断显示，在 5 个 source task、2 个 seed 上，CNN-only phase-uniform Policy 的 query action MAE 为 `0.11708`，优于 phase-balanced 的 `0.12295`。本诊断检查这一离线基线重建后能否在当前 LIBERO runtime 执行完整任务链。

- Policy：CNN-only；七个动作维度等权的 `uniform_action_mse`；每个 task-seed 固定 500 次 Adam update。
- 任务与 seed：task `0/1/3/4/7`，seed `20261021/20261022`。
- 在线状态：每个 source task 的 `candidate_nonregression` 初始状态索引 `0/1/2`，每个 episode 最多 300 个策略动作，reset 稳定化 5 步。
- 完整设计：10 个 task-seed Policy，共 30 个闭环 episode。每个重建模型在 rollout 前均与离线配对结果的 uniform-arm 最终参数 SHA-256 精确匹配。
- 边界：未读取 meta-query、validation、held-out 或 qualification；不做 checkpoint 选择；不更新 SPT/Knowledge，也不注册 Module。私有 simulator state 不进入 Policy 或 Knowledge。

配置：[CNN-only uniform closed-loop diagnostic](../configs/libero_cnn_only_uniform_closed_loop_diagnostic_v1.json)，SHA-256 `d22e6446924ab0e78675ffda31e5b499ed2c43cc6b0a2b5f75cfe9151ffd4cdf`。

结果文件保存在实验主机 `/tmp/libero-cnn-uniform-closed-loop-20261025-v1.json`，SHA-256 `b57badce5287aeb9ff161fbad9eda6f6dd5a200664111f327f5895250db537e4`。结果引用的离线配对 JSON SHA-256 为 `249fe27c13909dce98a17c5edbc6aa68c5ac3cb7cf527ac86fa1c0435e668c03`；训练代码提交 `1830179c0a27a780235faaf8e6f5214430e2c4fd`。闭环 runner 在提交 `2ceecaf9c5510f92958b6ba436846ae7000626c5` 上运行。Python `3.10.22`、PyTorch `2.4.1+cu124`、CUDA `12.4`、RTX 4090 `cuda:0`，总耗时约 607 秒。

## 结果

| task | source geometry | 成功 | 成功率 | 终止原因 | 平均步数 |
|---:|---|---:|---:|---|---:|
| 0 | table region | 0/6 | 0.000 | time limit 6 | 300.0 |
| 1 | table region | 0/6 | 0.000 | time limit 6 | 300.0 |
| 3 | movable object surface | 2/6 | 0.333 | success 2, time limit 4 | 242.2 |
| 4 | open container | 0/6 | 0.000 | time limit 6 | 300.0 |
| 7 | fixture surface | 0/6 | 0.000 | time limit 6 | 300.0 |
| **合计** |  | **2/30** | **0.067** | **success 2, time limit 28** | **283.1** |

两次成功都出现在 task 3、seed `20261021`：candidate 初始状态索引 0 在 145 步成功，索引 2 在 108 步成功。该 task-seed 的索引 1 失败；seed `20261022` 的三个状态均失败。seed `20261021` 总体成功率为 `2/15`，seed `20261022` 为 `0/15`。全部失败 episode 均到达 300 步 time limit，累计 sparse reward 为 0。

## 解释

1. **当前离线 BC 策略尚未形成可复用的闭环基础策略。** 30 个预先固定的 candidate 状态中只有 2 个成功，而且成功集中于单一 task-seed；不能据此进入正式 SPT 更新效果比较或 Module 资格检查。
2. **离线 query MAE 不能替代 runtime 成功率。** 当前数据再现了离线动作拟合与在线任务完成之间的明显落差。该诊断没有记录逐步 RGB、动作或本体序列，因此不能判断失败来自动作输出停滞/饱和、夹爪时序、闭环分布偏移或环境控制细节。
3. **范围有限。** 每 task-seed 仅测试 3 个 candidate-nonregression 初始状态，结果是可行性诊断而非正式性能估计；没有访问 qualification 或 held-out 状态，也没有跨 seed 任务顺序更新。

## 下一步

对预先指定的代表性状态记录逐步 7 维动作、公开 joint/gripper observation、动作变化和成功/终止标志，覆盖 task 3 成功 episode 以及 task 0/1/4/7 失败 episode。先判断策略是否持续输出近常数/饱和动作，或动作已变化但观测状态不按预期演化；再决定单变量地改动作控制、数据覆盖或状态反馈。若逐步 trace 暴露环境动作语义问题，应先修复并复核接口；若策略在状态不变时持续重复错误动作，继续查视觉/本体表示和在线分布漂移。此后才扩展固定候选状态的闭环复测，仍不接触资格划分。
