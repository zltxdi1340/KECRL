# LIBERO CNN-only phase-uniform Policy 逐步轨迹诊断（2026-10-25）

## 目的与边界

上一项固定 candidate-nonregression 闭环诊断中 30 个 episode 仅成功 2 次。这里抽取 7 条代表性轨迹，检查 Policy 输出是否停滞、Panda 关节是否响应，以及夹爪动作与任务成败的关系。

- 轨迹：task 0、1、4、7 各一条失败；task 3 / seed `20261021` 的一条成功和一条失败；task 3 / seed `20261022` 一条失败。
- Policy：逐条重建 500-update CNN-only phase-uniform 模型，运行前核对与离线 paired uniform arm 的最终参数 SHA-256；7 条重放都复现来源成功标记和终止步数。
- 记录：每步 7 维动作、action 后的公开 joint/gripper observation 和 reward；不记录图像帧或私有 simulator state。
- 在线状态：仅 source task 的 `candidate_nonregression` 状态索引；不访问 meta-query、validation、held-out 或 qualification，不选择 checkpoint，不更新 SPT/Knowledge，不注册 Module。

配置：[Policy trace diagnostic](../configs/libero_cnn_only_uniform_policy_trace_diagnostic_v1.json)，SHA-256 `d72d478859646d82cfaab1511017ad9b2f98a7bbd324c6ec0b2d18f357660ab3`。

逐步结果保存在实验主机 `/tmp/libero-cnn-uniform-policy-traces-20261025-v1.json`，SHA-256 `50a7799fdd9afe428823df67d753b020a946c662d7746cdc692f7a9005428b8c`。结果生成的代码提交为 `76b2865b47ba17a0c235707ab2da8f2532d8e8b5`；CUDA `cuda:0` / RTX 4090，轨迹采集和模型重建约 299 秒。

## 结果

`motion near-zero` 是前六个机械臂动作维度 L2 不超过 `0.05` 的比例；`close` run 是第七维预测值大于 `0.1` 的最长连续长度；`joint delta` 是相邻时刻七维 joint observation 的平均 L2 差。

| task / seed / init | outcome | open 命令数 | close 命令数 | 最长 close run | 夹爪符号切换 | mean joint delta |
|---|---|---:|---:|---:|---:|---:|
| 0 / 20261021 / 0 | 300 步失败 | 260 | 36 | 36 | 4 | 0.0126 |
| 1 / 20261021 / 0 | 300 步失败 | 299 | 0 | 0 | 0 | 0.0123 |
| 3 / 20261021 / 0 | 145 步成功 | 57 | 78 | 78 | 2 | 0.0293 |
| 3 / 20261021 / 1 | 300 步失败 | 189 | 107 | 107 | 2 | 0.0189 |
| 3 / 20261022 / 0 | 300 步失败 | 246 | 52 | 47 | 2 | 0.0253 |
| 4 / 20261021 / 0 | 300 步失败 | 187 | 98 | 79 | 3 | 0.0281 |
| 7 / 20261021 / 0 | 300 步失败 | 121 | 170 | 170 | 2 | 0.0153 |

当前 LIBERO runtime 使用 Panda gripper；其动作语义为 `-1 = open`、`+1 = close`。逐步 trace 中的夹爪状态也显示了相应的开合变化。仅作为训练数据动作分布参考，30 条 `meta_support` 示教的第七维最长连续正命令 run 中位数为：task 0 `44` 步、task 1 `63` 步、task 3 `44` 步、task 4 `87` 步、task 7 `92.5` 步。这个示教分布不是与 candidate 状态逐轨迹配对的成功保证。

## 诊断判断

1. **不是零动作或 adapter 无响应。** 七条轨迹的机械臂动作均未出现 near-zero 段，joint state 持续变化。简单的“Policy 输出常数/环境没有执行命令”解释不符合观测。
2. **部分失败伴随缺少或过短的夹爪 close 命令。** task 1 全程没有正向 close 命令；task 0 最长 close run 为 36 步，随后又打开。这与该策略未完成抓取相容，但仅凭 trace 还不能确认物体接触位置。
3. **close 时长单独不能解释成功。** task 3 一个失败初始状态有 107 步 close run，task 7 有 170 步，仍然失败；task 3 同一模型的成功轨迹则在持续 close 后成功。因此 grasp 时序有影响，但空间对准、 approach 路径和闭环观察也可能是限制因素。
4. **当前证据还不能确认视觉定位或控制器是剩余瓶颈。** 这些轨迹没有相机帧、末端执行器位姿或物体位姿；joint changes 只能说明机械臂关节在变化，不能说明末端接近了目标。

## 下一步

在同样固定的 7 条状态和同一 Policy 上，只额外保存若干预先固定时刻的 agentview 与 eye-in-hand policy RGB 帧，并与已有动作/关节 trace 对齐。先人工核对机械臂末端相对目标物的位置变化以及抓取发生前后的可见状态，再选择一个单变量诊断：如果轨迹显示目标接近错误或持续偏离，查 RGB/空间动作回归；如果已接触但 close/open 时序错误，单独查夹爪动作读出。该观察不使用私有物体坐标，也不改变 checkpoint 或资格划分。
