# LIBERO CNN-only uniform Policy 稀疏 RGB 快照诊断（2026-10-26）

## 目的与边界

上一项逐步 trace 诊断确认 7 条代表性轨迹中的机械臂动作和公开 joint/gripper 状态都在变化，但没有记录 Policy 实际看到的相机画面。本次在完全相同的 7 条 candidate-nonregression 轨迹上，按预先固定的时刻保存 `agentview_rgb` 和 `eye_in_hand_rgb`，检查末端接近、目标相对位置和抓取前后的可见状态。

本次仍是机制诊断，不是正式 SPI、SPT 更新或 Module 资格结果：

- Policy 是 5 个 source task、固定 seed、每个 task-seed 500 次更新重建的 CNN-only `uniform_action_mse` arm；不选择 checkpoint。
- 轨迹固定为 task `0/1/3/4/7` 的 7 个代表性 candidate-nonregression 初态，其中包含 task 3 / seed `20261021` 的 1 条来源成功轨迹和 1 条来源失败轨迹。
- 每条轨迹在 step `0/25/50/75/100/125/145/150/225/300` 保存两路 Policy-facing RGB；成功提前终止后不补造后续帧。
- 不读取私有 simulator state、物体坐标或成功判定内部变量作为 Policy 输入；不访问 meta-query、validation、held-out、qualification，不更新 SPT/Knowledge，不注册 Module。

配置：[RGB snapshot diagnostic config](../configs/libero_cnn_only_uniform_policy_rgb_snapshot_diagnostic_v1.json)，SHA-256：`9947b312e393acf14bf97a3cd020f546918cdd628567d6665de481288f5127fa`。

## 可复现性与结果

来源 trace 结果为 `/tmp/libero-cnn-uniform-policy-traces-20261025-v1.json`，SHA-256：`50a7799fdd9afe428823df67d753b020a946c662d7746cdc692f7a9005428b8c`。本次结果为 `/tmp/libero-cnn-uniform-policy-traces-rgb-20261026-v1.json`，SHA-256：`223b1fe8a02b1caf8a7812ab41ec164f59528f41eebcc7ce52e9970425994bce`。PNG 位于 `/tmp/libero-cnn-policy-rgb-snapshots-20261026-v1/`，未写入仓库。

运行时为 `libero-kecrl`、Python `3.10.22`、PyTorch `2.4.1+cu124`，使用 `cuda:1` / RTX 4090；本次过程约 273 秒。7/7 条重放均通过既有成功标志和终止步数校验，且每个 task-seed 重建模型的参数 SHA-256 均与既有 paired uniform arm 相同。

| task / seed / init | 结果 | 步数 | RGB 帧数 |
|---|---:|---:|---:|
| 0 / 20261021 / 0 | 失败，time limit | 300 | 20 |
| 1 / 20261021 / 0 | 失败，time limit | 300 | 20 |
| 3 / 20261021 / 0 | 成功 | 145 | 14 |
| 3 / 20261021 / 1 | 失败，time limit | 300 | 20 |
| 3 / 20261022 / 0 | 失败，time limit | 300 | 20 |
| 4 / 20261021 / 0 | 失败，time limit | 300 | 20 |
| 7 / 20261021 / 0 | 失败，time limit | 300 | 20 |
| **合计** | **1/7 成功** |  | **134** |

## 画面观察

以下判断来自稀疏的 64×64 `agentview` 和 `eye_in_hand` 帧，并与同一轨迹的 action/joint/gripper trace 对齐：

1. **成功轨迹不是静止或无接近。** task 3 / seed `20261021` / init `0` 中，agentview 从 step 0 到 75、125、145 显示末端持续改变姿态并进入目标所在区域；eye-in-hand 视野也随接近发生明显变化，轨迹在 145 步成功结束。
2. **同 task、同 seed 的初态仍会改变结果。** task 3 / init `1` 的 Policy 与成功样本使用同一个 task-seed 模型。它也产生持续运动并接近工作区，但在 step 150 到 300 没有形成稳定的完成关系，最终 time limit。该对照说明“模型已经学会动作”不能由一条成功样本推广到同任务所有 candidate 初态。
3. **部分跨任务失败表现为末端偏离目标工作区。** task 1 和 task 7 的 agentview 序列中，末端在中后段移向画面右侧/上侧并持续变化，没有在目标区域保持可见的接近—闭合—搬运关系。task 0 和 task 4 也有明显关节运动，但在稀疏帧中未形成稳定的目标接触和后续放置链。
4. **夹爪动作仍是必要但不是充分解释。** 之前的 trace 已发现 task 1 缺少正向 close，而 task 3 的失败样本和 task 7 都存在较长 close run；本次画面没有给出物体接触的私有坐标，因此不能把失败归结为单一夹爪时序问题。

## 诊断结论

本次结果排除了“Policy 完全不读图”或“adapter 完全不执行动作”这类最简单解释：成功样本和失败样本都产生了可见的末端运动，成功样本的画面确实呈现出向目标区域接近的过程。结果同时支持两个后续需要单独拆分的候选因素：

- RGB 读出后的空间接近和闭环反馈可能不足，导致末端不能持续保持在目标相对位置；
- candidate 初态变化会放大当前 Policy 的动作链误差，尤其是接近后的抓取/搬运阶段。

这组帧不能判定根因属于 CNN 表示、action head、训练状态覆盖、动作时序还是历史 HDF5 与当前 runtime 的输入语义差异。快照稀疏、分辨率低，未与同初态 expert RGB 逐步配对，也没有目标/末端私有坐标或干预，因此不能用于 checkpoint 选择、成功率估计或 Module 资格。

## 下一项诊断

保持同一 7 条初态和同一 CNN-only uniform Policy，构造“expert 观测—Policy 观测—动作”阶段对齐诊断：只比较预先固定的 approach、transport、release 窗口，报告相机帧变化、公开 joint/gripper 变化和动作误差；不把当前 runtime 生成的短前缀当作完整 HDF5 示教，也不访问 qualification。该对照用于区分：Policy 是否在已经看到目标后仍输出错误空间动作，还是在输入分布漂移后失去目标相对位置。只有这一步形成可重复的单变量证据后，才决定改 action head、扩大状态覆盖或重新处理输入语义。

