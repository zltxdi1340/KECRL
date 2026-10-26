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

## 与已有阶段诊断的关系

本仓库已经有一项独立的 [分层 expert-state / free-run 误差诊断](libero-stratified-error-diagnostic-2026-10-25.md)：在同一 5 个 source task、500-update causal Policy 和 seed `20261021` 上，它使用 meta-support/meta-query 示教，按 approach、transport、release 统计 action、joint-state 和 RGB 漂移。该诊断发现 approach 已有非零动作误差，transport 的状态漂移通常放大，task 4 open-container 的 query transport drift 和 eye-in-hand RGB MAE 最大；release 样本较少，不能单独作为全局夹爪根因。

这两项证据不能合并成一个成功率或误差估计：分层诊断使用的是 source 示教前缀，RGB 快照诊断使用的是 candidate-nonregression 初态的 Policy free-run。它们在机制上相互补充，支持“动作误差在闭环中累积为空间偏离”的工作假设，但仍未区分表示、action head、状态覆盖和输入协议的因果贡献。

## 下一项诊断

阶段误差、输入替换敏感性和 source-support replay 可行性审计已经完成。下一项不再重复这些检查，而是在同一批**实际可执行的 runtime 前缀**上配对比较历史 HDF5 观测训练与 runtime-native 观测训练：两臂使用相同动作目标、前缀长度、Policy、预算和 candidate 评估边界，只改变 observation 来源。该 pilot 用来判断输入语义差异是否值得建立独立 runtime-native 训练协议；它不能把 17/150 条完整前缀外推为全量示教，也不进入 qualification、held-out、SPT 更新或 Module 注册。

