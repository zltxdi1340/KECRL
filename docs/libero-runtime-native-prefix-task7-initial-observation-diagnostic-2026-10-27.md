# LIBERO task 7 初始观察与早期动作差异诊断（2026-10-27）

## 目的与边界

task 7 的 support-vs-free-run 分解已经表明，两臂都能在 support observation 上学到 close 阶段，但 HDF5 臂在 candidate state `0/8` 的前 80 步没有 close。这里继续检查 close 延迟是否从 candidate 初始 observation 就已经出现，还是先由第一步之后的运动差异造成。

协议：[initial observation diagnostic](../configs/libero_runtime_native_prefix_task7_initial_observation_diagnostic_v1.json)，SHA-256 `6e86399b61868a6fc2fa81d518f26f4d8d9e77f6ce74472e70a1a7368cd2a86c`。结果位于实验主机 `/tmp/libero-runtime-native-prefix-task7-initial-observation-20261027-v1.json`，SHA-256 `116c313f9d749e6c777de07a8279d1d15065f125e08014f8194c2be3cfc3a19c`。运行提交 `6bade954096c05968df183d94eebfaa303e7cd15`，Python `3.10.22`、PyTorch `2.4.1+cu124`、`cuda:1` / RTX 4090，耗时约 `242.62s`。

重建 task 7、training seed `20261022` 的 historical HDF5 和 runtime-native 两个 500-update 模型。每个模型按 `candidate_nonregression` 全部 `0..19` 顺序复用同一 adapter；只保存 state `0/8` 的前 20 步详细结果。source 使用 30 条 `meta_support` prefix，candidate 初态和前一项轨迹结果均通过 SHA 核验。没有访问 qualification、validation 或 held-out，没有选择 checkpoint、更新 SPT/Knowledge 或注册 Module。

## 结果

两个 candidate 初态在两臂中的公开初始 joint/gripper 状态完全相同，因为它们共享同一个固定环境初态。第一个动作都保持 open：

| 初态 | 两臂首动作 L2 差 | 平移分量差 | 旋转分量差 | gripper 分量差 | HDF5 前 20 步 close | runtime-native 前 20 步 close |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.211 | 0.175 | 0.017 | 0.116 | 无 | 无 |
| 8 | 0.261 | 0.197 | 0.020 | 0.170 | 无 | 无 |

前 20 步的平均六维运动命令 L2 为：state 0 HDF5 `0.525`、runtime-native `0.619`；state 8 HDF5 `0.569`、runtime-native `0.626`。结合上一项完整轨迹，runtime-native 随后在 state 8 的第 37 个动作、state 0 的第 65 个动作开始 close；HDF5 两个状态在 300 步内仍没有 close。

以每个 Policy 自己的视觉编码为距离空间，candidate 初始 RGB 到 30 条 support 起始 observation 的最近距离为：

| 初态 | HDF5 最近视觉距离 | runtime-native 最近视觉距离 | HDF5 最近 support demo | runtime-native 最近 support demo |
|---:|---:|---:|---:|---:|
| 0 | 0.613 | 0.699 | 13 | 13 |
| 8 | 0.545 | 0.695 | 24 | 12 |

这些距离没有经过跨任务或跨 seed 标定，不能作为 OOD 阈值。它们只说明两臂的 candidate 初始 RGB 在各自 learned feature 空间中的近邻排序和距离不同。对应的 proprioception 最近距离很小（state 0 `0.0389`，state 8 `0.0432`），因此当前差异主要表现在视觉输入和动作映射，而不是初始 joint/gripper 数值不同。

## 判断

1. close 缺失不是在第一步就由一个错误的 gripper 标签造成：两臂在前 20 步都按 open 阶段执行，符合 support 目标的早期阶段。
2. 两臂从相同公开 proprioception 初态开始就产生明显的平移动作差异，说明 observation arm/Policy 条件化会在第一步改变机械臂运动轨迹；这为后续状态漂移提供了入口。
3. 现有证据支持“早期运动差异与 free-run 状态反馈共同导致 close 时相延迟或消失”这一诊断假设，但还没有把视觉编码、动作 head、控制误差和状态反馈分离开。最近 support feature 距离也不是因果证据。
4. 不能把 runtime-native 的一次成功升级为 observation-source 优势：support 夹爪预测几乎相同，candidate 只覆盖 task 7 一个训练 seed 的两个状态，且 runtime-native state 0 仍然失败。

## 下一步

下一项固定这两个模型，做 teacher-forced 与 free-run 的逐步 action divergence：在 candidate 初态的真实 runtime observation 上记录第一步后的公开 observation，然后分别用同一模型输入、替换为另一臂首步动作的状态轨迹，比较第 1、3、5、10、20 步的 joint/RGB/action 偏差。该反事实只用于机制诊断，不改变训练数据、不选择 checkpoint，也不访问 qualification。
