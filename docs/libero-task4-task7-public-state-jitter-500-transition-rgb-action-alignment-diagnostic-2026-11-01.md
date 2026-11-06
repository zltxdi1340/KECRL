# LIBERO task 4/7 500-update transition 前 20 步 RGB/action 对齐审计（2026-11-01）

## 结果状态

本项按只读边界检查了 7 条 success-transition 轨迹的前 20 步。现有 baseline、jitter 和 transition artifacts 都没有 `agentview_rgb`、`eye_in_hand_rgb`、RGB archive 或对应 frame digest，因此严格的 RGB/action 对齐在当前证据中 **不可估计**。没有重放环境、没有补采 RGB、没有读取 qualification 数据。

可估计的部分是 action-only alignment：从两份闭环 JSON 的 `stepwise_public_state_trace` 提取前 20 步七维 action，比较 jitter−baseline 的 L2、translation、rotation 和 gripper 差异。

## Provenance

- baseline 输入：`/tmp/libero-task4-task7-full-policy-closed-loop-20261030-v1.json`，SHA-256 `03439cc0172f3ba29ee4527d3c64d1da16edd93d2eaf63134344e4c8b15f49ce`。
- jitter 输入：`/tmp/libero-task4-task7-public-state-jitter-500-closed-loop-20261031-v1.json`，SHA-256 `8c9c6167f316072efeba909eb5a2dbf8987369698eddbf5d496bdf8320897ed2`。
- transition 输入：`/tmp/libero-task4-task7-public-state-jitter-500-closed-loop-transition-mechanisms-20261031-v1.json`，SHA-256 `dd25d2a3bf9a392e6a9b2306e68f0e0f52ee704f172a13364e7dac2d1dc19211`。
- 输出：`/tmp/libero-task4-task7-public-state-jitter-500-transition-rgb-action-alignment-20261101-v1.json`，SHA-256 `6487168e9393906ba3ab8bc05fb02b444dc73cfaed8743a7ade3d6cfc999de17`。
- 分析脚本：[analyze_libero_task4_task7_public_state_jitter_500_transition_rgb_action_alignment.py](../experiments/analyze_libero_task4_task7_public_state_jitter_500_transition_rgb_action_alignment.py)。

## 前 20 步 action alignment

所有 7 条轨迹的前 20 步都没有 close command，因此 `first_close_within_window` 对两臂均为 `None`。七条轨迹的分组均值如下：

| transition | 条数 | 首动作 delta L2 | 20 步 mean action delta L2 | translation delta L2 | rotation delta L2 | gripper delta |
|---|---:|---:|---:|---:|---:|---:|
| jitter-only | 3 | `0.0903` | `0.0676` | `0.0587` | `0.0089` | `0.0190` |
| baseline-only | 4 | `0.2151` | `0.1330` | `0.1099` | `0.0115` | `0.0457` |

baseline-only 转移的前 20 步 action 差异更大，主要来自 translation 与 gripper 项；jitter-only 转移的差异较小，但这只是按 success transition 分组后的描述性结果，不能解释为 jitter 的因果作用。由于 RGB frame 不存在，不能判断这些 action 差异是否由视觉观测差异、状态输入或 action head 读出造成。

## 解释边界与下一步

当前证据只支持：7 条 transition 的前 20 步 action 输出可以离线对齐；RGB/action 联合对齐缺少必要 RGB 输入。若要完成真正的 RGB/action 对齐，需要在单独冻结的新 rollout 协议中保存两臂对应的 pre-action `agentview_rgb`、`eye_in_hand_rgb` 和 frame digest；这将是新数据采集，不属于本次只读分析，不能把现有结果补写成 RGB 结论。

因此不更新 SPT/Knowledge，不注册 Module，也不访问 qualification/held-out。

## 验证证据

- RGB/action alignment 专项测试：`2 passed`。
- 相关回归集合：`63 passed`。
- RGB key/archive 缺失审计、7 条 action window 和 provenance 独立复核通过。
- `py_compile` 与 `git diff --check` 通过。
