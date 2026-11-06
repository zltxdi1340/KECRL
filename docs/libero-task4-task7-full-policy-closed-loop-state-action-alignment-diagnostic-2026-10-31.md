# LIBERO task 4/7 Full Policy 闭环状态-动作对齐诊断（2026-10-31）

## 目的与边界

本项在上一项 120 条闭环轨迹归因的基础上，分析固定 Full Policy action 与公开 joint/gripper state 的逐步响应，重点检查首步、首次 close 前后 `-2..+2` 步窗口和前 1/3/5/10/20 步窗口。它只读取已有 JSON 中的公开 `action`、`joint_step_delta`、`gripper_step_delta` 和 offset，不重放环境、不重建模型、不训练，也不读取 RGB、私有 simulator state、qualification、validation、held-out 或 final evaluation。

所有状态 delta 都是相对当前闭环轨迹的公开 post-action state 变化，不是 expert-action error，也不是环境动力学辨识。close 仍定义为 `action[6] > 0`；首次 close 窗口是 Policy 命令时序窗口，不是目标接近阶段。成功样本只有 7 条，所有比较均为描述性均值/中位数，不做显著性或因果检验。

## 来源与验证

- 输入：`/tmp/libero-task4-task7-full-policy-closed-loop-20261030-v1.json`，SHA-256 `03439cc0172f3ba29ee4527d3c64d1da16edd93d2eaf63134344e4c8b15f49ce`。
- 脚本：[experiments/analyze_libero_task4_task7_full_policy_closed_loop_state_action_alignment.py](../experiments/analyze_libero_task4_task7_full_policy_closed_loop_state_action_alignment.py)，SHA-256 `12e810c11aad4f289e8001d8288e2935a522699233b4d1f2ce59a84e1bee2f96`。
- 输出：`/tmp/libero-task4-task7-full-policy-closed-loop-state-action-alignment-20261031-v1.json`，SHA-256 `d50d98c40b2152d66026acc72369ca5c3233b6b329814a3b847cbc76ce7d3b77`。
- 新增状态-动作对齐测试和上一项分析测试共 `8 passed`；脚本 `py_compile` 通过。

## 结果

### 首步与首次 close 响应

下表为均值/中位数。首次 close 与 close 后指标只在相应轨迹存在时统计：成功组为 `7/7`，time-limit 组为 `81/113`。

| 指标 | success | time-limit |
|---|---:|---:|
| 首步七维 action norm | 0.9353/0.8524 | 0.9692/1.1190 |
| 首步 joint step delta L2 | 0.00752/0.00609 | 0.00864/0.00885 |
| 首步 gripper step delta L2 | 0.000389/0.000389 | 0.000393/0.000388 |
| 首次 close action norm | 0.6695/0.6090 | 0.6101/0.5463 |
| 首次 close joint step delta L2 | 0.02218/0.02393 | 0.02299/0.01738 |
| 首次 close gripper step delta L2 | 0.000702/0.000586 | 0.000733/0.000472 |
| close 后一步 joint step delta L2 | 0.01723/0.01651 | 0.02137/0.01543 |
| close 后一步 gripper step delta L2 | 0.002270/0.002208 | 0.002275/0.002277 |

首步 joint/gripper 响应在两类 outcome 间接近，说明这份公开 trace 没有显示一个统一的“第一步已经明显失控”的模式。首次 close 后，成功组的 joint delta 均值下降幅度较大，time-limit 组下降较小；但成功样本少，且该统计受 task/seed 和成功提前终止影响，不能解释为 close 导致成功。

### 前缀窗口

每个数先在单条 episode 的前 `h` 步求均值，再在 outcome 组内求均值，避免长 time-limit 轨迹支配短成功轨迹。

| 前缀步数 | success joint delta | time-limit joint delta | success gripper delta | time-limit gripper delta | success action norm | time-limit action norm |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.00752 | 0.00864 | 0.000389 | 0.000393 | 0.9353 | 0.9692 |
| 3 | 0.01194 | 0.01298 | 0.002591 | 0.002573 | 0.9302 | 0.9774 |
| 5 | 0.01460 | 0.01568 | 0.003749 | 0.003722 | 0.9856 | 1.0238 |
| 10 | 0.02048 | 0.02173 | 0.002552 | 0.002556 | 1.0861 | 1.1088 |
| 20 | 0.02985 | 0.03062 | 0.001313 | 0.001315 | 1.1681 | 1.1804 |

前 20 步的 gripper delta 几乎重合；joint delta 和 action norm 也只有小的均值差异。因而现有公开状态记录不支持把“早期机械臂响应幅度”单独当作跨 task 的根因。

### 首次 close 窗口（相对步）

`-1` 是首次 close 前一步，`0` 是首次正向 close，`+1/+2` 是 close 后的动作步。下表为均值；time-limit 组的 close 窗口只覆盖出现 close 的轨迹。

| 相对步 | success gripper action | time-limit gripper action | success joint delta | time-limit joint delta | success gripper delta | time-limit gripper delta |
|---:|---:|---:|---:|---:|---:|---:|
| -1 | -0.1686 | -0.1889 | 0.02724 | 0.02525 | 0.000414 | 0.000362 |
| 0 | 0.0660 | 0.1156 | 0.02218 | 0.02299 | 0.000702 | 0.000733 |
| +1 | 0.2889 | 0.1714 | 0.01723 | 0.02137 | 0.002270 | 0.002275 |
| +2 | 0.4526 | 0.3679 | 0.01464 | 0.02066 | 0.003622 | 0.003451 |

成功轨迹在 close 后的正向 gripper action 均值更高、joint delta 均值下降更快；但 task 分层方向不完全一致，且没有目标相对位置或专家 action 作为参照。这只能作为后续 phase/action 对齐的线索。

### Task 分层

| task / outcome | episode | 无 close | 首次 close joint delta | close 后一步 joint delta | close 后一步 gripper delta |
|---|---:|---:|---:|---:|---:|
| 4 / success | 3 | 0 | 0.02403 | 0.02333 | 0.002565 |
| 4 / time-limit | 57 | 5 | 0.02601 | 0.02527 | 0.002376 |
| 7 / success | 4 | 0 | 0.02078 | 0.01267 | 0.002049 |
| 7 / time-limit | 56 | 27 | 0.01757 | 0.01439 | 0.002095 |

task 7 的 time-limit 轨迹中 `27/56` 没有 close，task 4 为 `5/57`；这是最明显的任务条件差异。对于已经 close 的轨迹，task 4 和 task 7 的 joint/gripper 响应方向不完全相同，因此不能用一个全局 close timing 或 state-drift 规则解释两项任务。

## 解释边界与下一步

本项没有发现足以支持单一根因的公开状态响应模式：首步响应接近，close 后响应存在 task-specific 差异，且成功样本很少。`joint_response_ratio` 与 `gripper_response_ratio` 只是 action 与公开 state delta 的归一化描述，不是动力学参数或控制稳定性证明。

在当前“最多使用已有 120 条轨迹、不得重新训练”的预算下，不能再通过新环境 rollout 补采 RGB 或做 observation replacement；已有 2026-10-28 observation archive 属于不同 runtime-native prefix Policy，不能与本 Full Policy artifact 混用。下一步若要做真正的 observation/state replacement，必须先单独冻结新的非 formal diagnostic 配置，明确固定 checkpoint、候选初态、短窗口和 provenance；在获得该授权前保持当前协议，不访问 qualification、validation、held-out 或 final evaluation。
