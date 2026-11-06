# LIBERO task 4/7 Full Policy 闭环 close/release 连续段响应诊断（2026-10-31）

## 目的与边界

本项继续使用同一份状态-动作对齐 artifact，分析 Policy 夹爪命令的连续 close/release 段及其公开 joint/gripper state 响应。它只读取已有 120 条轨迹，不新增 rollout、不训练、不重建模型，也不访问 qualification、validation、held-out 或 final evaluation。

`close run` 定义为连续的 `action[6] > 0`，`release run` 定义为首次 close 之后连续的 `action[6] < 0`。run 的长度和 state path 会受 episode 成功提前终止或 time-limit 截断影响；因此它们是命令轨迹描述，不是完整技能阶段，也不是因果或动力学估计。

## 来源与验证

- 输入：`/tmp/libero-task4-task7-full-policy-closed-loop-state-action-alignment-20261031-v1.json`，SHA-256 `d50d98c40b2152d66026acc72369ca5c3233b6b329814a3b847cbc76ce7d3b77`。
- 脚本：[experiments/analyze_libero_task4_task7_full_policy_closed_loop_phase_response.py](../experiments/analyze_libero_task4_task7_full_policy_closed_loop_phase_response.py)，SHA-256 `f91dd7ce3d21463b26a3c134993c429413d6ec2be373356f05ba4bcf834f56a0`。
- 输出：`/tmp/libero-task4-task7-full-policy-closed-loop-phase-response-20261031-v2.json`，SHA-256 `b95161733640fb39a9be5a68eb6a8a7597377598d63a9616b9df7b4ac0bc47d5`。
- 新增 phase-response 测试 `4 passed`；脚本 `py_compile` 通过。

## 结果

### Outcome 分层

| outcome | episode | 无 close | close 后无 release | 首个 close run 长度 | 首个 release run 长度 |
|---|---:|---:|---:|---:|---:|
| success | 7 | 0 | 3 | 112.57/115 (`n=7`) | 2.75/2.5 (`n=4`) |
| time-limit | 113 | 32 | 15 | 89.40/35 (`n=81`) | 95.24/60.5 (`n=66`) |

失败轨迹中 32 条没有 close，另有 15 条 close 后没有 release；其余 66 条出现 release 后仍然运行到 time-limit。出现 release 的失败轨迹通常保持负向命令较长，但这是因为失败 episode 多运行到 300 步，存在明显的终止截断影响，不能将长 release 段解释为失败的直接原因。

### Task 分层

| task / outcome | episode | 无 close | close 后无 release | 首个 close run 长度 | 首个 release run 长度 |
|---|---:|---:|---:|---:|---:|
| 4 / success | 3 | 0 | 2 | 113.0/89 | 3.0/3.0 (`n=1`) |
| 4 / time-limit | 57 | 5 | 11 | 60.52/25.5 (`n=52`) | 108.22/80 (`n=41`) |
| 7 / success | 4 | 0 | 1 | 112.25/116 | 2.67/2 (`n=3`) |
| 7 / time-limit | 56 | 27 | 4 | 141.17/149 (`n=29`) | 73.96/44 (`n=25`) |

close run 的方向依赖任务：task 4 成功组比失败组更长，task 7 则相反。因而“统一延长 close”不能作为跨任务修复规则。task 7 的主要差异仍是大量失败轨迹没有 close，而不是 close run 普遍过短。

### 公开状态响应

首次 close run 内的 joint step path 均值为 success `2.768`、time-limit `2.099`；gripper step path 为 `0.0683`、`0.0594`。但 task 4 成功/失败的 joint path 为 `2.799/2.038`，task 7 为 `2.744/2.210`，这些差异与 run 长度和 episode 截断耦合，不能转化为控制效率或目标接近结论。

首次 release run 的 joint path 均值为 success `0.0326`、time-limit `1.502`，gripper path 为 `0.00536`、`0.0349`。这里的差异主要反映失败轨迹持续执行 release 命令；成功轨迹通常在短 release 段后结束或根本没有 release。

首次 close run 的 joint offset L2 变化均值为 success `0.924`、time-limit `0.589`，gripper offset L2 变化为 `-0.00161`、`0.00080`。task 4/7 的方向不同，且 offset L2 不是带方向的目标距离，因此只能作为公开 reset-relative 状态描述。

## 解释边界与下一步

这项分析把失败拆成三种命令模式：无 close、close 后无 release、release 段持续到 time-limit。它没有证明任何一种模式是根因，也不能从公开 state delta 判断物体是否被抓取、是否接近目标或是否完成真实 release。所有统计都来自此前已使用过的 candidate nonregression 初态。

当前 120 条轨迹预算已用尽。若要进一步做 RGB observation replacement 或目标相对几何诊断，必须先冻结新的非 formal 配置、固定 checkpoint/初态/短窗口和 provenance；在此之前不新增 rollout、不调参、不更新 SPT/Knowledge、不注册 Module。
