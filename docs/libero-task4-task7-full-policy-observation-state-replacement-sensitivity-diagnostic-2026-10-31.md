# LIBERO task 4/7 Full Policy observation replacement sensitivity 分层诊断（2026-10-31）

## 目的与边界

本项只读取已完成的 v2 observation/state replacement artifact，按 task、training seed 和 source full-run outcome 分层，检查 public state reset drift 与 fixed Policy action delta 的 episode-level 关系。它不执行环境、不训练、不访问 qualification，也不把 Pearson 描述性相关写成显著性或因果结论。

## 来源与验证

- 输入：`/tmp/libero-task4-task7-full-policy-observation-state-replacement-20261031-v2.json`，SHA-256 `1f225135c3dd1da796660cbb470fc055551a6d4edb03e0794647afc1b96404a9`。
- 脚本：[analyze_libero_task4_task7_full_policy_observation_state_replacement_sensitivity.py](../experiments/analyze_libero_task4_task7_full_policy_observation_state_replacement_sensitivity.py)，SHA-256 `bb49ccc8e8caa640467a6c849caff62ef7b9013766f8eaa880ba818f251732b0`。
- 输出：`/tmp/libero-task4-task7-full-policy-observation-state-replacement-sensitivity-20261031-v1.json`，SHA-256 `e61495fe52a31b2a5d0cf023ff551be11b32f4986e5b29a9226856986d837b14`。
- 新增测试 `3 passed`；脚本编译通过。

## 第 20 步分层结果

相关系数使用每条 episode 的第 20 步 horizon 均值；它只描述同一分层内的线性共同变化。

| 分层 | episodes | joint reset drift 均值 | action L2 均值 | joint drift ↔ action L2 | joint drift ↔ translation L2 |
|---|---:|---:|---:|---:|---:|
| overall | 120 | 0.221693 | 0.076515 | -0.391 | -0.402 |
| task 4 | 60 | 0.301590 | 0.065864 | 0.694 | 0.689 |
| task 7 | 60 | 0.141797 | 0.087166 | 0.428 | 0.401 |
| source success | 7 | 0.207511 | 0.078606 | -0.778 | -0.782 |
| source time-limit | 113 | 0.222572 | 0.076386 | -0.363 | -0.374 |

task 4 和 task 7 内部的相关方向均为正，而 pooled overall 方向为负。这说明任务间的 drift 水平和 Policy sensitivity 共同造成混合相关反转；不能把 overall 负相关解释成“state drift 越大 action 越稳定”。source success 只有 7 条，相关数不稳定，不用于选择 checkpoint 或设置。

training seed 的第 20 步 action L2 均值为 `0.072833`、`0.071196`、`0.085517`（按 `20261022/20261023/20261024`），只作分层描述。

## 解释边界

该分析支持有限判断：public state reset replacement 的 action sensitivity 具有 task-conditioned 结构，task 4 的 joint drift 与 translation sensitivity 共同变化更明显，task 7 的平均 sensitivity 更高但 drift 更小。它仍不能区分视觉 encoder、fusion、控制器、数据覆盖或分布偏移的因果贡献，也不能预测长程成功。

当前不修改 loss、不训练新 Policy、不更新 SPT/Knowledge、不注册 Module。后续若要做单变量训练修复，必须先根据 task-conditioned 结果单独冻结新的非 formal 配置并保持 train/validation/final 角色分离。
