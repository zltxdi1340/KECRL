# LIBERO task 4/7 500-update transition RGB/action 短窗口诊断（2026-11-01）

## 后续复核修订

只读源前缀审计发现，新采集内部 baseline/jitter 的 reset digest 为 `7/7` 对齐，但与原始 300-step 闭环前缀相比，只有 task 7 的 `10/10` 个窗口完全复现；task 4 的 `4` 个窗口 reset digest 和 action/proprioception 不一致，action 分量最大绝对差为 `0.099773`。下列两组汇总须解释为新采集窗口的局部描述，不能全部称为原来 7 条 success transition 的精确复现；task 4 后续需单列。审计与后续计划见 [500-update 后续计划](libero-500-update-next-step-plan-2026-11-01.md)，原始 artifact 未改写。

## 范围与协议

本项按冻结协议对 7 条 outcome-stratified success transition 采集 paired baseline/jitter 短窗口：两臂共 `14` 个 episode，每条保存前 `20` 个 pre-action step 的 `agentview_rgb`、`eye_in_hand_rgb`、joint/gripper state 和 action。checkpoint 固定为 500-update baseline/jitter，环境 seed `20261022`、stabilization `5`，不评价新的成功率，不读取 private simulator state，不访问 qualification/held-out/final evaluation。

transition state 是由既有闭环结果筛出的机制分层，不能称为 unseen 或正式测试集。该实验新增了短窗口 observation capture，因此它不属于此前的纯只读分析；后续 RGB/action 数值仍只作局部机制诊断。

## Provenance

- 配置：[libero-task4-task7-public-state-jitter-500-transition-rgb-action-diagnostic_v1.json](../configs/libero-task4-task7-public-state-jitter-500-transition-rgb-action-diagnostic_v1.json)，SHA-256 `0de778e7073990b01d0e2e1bc0add193521c70142d4023c37e528d68b3a5a451`。
- runner：[run_libero_task4_task7_public_state_jitter_500_transition_rgb_action_diagnostic.py](../experiments/run_libero_task4_task7_public_state_jitter_500_transition_rgb_action_diagnostic.py)，SHA-256 `a43aa204685681e5feddf8b27e43506c832715aa372c80c3b243e16b7ceb0a4b`。
- capture result：`/tmp/libero-task4-task7-public-state-jitter-500-transition-rgb-action-20261101-v3.json`，SHA-256 `f647ce8642b03f93f92c72c68241038c952c6f07684df266c4b7f2423291e2a3`。
- archive：`/tmp/libero-task4-task7-public-state-jitter-500-transition-rgb-action-archive-20261101-v3`，14 文件，manifest SHA-256 `0c74c53ce4d0e7624d00821425e4ac0628a97c194f7bb33b9959845108c40403`。
- 离线分析：[analyze_libero_task4_task7_public_state_jitter_500_transition_rgb_action_capture.py](../experiments/analyze_libero_task4_task7_public_state_jitter_500_transition_rgb_action_capture.py)。
- 分析结果：`/tmp/libero-task4-task7-public-state-jitter-500-transition-rgb-action-analysis-20261101-v2.json`，SHA-256 `f9f141d15555af40215d0d2e924b7e9735e9497342487a2f19508f8678f75518`。

第一次 capture 因隐藏态 helper import 路径错误 fail-closed，第二次因同一模块路径修复不完整 fail-closed；没有写出结果 JSON，partial 目录保留。成功 v3 的 14 个 archive 均通过 hash 核对。

## RGB/action 对齐结果

| transition | 条数 | agentview RGB MAE | eye-in-hand RGB MAE | captured action delta L2 | baseline Policy RGB-only swap | baseline Policy proprio-only swap |
|---|---:|---:|---:|---:|---:|---:|
| jitter-only | 3 | `0.00333` | `0.01955` | `0.06773` | `0.01976` | `0.00672` |
| baseline-only | 4 | `0.00477` | `0.02586` | `0.13467` | `0.02838` | `0.01131` |

在这 7 对局部窗口中，eye-in-hand RGB MAE 高于 agentview；固定 baseline Policy 的 RGB-only input swap action 变化也高于 proprio-only swap。jitter Policy 的对应均值为：jitter-only RGB-only `0.01618`、proprio-only `0.00608`；baseline-only RGB-only `0.03718`、proprio-only `0.01214`。

captured action delta 同时包含 RGB、proprioception 和 Policy 权重差异；cross-readout 只测固定 Policy 的局部输入敏感性。因此这些数值不能写成相机因果、视觉根因或闭环修复效果。

## 解释边界与下一步

结果支持两个分开的局部描述：新采集窗口中 eye-in-hand 图像像素差高于 agentview；两路相机一起替换引起的动作差高于单独 joint/gripper 替换。当前代码没有分别替换 agentview 与 eye-in-hand，因此尚不能比较两路相机各自对 action 的影响。task 4 的源前缀复现未通过，样本又按源结果分层，不能推广到全部 120 对 episode。

下一步保持 500-update checkpoint 不变，先完善源前缀复现检查，再使用现有 archive 完成同观察跨 Policy 与逐相机离线分解；暂不直接扩采全部 120 对。不要根据这些窗口选择 checkpoint、更新 SPT/Knowledge 或注册 Module。

执行更新：上述源前缀检查与离线分解已完成；task 7 五对构成主要分析，task 4 两对仍单列。主要分析中，同观察 Policy 动作差为 `0.100411–0.111334`，固定 Policy 完整观察替换差为 `0.026819–0.037324`；眼在手/外部相机已分别替换，不能再把“两路相机一起替换”误读为逐相机结果。下一步先冻结 task 4 源前缀复现修复协议，详见 [证据核对与离线分解报告](libero-task4-task7-evidence-decomposition-diagnostic-2026-11-01.md)。

后续复现修复：恢复从 state 0 开始的重置顺序后，新采的四个 task 4 目标窗口全部精确匹配源 reset/action/公开状态；原生抽屉属性采样器配置存在随 reset 累积的行为。新采集保持现有 500-update 权重、另行冻结 56 个短窗口预算，未修改本报告对应的旧 archive 或统计。下一步只读分析修复后的 task 4 数据，详见 [task 4 按原重置顺序复现诊断](libero-task4-ordered-prefix-reproduction-diagnostic-2026-11-01.md)。

## 验证证据

- RGB/action protocol 测试：`6 passed`。
- capture analyzer 专项测试：`2 passed`。
- 相关回归集合：`65 passed`。
- 14 个 archive、cross-readout natural action reproduction、RGB MAE、action delta 和 manifest hash 独立复核通过。
- `py_compile` 与 `git diff --check` 通过。
