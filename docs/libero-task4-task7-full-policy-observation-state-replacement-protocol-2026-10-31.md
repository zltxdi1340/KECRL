# LIBERO task 4/7 Full Policy observation/state replacement 协议冻结（2026-10-31）

## 目的与当前状态

本项冻结一个未来用于区分 Full Policy 映射差异与公开 joint/gripper observation drift 的短窗口敏感性诊断协议。它是 protocol-only，当前没有执行 rollout、没有训练、没有生成 result artifact，也没有访问 qualification、validation、held-out 或 final evaluation。

当前阶段的新增 episode budget 明确为 `0`。协议冻结不等于授权执行；若未来要运行，必须先显式改变当前预算边界并重新记录 provenance。

## 配置与来源

- 配置：[libero-task4-task7-full-policy-observation-state-replacement-diagnostic_v1.json](../configs/libero-task4-task7-full-policy-observation-state-replacement-diagnostic_v1.json)，SHA-256 `f3cef7aa3b74425ecf8a529132ddeab04293a4104acea8c5beb5b87ee9111c5d`。
- 只读验证器：[validate_libero_task4_task7_full_policy_observation_state_replacement_protocol.py](../experiments/validate_libero_task4_task7_full_policy_observation_state_replacement_protocol.py)，SHA-256 `f3eda2fdf6faafe28a284ed5b2d2fddd25d13d60af8cb196f608ea56500308e9`。
- source closed-loop result：`/tmp/libero-task4-task7-full-policy-closed-loop-20261030-v1.json`，SHA-256 `03439cc0172f3ba29ee4527d3c64d1da16edd93d2eaf63134344e4c8b15f49ce`。
- source closed-loop 配置 SHA-256：`8a485866319a386610f141d81169c073cfc332463d273899ba4762e76c0d2969`。
- source runner commit：`fc79cd65009460c40db889b339eb77317bf80806`。

验证器会逐项检查 source result、6 个 task×training-seed checkpoint 的 file SHA/weight SHA、candidate state `0..19`、环境 seed `20261022`、stabilization `5`、qualification 边界和当前 episode budget。验证输出为 `frozen_protocol_not_run`，current rollout authorized 为 `false`，budget 为 `0`。

## 冻结设计

- 任务：source SPI task `4, 7`。
- training seed：`20261022/20261023/20261024`；每个 task×seed 使用 source Full Policy 闭环结果中已经固定的 checkpoint，不重新训练、不选择 checkpoint。
- 候选初态：`candidate_nonregression 0..19`，这些状态此前已使用，不是 unseen test，也不是 qualification。
- 若未来获准运行：每个 task×seed×state 只采集前 `20` 个 policy pre-action observation，按原协议使用 environment seed `20261022`、stabilization `5` 和同一 adapter 顺序；不执行完整 300-step rollout。
- 采集 key：`agentview_rgb`、`eye_in_hand_rgb`、`joint_states`、`gripper_states`；不采集或使用 private simulator state。

离线比较只在固定模型输出层进行：

1. **natural**：使用记录的自由运行 pre-action observation；
2. **public-state reset replacement**：保持两路 RGB 不变，将 `joint_states` 和 `gripper_states` 替换为该 episode post-stabilization reset 的公开值；替换后的 action 不送回环境。

同时记录 shared natural hidden 和 independent replacement hidden 两种模式，汇总 action/translation/rotation/gripper 差异、首次 close 和 gripper sign switch。该 replacement 测量的是固定 Policy 的输入敏感性，不是动力学辨识、专家 action error、目标相对几何或因果干预。

## 当前边界

本协议没有 result、没有成功率、没有 checkpoint 选择，也不改变 SPT、Knowledge 或 Module 状态。此前 2026-10-28 的 runtime-native prefix observation archive 属于不同 Policy，禁止混入本协议。只有在未来明确增加预算并重新冻结执行 provenance 后，才能运行该 protocol；在此之前保持当前分析边界。
