# LIBERO task 4/7 public state jitter augmentation 协议冻结（2026-10-31）

## 目的与状态

基于已完成的 Full Policy state replacement sensitivity 诊断，本项冻结一个单变量 public joint/gripper state jitter augmentation 诊断，用于检查训练时对公开本体状态扰动的鲁棒性。当前协议为 `frozen_protocol_not_run`，training budget 和 episode budget 均为 `0`，没有启动训练。

## 配置与来源

- 配置：[public-state-jitter-augmentation-diagnostic_v1.json](../configs/libero-task4-task7-public-state-jitter-augmentation-diagnostic_v1.json)，SHA-256 `b9f7e1231de342eced5bf53eca3d4f7d194e163b9a58796668078fcdad9b592d`。
- 验证器：[validate_libero_task4_task7_public_state_jitter_augmentation_protocol.py](../experiments/validate_libero_task4_task7_public_state_jitter_augmentation_protocol.py)，SHA-256 `d6a38ac757238a9b6c583309dbe1adf9f56b9981c4dd2ba3ee3c801e55436f12`。
- source query config SHA-256：`9d355ffa017b4b32b87336cecdbd89a33a67a1089cd72e37110043673b78793c`。
- source query result：`/tmp/libero-source-meta-query-generalization-task4-task7-20261030-v1.json`，SHA-256 `091bbb49669d2f0c4196f856ec5f9618e51a08cbc08709dbc844fb747880ac27`。

验证器会检查 source support/query role、causal HDF5 alignment、CNN-only Policy、source result 完整性、candidate/qualification 锁定、optimizer budget 和单变量 augmentation 边界。

## 冻结设计

- task：source SPI task `4, 7`。
- training seeds：`20261022/20261023/20261024`。
- data：`meta_support` demonstrations `0..29` 训练，独立 `meta_query` demonstrations `30..49` 离线评价；不读取 candidate nonregression。
- Policy：CNN-only Full Policy，Adam、learning rate `0.0003`、batch size `8`、500 optimizer updates、gradient clip `1.0`、uniform seven-dimensional action MSE。
- paired arms：`source_baseline` 与 `public_state_jitter`；两臂使用相同初始化和 batch order，不选 checkpoint。
- 唯一训练变量：对 `joint_states` 每元素加入独立对称 uniform jitter `[-0.05, 0.05]`，对 `gripper_states` 加入 `[-0.005, 0.005]`；RGB 和 action target 保持不变，augmentation 只在 training 中开启，query evaluation 使用原始 observation。
- augmentation seed：每个 training seed 加固定 offset `910041`，保证可复现。

## 边界

本项不是正式 SPI 实验，不用于 checkpoint 选择、SPT 更新、Knowledge 更新或 Module qualification。v1 配置仍保留 `training budget=0` 的冻结历史；用户随后明确授权增加预算并启动 v2 训练，实际运行和结果 provenance 见 [public-state jitter training diagnostic](libero-task4-task7-public-state-jitter-augmentation-diagnostic-2026-10-31.md)。评价仍只比较 support fit 与 disjoint meta-query MSE；不能把 query MSE 直接写成闭环成功率或 Module 资格。
随后又按独立授权冻结并执行了 `500/1000/2000` updates 的 baseline/jitter budget sweep；该对照的曲线和 provenance 见 [public-state jitter budget-sweep 诊断](libero-task4-task7-public-state-jitter-budget-sweep-diagnostic-2026-10-31.md)。
