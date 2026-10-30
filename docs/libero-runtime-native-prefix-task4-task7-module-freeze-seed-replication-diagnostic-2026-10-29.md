# LIBERO task 4/7 冻结层级 seed 复核（2026-10-29）

## 目的和范围

上一项在 task 4/7、一个训练 seed 上观察到 source-support MSE 随可训练范围收窄而递增。本项用两个新增初始化 seed `20261023/20261024` 重复同一诊断，seed `20261022` 的既有结果作为第三个 paired seed。固定 HDF5 causal source-support 前缀、CNN-only uniform seven-dimensional action MSE、500 次 Adam 更新、两个 observation arm 和四种冻结范围；每个 task/seed/observation arm 内四臂使用相同初始化和 batch 顺序。

本项只评估 source-support prefix 的离线监督误差。没有读取 candidate observations，没有执行策略 rollout，也没有访问 source `meta_query`、validation、held-out、qualification 或 final evaluation。它只检查训练初始化敏感性，不估计 query 泛化。

## 协议与结果来源

- 配置：[module-freeze seed replication](../configs/libero-runtime-native-prefix-task4-task7-module-freeze-seed-replication-diagnostic_v1.json)，SHA-256 `9d72043363d8da6cd40e26f4cda445d9b4c1da99f01879a92f5dee387b46ddb9`。
- 结果：`/tmp/libero-runtime-native-prefix-task4-task7-module-freeze-seed-replication-20261029-v1.json`，SHA-256 `18aa6a45cbc49b1773bdea34fe3ce64661bd445bdded6cf3ca4ca2e60cb5dbc7`。
- 诊断脚本 SHA-256 `c8843e517c04eddf407fff8a78644ba3db98c4576c364306f0d351e08dd61d31`；共享训练器 SHA-256 `780e26d1a50d452609fb24b2a7d4740351f4f044dadc03a286e8523bfbe22e22`。
- 新增 seed 运行 32 个模型，每个 500 updates，耗时约 `110.54 s`；Python `3.10.22`、PyTorch `2.4.1+cu124`、`cuda:1` / NVIDIA GeForce RTX 4090。
- 四个冻结范围的初始化参数 SHA 在每个 seed/observation arm 内一致。
- 结果 SHA-256：`18aa6a45cbc49b1773bdea34fe3ce64661bd445bdded6cf3ca4ca2e60cb5dbc7`；配置 SHA-256：`9d72043363d8da6cd40e26f4cda445d9b4c1da99f01879a92f5dee387b46ddb9`。

## 三 seed source-support MSE

下表每个数先在同 seed 的 historical HDF5 与 runtime-native observation arm 间取均值。seed `20261022` 来自先前的冻结层级诊断，另外两个 seed 来自本次复核。

| task | seed | full Policy | 冻结 visual | 冻结 visual + fusion | 仅 action head |
|---:|---:|---:|---:|---:|---:|
| 4 | 20261022 | 0.02894 | 0.04703 | 0.07046 | 0.21094 |
| 4 | 20261023 | 0.03080 | 0.04276 | 0.07400 | 0.21165 |
| 4 | 20261024 | 0.03198 | 0.04964 | 0.07320 | 0.20701 |
| 7 | 20261022 | 0.03295 | 0.05101 | 0.07588 | 0.18296 |
| 7 | 20261023 | 0.03257 | 0.05256 | 0.08178 | 0.18213 |
| 7 | 20261024 | 0.03421 | 0.07464 | 0.07852 | 0.18164 |

按 HDF5/runtime-native observation arm 分开的每个 seed 结果也都保持严格排序：

| task | seed | observation arm | full Policy | 冻结 visual | 冻结 visual + fusion | 仅 action head |
|---:|---:|---|---:|---:|---:|---:|
| 4 | 20261022 | HDF5 | 0.02656 | 0.04649 | 0.06910 | 0.21104 |
| 4 | 20261022 | runtime-native | 0.03132 | 0.04758 | 0.07181 | 0.21084 |
| 4 | 20261023 | HDF5 | 0.03091 | 0.04194 | 0.07181 | 0.21177 |
| 4 | 20261023 | runtime-native | 0.03069 | 0.04358 | 0.07619 | 0.21154 |
| 4 | 20261024 | HDF5 | 0.03202 | 0.04609 | 0.07229 | 0.20712 |
| 4 | 20261024 | runtime-native | 0.03193 | 0.05318 | 0.07410 | 0.20689 |
| 7 | 20261022 | HDF5 | 0.03300 | 0.05101 | 0.07561 | 0.18288 |
| 7 | 20261022 | runtime-native | 0.03290 | 0.05100 | 0.07616 | 0.18304 |
| 7 | 20261023 | HDF5 | 0.03109 | 0.05417 | 0.08237 | 0.18209 |
| 7 | 20261023 | runtime-native | 0.03405 | 0.05096 | 0.08119 | 0.18217 |
| 7 | 20261024 | HDF5 | 0.03355 | 0.07622 | 0.07957 | 0.18154 |
| 7 | 20261024 | runtime-native | 0.03487 | 0.07305 | 0.07747 | 0.18173 |

四臂严格排序在 `task × seed × observation arm` 的 `12/12` 组都成立。按三个 seed 的 task 均值汇总：

| task | full Policy | 冻结 visual | 冻结 visual + fusion | 仅 action head |
|---:|---:|---:|---:|---:|
| 4 | 0.03057 | 0.04648 | 0.07255 | 0.20987 |
| 7 | 0.03324 | 0.05940 | 0.07873 | 0.18224 |

task 7 的 `freeze_visual` 对初始化更敏感（三个 seed 的 task/arm 均值范围 `0.05101–0.07464`），但每一个 HDF5/runtime-native 臂仍满足 `full < freeze_visual < freeze_visual_fusion < action_head_only`。因此，前一项层级趋势不是单一训练初始化偶然造成的；但它依然只是 source-support 拟合现象。

## 结论边界

1. 冻结范围收窄导致 source-support MSE 上升的排序，在 task 4/7、两个新增 seed 和两种 observation arm 上均复现。
2. 结果支持视觉 encoder 与 fusion/recurrent 可训练路径参与拟合当前 source-support 前缀；它不证明某模块对闭环成功有独立因果贡献。冻结会改变参数数量、梯度路径与优化空间。
3. HDF5 与 runtime-native 两臂上的排序一致，但本项未检验 observation 来源的 query 优势。
4. 所有任务仍只有离线 BC readout 证据，不是正式 SPI qualification、SPT 更新、Knowledge 更新或 Module 资格。

## 下一步

在同样三 seed 和冻结层级上训练 full Policy 与中间冻结臂，只使用 source `meta_support` 拟合，并在与 support demonstration index 完全不相交的 source `meta_query` (`30–49`) 上测量 HDF5 causal pre-action action MSE。source `meta_query` 是预先划分的 source SPI 角色，不是 sealed held-out；报告仍限定为同族示教离线泛化，不延伸到在线适应、跨 SPI 迁移或正式 qualification。
