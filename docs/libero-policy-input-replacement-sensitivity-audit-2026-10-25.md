# LIBERO 固定 Policy 输入替换敏感性审计（2026-10-25）

## 目的与边界

上一项输入对齐审计已经证明：当前 runtime 在执行历史 HDF5 动作时，和 causal BC 数据视图在第一个动作后产生了不同的 RGB 与本体感知输入。本次进一步问一个更直接的问题：如果只把 Policy 看到的 observation 换掉，动作输出会改变多少？

本次按既有分层诊断的同一配置和 seed 重建 task 3、task 7 的 causal Policy，并校验最终参数摘要与既有结果一致。没有读取 validation、held-out 或 qualification，没有执行 Policy 动作，没有选择 checkpoint，没有更新 SPT/Knowledge，也没有注册 Module。历史 HDF5 只读，私有 simulator state 不进入 Policy。

- 配置：[policy input replacement sensitivity audit](../configs/libero_policy_input_replacement_sensitivity_audit_v1.json)，SHA-256 `6d52c3682dad2ec93903c536f3d06556598862275d90afd5e2e0fdee1b6d2915`。
- 结果：`/tmp/libero-policy-input-sensitivity-20261025-v2.json`，SHA-256 `f6b18b4ebacb111b371d3c1dc29d48c90e84d491d5a1b4436a016066f971915b`。
- 代码提交：`0c5b8ea18179ddb24e3c26ce8d0d1b81644c499f`（运行时记录的提交）。
- CUDA：`cuda:1` / RTX 4090；seed `20261021`；causal Policy 训练预算 500 optimizer updates。
- 两个重建 Policy 的最终摘要：task 3 `75c73689043b7930eb62d86caf866388b43a2d15e5e8262b67fadfad1840c9d5`，task 7 `beb3f8e88cb51f37d54a683001adee1b609a2a11be07b58f879804a917d346dc`。二者均与既有 stratified diagnostic 的固定结果一致。

## 比较定义

对每条历史示教的 action step `t`，目标 action 始终是同一个 HDF5 `actions[t]`。

- **历史 causal 输入**：`t=0` 使用当前 runtime 的初始 observation；`t>0` 使用 HDF5 `obs[t-1]`。这和现有 `RoleRestrictedLiberoSequenceDataset` 的 causal view 一致。
- **当前 runtime 输入**：使用同一 HDF5 `init_state` 和同一历史动作前缀，在当前环境中实际生成的动作前 observation。
- **共享历史 hidden**：两种 observation 在每一步使用同一个由历史输入连续更新的 hidden。它近似隔离“当前输入替换”本身对 action head 的即时影响。
- **独立分支 hidden**：历史和当前输入各自连续更新自己的 GRU hidden，测量输入差异经过记忆状态传播后的总影响。

Policy 输出没有送回环境；因此这里比较的是同一 target action 下的输出变化，不是新的闭环成功率。

## 结果

四条轨迹共有 456 个重叠 action step。下表的 action MAE 是当前输入分支输出减历史输入分支输出的绝对误差均值；`target MAE` 是对应分支相对同一个 HDF5 target action 的绝对误差。

| case | steps | 共享历史 hidden：替换 action MAE | 共享 hidden gripper 翻转 | 历史/当前 target MAE | 独立分支 hidden：替换 action MAE | 独立 hidden 当前 target MAE |
|---|---:|---:|---:|---:|---:|---:|
| task 3 failed demo 1 | 101 | 0.02666 | 4/101 | 0.11590 / 0.12142 | 0.05387 | 0.13359 |
| task 3 success demo 0 | 88 | 0.00595 | 0/88 | 0.07981 / 0.08012 | 0.00793 | 0.08075 |
| task 7 failed demo 0 | 155 | 0.02407 | 2/155 | 0.09526 / 0.10158 | 0.03493 | 0.10589 |
| task 7 success demo 1 | 112 | 0.02693 | 0/112 | 0.07001 / 0.07016 | 0.04115 | 0.07715 |

加权汇总如下：

- 共享历史 hidden：替换 action MAE `0.02185`，translation MAE `0.03725`，gripper MAE `0.03182`，gripper 符号翻转 `6/456`；历史/当前 target MAE 为 `0.09065 / 0.09412`。
- 独立分支 hidden：替换 action MAE `0.03544`，translation MAE `0.05734`，gripper MAE `0.06131`，gripper 符号翻转 `12/456`；历史/当前 target MAE 为 `0.09065 / 0.10012`。
- 四条轨迹的第一个非零 action 输出差异都在 step 1，与上一项输入对齐审计发现的首个 observation 分叉位置一致。

## 诊断结论

1. **输入语义分叉会直接改变 Policy 输出。** 在固定 hidden 的比较中，替换历史 observation 已经产生可测的动作变化；因此把历史 HDF5 causal 输入和当前 runtime 输入混在同一项误差中，会把运行时差异传递到 action head。
2. **它是放大因素，不是完整根因。** 两条成功控制示教的 shared-hidden target MAE 几乎不变（task 3 demo 0 为 `0.07981→0.08012`，task 7 demo 1 为 `0.07001→0.07016`），但两条失败示教的 target MAE 均增加约 `0.005–0.006`，且出现 gripper 符号翻转。历史输入分支本身已有较大的动作误差，所以输入对齐不能解释全部失败。
3. **记忆传播会放大输入差异。** 独立 hidden 的替换误差高于共享 hidden（加权 `0.03544` 对 `0.02185`），说明 observation 分叉不仅影响当前 action，也会改变后续 GRU 状态。这个结果支持把“输入语义/运行时一致性”和“Policy 表示与动作 head 学习”分开诊断。
4. **当前结果不能说明哪一种输入协议应直接进入正式训练。** 本次只在四条 source demo 上重放，没有 validation/held-out，也没有比较成功率；它证明的是机制敏感性，不是方法优劣或 SPI/Module 资格。

## 对后续工作的影响

下一步先冻结一个明确的 LIBERO 训练/评估 observation 协议：若目标是当前 runtime 闭环，就需要构造同一 runtime 语义的训练输入；若保留历史 HDF5 视图，则必须把它当作离线行为拟合协议，并单独报告 runtime 对齐误差。协议冻结后，再做同一空间和同一输入语义下的阶段平衡、gripper loss、CNN-only/CNN+GRU 对照；这些对照才能回答 head、覆盖和记忆分别造成多少影响。

本审计仍不更新 SPT/Knowledge，也不产生可注册 Module。它为后续 SPI 实例迁移提供的是数据协议边界和故障归因证据。
