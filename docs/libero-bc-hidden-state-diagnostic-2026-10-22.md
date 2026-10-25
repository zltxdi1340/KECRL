# LIBERO 隐藏态时域诊断（2026-10-22）

**状态：v1/v2 单 seed 诊断完成。** 使用固定的 same-index 与 causal pre-action Policy 权重检查 GRU 隐藏态处理对离线误差和闭环行为的影响。本记录不是模型选择、正式架构结论或 Module 资格报告。

## 目的和固定边界

因果对齐诊断显示两种数据视图都没有稳定闭环。候选 Policy 以长度 10 的独立序列窗口训练，窗口 hidden state 从零开始；原始闭环 evaluation 则连续携带 GRU state。本轮在同一组训练后权重上比较 continuous、每 10 步重置、stateless 三种推理方式，并在完整专家轨迹上对同一因果视图比较连续 teacher-forced hidden 与每窗重置 hidden。

- task 0，训练 seed `20261021`；source `meta_support` demonstrations 0--29、`meta_query` 30--49；官方 candidate 初态 0/1/2。
- 两个训练 arm、每臂 Adam 500 updates；此 follow-up 重训后强制核对 final Policy SHA-256，确保与先前因果对齐 v1 的两组权重完全一致。
- 两臂各 `37,534` 个有效 action-step，初始 Policy hash、batch 顺序 seed、学习率、优化器、batch 和数据窗口均匹配。
- support/query 专家回放仍各 `3/3`。validation、held-out、qualification 不访问，SPT 不变。
- 配置：[hidden-state diagnostic](../configs/libero_spatial_bc_hidden_state_diagnostic_v1.json)，SHA-256 `dd85f45d9ea9722a8a2c67a6d247a3c25b8897e1defefc977a1e9f042647fd97`。
- runner：[run_libero_bc_causal_alignment_diagnostic.py](../experiments/run_libero_bc_causal_alignment_diagnostic.py)。
- 代码提交 `20155113a6f067bd8e8630cc755f4017a937969c`；CUDA `cuda:1` / RTX 4090。
- 结果文件 `/tmp/libero-bc-hidden-state-20261022-v2.json`，SHA-256 `8402879db7d6ea6b4ac3598f9622d9c6eab5f0efce5ef0c87f08a19fdb1b918f`。
- 运行耗时 `414.79s`，其中训练及 Policy evaluation `352.48s`，峰值 CUDA allocation `292,003,840 bytes`。

## 逐步动作偏差与自由轨迹状态漂移（已完成）

为避免把“GRU 长记忆”当成直接解释，本轮用固定的 causal Policy 权重，在开放的 support/query demonstration 0/1/2 与 30/31/32 上配对记录：

- 沿专家执行轨迹逐步向 Policy 提供 pre-action observation，记录 teacher-forced Policy action 与专家 action 的逐维误差及首个 gripper sign mismatch；
- 从相同 `init_state` 重新开始，让 Policy 自由执行，再与专家实际成功步数范围内的 joint/gripper 状态、agentview 和 eye-in-hand RGB 做 10 步分箱比较；
- 不触及 official candidate、validation、held-out 或 qualification 初态，不训练、不选模型。

配置 SHA-256 为 `e9170ba4f10c48d50c895577d121f45caf9952318e86f229cd58b9df173a84bc`；运行时重新训练的 causal Policy 与锁定权重 SHA-256 完全一致。CUDA `cuda:1` / RTX 4090，耗时 `462.51s`。修正后的结果文件 `/tmp/libero-bc-action-divergence-20261022-v2.json`，SHA-256 `cb2b207d97049076df048fe7e89749b183f58311da2bdb55d123a9c687b76074`；代码使用提交 `34a2f3d1b08e6fc44d932dbd52476cff248c5e98`，该版本按专家实际成功步数截断比较。

六条专家轨迹均成功。HDF5 中存储的动作条数会超过成功终止时实际执行的步数，因此本表只按真实执行的专家步数比较；每个 Policy rollout 均运行至 300 步上限且未成功。

| 角色 / demo | 专家实际步数（存储动作数） | Policy 步数 | 重叠步数 | teacher-forced 动作 MAE | 夹爪符号准确率 / 首次分歧步 | 自由 rollout 关节状态 L2 漂移 | agentview / eye-in-hand RGB MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| support / 0 | 82（98） | 300 | 82 | 0.112 | 0.988 / 35 | 0.379 | 5.60 / 27.70 |
| support / 1 | 73（84） | 300 | 73 | 0.114 | 0.959 / 32 | 0.345 | 6.32 / 28.39 |
| support / 2 | 79（90） | 300 | 79 | 0.125 | 0.975 / 37 | 0.345 | 5.51 / 27.79 |
| query / 30 | 114（127） | 300 | 114 | 0.172 | 0.833 / 32 | 0.322 | 6.40 / 30.17 |
| query / 31 | 69（151） | 300 | 69 | 0.111 | 0.942 / 37 | 0.280 | 5.69 / 23.10 |
| query / 32 | 82（93） | 300 | 82 | 0.123 | 0.988 / 40 | 0.356 | 6.35 / 26.31 |

按 499 个实际重叠动作步加权汇总，七维动作 MAE 为 `0.130`；平移维误差更大（x `0.243`、y `0.133`、z `0.207`），三个旋转维约 `0.038`--`0.043`，夹爪为 `0.203`。整体夹爪符号准确率 `0.940`，但六条 demo 的首次符号分歧都出现在第 `32`--`40` 步。

| 专家时间段 | 覆盖步数 | 动作 MAE | 关节状态 L2 漂移 | 夹爪状态 L2 漂移 | agentview / eye-in-hand RGB MAE | 夹爪符号准确率 |
|---|---:|---:|---:|---:|---:|---:|
| 0--10 | 60 | 0.069 | 0.038 | <0.001 | 0.72 / 8.62 | 1.000 |
| 10--20 | 60 | 0.076 | 0.098 | <0.001 | 2.42 / 13.97 | 1.000 |
| 20--30 | 60 | 0.108 | 0.165 | <0.001 | 5.63 / 17.14 | 1.000 |
| 30--40 | 60 | 0.181 | 0.243 | 0.003 | 5.85 / 26.84 | 0.800 |
| 50--60 | 60 | 0.144 | 0.519 | 0.051 | 8.47 / 38.20 | 0.917 |

**诊断判断。** 关节与图像偏差在夹爪首次符号分歧前已经增加；最初 10 步可见非零平移动作误差和小幅状态偏移，20--30 步仍无夹爪符号错误，但关节漂移已增至 `0.165`。这排除了“夹爪首次分歧是所有轨迹偏移的起点”这一简单解释，支持“连续平移动作误差先把自由执行带离示教状态，后续偏移继续累积”的工作假设。它仍是观测相关性：RGB、动作预测、控制响应和状态分布在本诊断中没有被独立干预，不能据此判定视觉网络或动作头就是根因。尤其 eye-in-hand RGB 的误差是像素强度尺度上的 MAE，不能直接解释成空间误差。

局部单步反事实诊断的协议和结果见下节。它固定当前 causal Policy 和六个已开放 demo，在专家轨迹的 `0/10/20/30/40/50` 步及 v2 中观察到的首次夹爪分歧处，从相同专家前缀重放到同一 pre-action 状态，再分别执行专家动作和 Policy 预测动作。两条独立重放的 pre-action joint、gripper 和 RGB 必须完全一致，否则 fail closed。
## 局部单步反事实结果（2026-10-23）

该检查已完成。它只重建锁定的 causal Policy，不进行 checkpoint 选择，不访问 official candidate、validation、held-out 或 qualification 状态。

- 配置：[one-step counterfactual diagnostic](../configs/libero_spatial_bc_one_step_counterfactual_diagnostic_v1.json)，SHA-256 `6912496afe6f6e95f2cc60c5ca76096b9032980de40edec5db346fc2e1bd372b`。
- 结果：`/tmp/libero-bc-one-step-counterfactual-20261023-v1.json`，SHA-256 `949dd2fd9d54d6c53c4a045abcc0fc596f70b4f5fedc9466155a58ac43ba3f75`。
- 代码提交：`779cf67098d1f6c4e43afaa6a4a2411d795cdc70`；CUDA `cuda:1` / RTX 4090；耗时 `237.17s`。
- 两个训练 arm 仍各执行 500 updates，均消费 `37,534` 个有效动作步；causal arm 的最终 Policy SHA-256 与锁定值一致。
- 六个 support/query demonstration 共 41 个目标位置。每个位置分别重放专家动作和 Policy 动作；两次重放的 pre-action joint、gripper、agentview RGB、eye-in-hand RGB 差异全部为 `0`，说明分叉发生在同一可复现观测状态。

所有目标位置的汇总如下：

| 指标 | 平均 | 最大 |
|---|---:|---:|
| 七维动作 MAE | 0.1319 | 0.3398 |
| 动作 L2 | 0.5522 | 1.9194 |
| 单步后 joint 状态间隙 L2 | 0.00674 | 0.01678 |
| 单步后 gripper 状态间隙 L2 | 0.000102 | 0.000527 |
| 下一步 agentview RGB MAE | 0.505 | 1.502 |
| 下一步 eye-in-hand RGB MAE | 4.092 | 9.570 |

按目标时间段汇总，`0--20` 步的动作 MAE / joint 间隙 / eye-in-hand RGB MAE 为 `0.078 / 0.00439 / 2.32`；`20--30` 步为 `0.117 / 0.00617 / 3.14`；`30--40` 步为 `0.189 / 0.00761 / 5.62`；`40--60` 步为 `0.140 / 0.00859 / 4.94`。因此中段动作偏差和单步视觉变化更大，但并非所有指标随时间单调增加。

六条轨迹的首次夹爪符号分歧处均有可见夹爪即时响应：夹爪状态间隙约 `0.00049--0.00053`。然而这些单步 joint 间隙仍明显小于前一轮自由 rollout 在专家时域上的平均关节漂移 `0.338`，单步 eye-in-hand RGB 间隙也小于自由 rollout 的平均 `27.51`。这说明环境对单个动作差异的即时响应是可测的，但不能单独解释长程失败；长程失败仍需要考虑连续动作误差、视觉观测分布变化和 Policy 离开示教状态后的闭环反馈。

**当前判断。** 已排除“前缀重放不一致”这一诊断污染源，也确认夹爪分歧在其发生位置会改变夹爪状态。当前证据支持“早期平移动作误差与后续闭环状态偏移共同作用”的工作模型，不支持先把根因归结为 GRU 或单独的夹爪 head。

## 状态分布偏移追踪结果（2026-10-23）

该诊断已经完成。它重训两臂以复现锁定权重，然后只在 causal pre-action 臂的六条已开放 support/query demonstration 上做受控分叉；不访问 official candidate、validation、held-out 或 qualification，不选择 checkpoint，不更新 SPT/Knowledge，也不注册 Module。

- 配置：[state-distribution-shift diagnostic](../configs/libero_spatial_bc_state_distribution_shift_diagnostic_v1.json)，SHA-256 `53f07caf1043fe170d6679d1f052c5a5e9f6b113d0e5fda9f62247bc7207d489`。
- 结果文件：`/tmp/libero-bc-state-distribution-shift-20261023-v1.json`，SHA-256 `e36fdfb78add061dcac6c9c3c8deee2dfd0e6efb14a22007e66dbd607ac1c6e6`。
- 运行代码提交：`7fae662385195a21053fff6541acc532857eb04b`；CUDA `cuda:1` / RTX 4090；总耗时 `255.05s`，峰值 CUDA allocation `292,003,840 bytes`。
- 两个训练 arm 均执行 500 updates、消费 `37,534` 个有效动作步，最终 Policy SHA-256 与配置中的锁定值一致。六条 demonstration 的 36 个目标前缀在两套独立环境中的 joint、gripper、agentview RGB、eye-in-hand RGB 和 hidden 均完全复现；1/3/5 步窗口没有提前终止。

在每个目标前缀，先比较专家状态上的 Policy 输出与“自由状态、沿用专家 hidden”的输出，再比较自由状态和自由 hidden 的输出。数值为所有目标、所有相对步的均值：

| 自由窗口 | 样本数 | 状态条件动作 MAE | hidden 条件动作 MAE | 合计动作 MAE | 窗口前 joint 间隙 L2 | 窗口前 hidden 间隙 L2 | 窗口后 joint 间隙 L2 | 窗口后 eye-in-hand RGB MAE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 36 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.006879 | 3.932 |
| 3 | 108 | 0.002463 | 0.000595 | 0.002847 | 0.009100 | 0.015564 | 0.021301 | 6.941 |
| 5 | 180 | 0.004928 | 0.001997 | 0.006489 | 0.023534 | 0.061219 | 0.037723 | 9.193 |

在这个单 seed、短窗口诊断中，状态条件项大于 hidden 条件项：3 步窗口的动作 MAE 约为 `0.002463` 对 `0.000595`，5 步窗口约为 `0.004928` 对 `0.001997`。hidden 项并非零，并且随窗口变长而增加。因此目前更符合“Policy 离开示教状态后，观测状态漂移是早期输出变化的主要可见来源，hidden 演化提供次级贡献”的描述；这不是视觉网络、GRU 或动作 head 的因果归因，也不能外推为正式 SPI 性能结论。1 步窗口的两项均为零是预期现象，因为分叉前两臂的状态和 hidden 被要求完全相同。

## RGB 与本体感知模态分解结果（2026-10-23）

在上一节相同的专家/自由分叉上继续做输入替换：固定专家 hidden，分别替换 RGB、本体感知，或同时替换两类输入；再保留自由 hidden 的完整自由输入作为 recurrent 项。该结果仍只使用六条开放 support/query demonstration 的 36 个目标前缀，不访问 official candidate、validation、held-out 或 qualification。

- 配置：[state-modality-shift diagnostic](../configs/libero_spatial_bc_state_modality_shift_diagnostic_v1.json)，SHA-256 `b5b5b90c434558827277ecfc3b7155f5da014af122edc571903e76c0346a078c`。
- 结果文件：`/tmp/libero-bc-state-modality-shift-20261023-v1.json`，SHA-256 `1f994ce55561019da9215f681241fabe7ccab7dc34949385caf6c5014072127b`。
- 运行代码提交：`ff7a8f77bfd906c1720768a306ad605cdada7dcc`；CUDA `cuda:1` / RTX 4090；总耗时 `251.86s`，峰值 CUDA allocation `292,003,840 bytes`。两臂仍各 500 updates、各消费 `37,534` 个有效动作步，causal Policy 最终哈希为锁定值 `6f647466576f1e763ddbc06f8885e4c2f900e0d2635036161bdff991ec5aa3ae`。

下面的 MAE 是各目标、各相对步、七维动作的平均绝对偏移；RGB 与本体感知项都保持另一类输入为专家值并保持专家 hidden，hidden 项是完整自由输入与固定专家 hidden 输入的差：

| 自由窗口 | 样本数 | RGB 条件动作 MAE | 本体感知条件动作 MAE | 两类输入联合 MAE | RGB/本体交互 MAE | hidden 条件动作 MAE |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 36 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 3 | 108 | 0.002555 | 0.000442 | 0.002463 | 0.000013 | 0.000595 |
| 5 | 180 | 0.005063 | 0.001098 | 0.004928 | 0.000084 | 0.001997 |

在这组受控输入替换中，RGB 项约为本体感知项的 `5.8` 倍（3 步）和 `4.6` 倍（5 步）；hidden 项小于 RGB、但大于本体感知项；交互项接近零。这支持“短窗口动作输出变化主要由视觉观测漂移驱动，循环 hidden 提供次级贡献，本体感知漂移较小”的局部判断。它仍不能说明视觉编码器本身、相机视角、动作 head 或数据覆盖谁是最终根因，因为输入替换测量的是锁定 Policy 在当前轨迹附近的敏感性，不是修复后的长程成功率。

## agentview 与 eye-in-hand 相机分解结果（2026-10-24）

在相同的六条 demonstration、相同的专家 hidden 和专家本体感知输入下，进一步只替换一路 RGB，或同时替换两路 RGB。这里的相机项不混入本体感知漂移；hidden 项仍来自完整自由输入与固定专家 hidden 输入的差异。

- 配置：[state-camera-shift diagnostic](../configs/libero_spatial_bc_state_camera_shift_diagnostic_v1.json)，SHA-256 `ad189b22f40f26bbfc72f2d8a15e07e23429790a3e71c4117a49d4b09b965c15`。
- 结果文件：`/tmp/libero-bc-state-camera-shift-20261024-v1.json`，SHA-256 `32a4b8641a0e47faea1e4298b0ea85209808735f9b715cfeab2d0e90c3ae07b3`。
- 运行代码提交：`a4459beaba5c3487c26ccb57cf96c231a1456ef4`；CUDA `cuda:1` / RTX 4090；总耗时 `146.67s`，峰值 CUDA allocation `292,003,840 bytes`。两臂各 500 updates、各消费 `37,534` 个有效动作步，causal Policy 最终哈希仍为锁定值 `6f647466576f1e763ddbc06f8885e4c2f900e0d2635036161bdff991ec5aa3ae`。

| 自由窗口 | 样本数 | agentview 条件动作 MAE | eye-in-hand 条件动作 MAE | 两路相机联合 MAE | 相机交互 MAE | hidden 条件动作 MAE |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 36 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 3 | 108 | 0.000208 | 0.002530 | 0.002555 | 0.000010 | 0.000595 |
| 5 | 180 | 0.000359 | 0.005061 | 0.005063 | 0.000028 | 0.001997 |

在这个单 seed、短窗口诊断中，eye-in-hand 项约为 agentview 项的 `12.1` 倍（3 步）和 `14.1` 倍（5 步），两路交互接近零。它支持“当前自由执行附近的 RGB 敏感性主要来自末端相机输入”的局部判断；它仍不能证明 eye-in-hand 编码器或相机本身是长程失败的唯一根因。下一步先在不同训练 seed 上复核这个排序，再决定是否进行单变量视觉修复 pilot。

## 相机贡献的多 seed 复核结果（2026-10-24）

为检查相机贡献排序是否依赖单个初始化，保持同一 task、support/query demonstration、500-update 预算、causal Policy、1/3/5 步窗口和正式评估隔离，使用 `20261021`、`20261022`、`20261023` 三个 replicate seed 重新训练并锁定两臂最终 Policy 哈希。每个 seed 的 36 个目标前缀均通过状态/hidden 精确复现检查，官方 candidate、validation、held-out 和 qualification 均未访问。

| replicate seed | 3 步 agentview | 3 步 eye-in-hand | 3 步 hidden | 5 步 agentview | 5 步 eye-in-hand | 5 步 hidden |
|---:|---:|---:|---:|---:|---:|---:|
| 20261021 | 0.000208 | 0.002530 | 0.000595 | 0.000359 | 0.005061 | 0.001997 |
| 20261022 | 0.000738 | 0.003013 | 0.000421 | 0.001555 | 0.006102 | 0.001715 |
| 20261023 | 0.000291 | 0.003529 | 0.000381 | 0.000575 | 0.008513 | 0.002148 |
| 均值 ± 样本标准差 | 0.000412 ± 0.000285 | 0.003024 ± 0.000500 | 0.000465 ± 0.000114 | 0.000830 ± 0.000637 | 0.006559 ± 0.001771 | 0.001953 ± 0.000220 |

三个 seed 中 eye-in-hand 的条件动作偏移均大于 agentview：3 步比例分别约为 `12.2`、`4.1`、`12.1`，5 步比例约为 `14.1`、`3.9`、`14.8`。seed `20261022` 的 agentview 偏移相对更大，但没有改变排序。相机交互项的跨 seed 均值为 3 步 `0.000022`、5 步 `0.000093`，仍明显小于 eye-in-hand 项。

三份复核结果文件及配置哈希如下：

- seed `20261021`：结果 SHA-256 `32a4b8641a0e47faea1e4298b0ea85209808735f9b715cfeab2d0e90c3ae07b3`，配置 SHA-256 `ad189b22f40f26bbfc72f2d8a15e07e23429790a3e71c4117a49d4b09b965c15`。
- seed `20261022`：结果 SHA-256 `185fb810a203db9a6e78efc969d07b0d58f1d0c2c7f72703cd691de0b7efa83e`，配置 SHA-256 `0450205f0d4d83de2df25658f247f9fba286b2cb61a45e89c2e8e652bcf69c47`。
- seed `20261023`：结果 SHA-256 `9600356c4ad522f15c74e3b1b52529abee66cb01771085ebe3e326e531e2846f`，配置 SHA-256 `bae66d52fe36b698288264127f0689694d3826f6f53ac7ca03049466e1c7739d`。

这使“eye-in-hand 是当前局部 RGB 敏感性的主要来源”从单 seed 工作假设提升为三 seed 机制诊断结论，但仍不是视觉修复有效性、长程成功率或正式 SPI 资格结论。

## eye-in-hand 平移鲁棒性 pilot（2026-10-24）

为检验相机敏感性是否可以通过单变量输入处理缓解，固定 task 0、causal pre-action 数据、meta-support 训练角色、500 次 optimizer update、连续 hidden 部署和 support/query 各 3 条开放 demonstration，只在实验臂的训练阶段对 eye-in-hand RGB 序列施加每个窗口共享的整数平移（反射填充，最大 4 像素）。baseline 不做增强；两臂共享初始 Policy、batch 顺序、有效动作步预算和评估边界。增强只作用于训练，评估使用原始图像。

- 配置：[eye-hand robustness pilot](../configs/libero_spatial_eye_hand_robustness_pilot_v1.json)，SHA-256 `be761489bb8d52e02137fdf9d8559410fd176953e6a66529b94cd17e198b1c36`。
- 结果文件：`/tmp/libero-eye-hand-robustness-pilot-20261024-v1.json`，SHA-256 `144a61b132b0e7b111c524eb3468816f5bdd7790ea449abb0317989c6e663e00`。
- 运行提交：`3554a50336a9bc096903fcd439ab17b4c6c4dc88`；CUDA `cuda:1` / RTX 4090；耗时 `1026.27s`。三个 seed 的 baseline 最终 Policy 哈希均与既有锁定值一致；两臂各消费 `37,534`、`37,547`、`37,537` 个有效动作步。

| seed | query MSE baseline → 增强 | support 成功 baseline → 增强 | query 成功 baseline → 增强 | eye-in-hand 敏感性 3 步 baseline → 增强 | 5 步 baseline → 增强 |
|---:|---:|---:|---:|---:|---:|
| 20261021 | 0.09682 → 0.09373 | 0/3 → 2/3 | 0/3 → 3/3 | 0.002530 → 0.002862 | 0.005061 → 0.004916 |
| 20261022 | 0.08242 → 0.09006 | 0/3 → 1/3 | 0/3 → 0/3 | 0.003013 → 0.002404 | 0.006102 → 0.006264 |
| 20261023 | 0.08832 → 0.08657 | 0/3 → 0/3 | 0/3 → 0/3 | 0.003529 → 0.003342 | 0.008513 → 0.006620 |
| 均值 | 0.08919 → 0.09012 | 0/9 → 3/9 | 0/9 → 3/9 | 0.003024 → 0.002869 | 0.006559 → 0.005933 |

增强臂的 query MSE 只有两个 seed 下降，闭环新增的 3 个 query 成功全部来自 seed `20261021`；相机敏感性均值下降，但 seed `20261021` 的 3 步和 seed `20261022` 的 5 步反而上升。因而本 pilot 提供了视觉输入处理可能改善局部行为的信号，却没有稳定的跨 seed 修复证据。它仍是非正式 pilot：没有访问 official candidate、validation、held-out 或 qualification 状态，没有选择 checkpoint，没有更新 SPT/Knowledge，也没有注册 Module。

下一项先不扩大到 A/B/C。固定这三组已训练 Policy 和同一批 expert-state，在原始 eye-in-hand 图像上施加可复现的 `±1/±2/±4` 像素平移，直接测量动作变化及 gripper/translation 分量变化；该测试用于判断训练增强是否真正降低相机输入敏感性。若没有稳定降低，再转向 action head/数据覆盖的诊断；若降低且闭环信号可复现，才冻结更大 seed 的视觉修复 pilot。

## 固定 expert-state 的合成相机敏感性结果（2026-10-24）

为把“增强后闭环偶尔成功”与“增强确实改变了相机敏感性”分开，固定三 seed 的 baseline 和 eye-in-hand 平移增强 Policy，在同一批专家重放状态上逐帧计算动作。每个状态只替换 eye-in-hand 图像，agentview、joint、gripper 和 baseline 的连续 hidden 保持不变；分别施加左右/上下方向的 1、2、4 像素反射填充平移。每个 seed 的 support/query 共捕获 `234 + 265 = 499` 个专家 action 前状态。

- 配置：[synthetic eye-hand sensitivity diagnostic](../configs/libero_spatial_eye_hand_synthetic_sensitivity_diagnostic_v1.json)，SHA-256 `735ce9b33d4d97c14fe640875c162d7539d2502afdf5df0ec59c263a2e669ace`。
- 结果文件：`/tmp/libero-eye-hand-synthetic-sensitivity-20261024-v1.json`，SHA-256 `02234bfb5712172abb9c3847b8d4cbb068b301dacb4e1087fa56a9b1de1a357`。
- 运行提交：`967d1c00ef3910777778970f4bebfb7a65718706`；CUDA `cuda:1` / RTX 4090；耗时 `329.73s`。

下面是四个方向聚合后、三个 seed 的均值；括号内为增强臂减 baseline，负值表示敏感性下降。

| 平移半径 | action MAE baseline → 增强 | translation MAE baseline → 增强 | gripper MAE baseline → 增强 | gripper sign flip baseline → 增强 |
|---:|---:|---:|---:|---:|
| 1 px | 0.009134 → 0.002892 (`-0.006243`) | 0.006640 → 0.002887 (`-0.003753`) | 0.038940 → 0.009883 (`-0.029057`) | 0.014362 → 0.004509 (`-0.009853`) |
| 2 px | 0.019944 → 0.004796 (`-0.015147`) | 0.013042 → 0.004785 (`-0.008257`) | 0.090068 → 0.016369 (`-0.073699`) | 0.036072 → 0.007849 (`-0.028223`) |
| 4 px | 0.024397 → 0.008782 (`-0.015616`) | 0.016495 → 0.008573 (`-0.007921`) | 0.108311 → 0.030340 (`-0.077971`) | 0.043754 → 0.016700 (`-0.027054`) |

在四方向聚合、三个 seed 和三个半径上，上述整体指标的差值均为负；上下方向的下降通常大于左右方向，左右方向在半径 1 px 的个别 seed 接近零或略升。这个结果支持“训练增强改变了 eye-in-hand 输入到动作 head 的局部敏感性”，而不是只改变了某一条开放 rollout。它同时显示增强对 gripper 输出的敏感性降低尤其明显。

这仍不能推出长程任务成功。前一 pilot 的增强臂 query 成功为 `3/9`，且全部来自一个 seed；query MSE 三 seed 均值从 `0.08919` 变为 `0.09012`。因此当前解释是：相机局部敏感性已被有效缓解，但专家动作拟合、早期动作误差和离开示教分布后的状态反馈仍未解决。该诊断没有访问 official candidate、validation、held-out 或 qualification 状态，没有选择 checkpoint，没有更新 SPT/Knowledge，也没有注册 Module。

下一项比较两臂在原始 expert-state 上的动作误差时间曲线和动作分量（translation/rotation/gripper）首次偏差位置，并与受控 free-run 的首个状态偏移对齐。若增强只降低扰动敏感性而不降低原始动作误差，问题应转向数据覆盖或 action head；若原始动作误差也下降但 free-run 仍失败，重点转向状态分布和闭环反馈。

## 原始 expert-state 动作误差与 free-run 首次偏移对齐（2026-10-24）

这项诊断重新训练并锁定三 seed 的 baseline 与平移增强 Policy，在相同的六条开放 demonstration 上同时记录：

1. expert-state 输入下的 teacher-forced 动作误差，按每 10 步报告整体、translation、rotation 和 gripper 分量；
2. free-run 分支相对专家轨迹的 joint、gripper 和 eye-in-hand RGB 状态差异；
3. 固定状态阈值的首次越界位置，以及 support/query 开放 rollout 成功数。

- 配置：[expert error/free alignment diagnostic](../configs/libero_expert_error_free_alignment_diagnostic_v1.json)，SHA-256 `8845b51f5c7437794648ab06b3e7cdb6beff49540795cf164f82917b3c04e08c`。
- 结果文件：`/tmp/libero-expert-error-free-alignment-20261024-v1.json`，SHA-256 `4901dc0783cd5d4a91816c97a617f75548d48fff36a4571d797444c1f9f20098`。
- 运行提交：`434569fcc22ed73143c6007b268ee1c53ac89025`；CUDA `cuda:1` / RTX 4090；耗时 `453.92s`。三 seed 两臂最终 Policy 哈希全部通过配置锁定值校验。

三 seed、四方向 support/query 的平均时间段结果如下：

| 角色 | Policy | 0–10 步 action MAE | 0–10 步 translation MAE | 10–20 步 action MAE | joint-state 首次越界中位数 | gripper-state 首次越界中位数 |
|---|---|---:|---:|---:|---:|---:|
| support | baseline | 0.0748 | 0.1306 | 0.0895 | 2 | 38 |
| support | eye-hand translation | 0.0614 | 0.1056 | 0.0763 | 2 | 38 |
| query | baseline | 0.0755 | 0.1406 | 0.0692 | 2 | 39 |
| query | eye-hand translation | 0.0574 | 0.1024 | 0.0496 | 2 | 39 |

增强臂在前 20 步的原始 expert-state 动作误差通常更低，但 joint-state 的首次越界仍在第 2–3 步；因此减小相机输入敏感性没有消除最早的状态偏移。随后 baseline 的 joint drift 在 support 的 30–60 步平均约 `0.61–0.73`，增强臂约 `0.38–0.43`；query 对应 baseline 约 `0.55–0.65`，增强臂约 `0.32–0.46`，说明增强可能减缓部分反馈漂移，但并未保证闭环成功。三 seed 九条 support 和九条 query 的 baseline rollout 均为 `0/9`；增强臂为 support `3/9`、query `3/9`，与前一 pilot 一致，新增成功仍集中在 seed `20261021`。

配置中的 `action_l2=0.05` 阈值在所有 episode 的第 0 步都越界，因此不能用它判定“动作误差何时首次产生”；报告中的时序判断采用原始动作误差曲线和 joint/gripper/RGB 状态阈值。当前证据链为：

- **动作拟合缺口先于状态漂移出现。** 第 0–10 步已经有约 `0.06–0.08` 的 action MAE 和约 `0.10–0.14` 的 translation MAE，尚未经过长程 free-run 反馈；
- **相机增强能改变映射但不足以修复闭环。** 它降低了前 20 步误差和固定 expert-state 的合成相机敏感性，但首次 joint 越界仍在第 2–3 步；
- **后段是反馈放大。** joint drift 先扩大，gripper-state 通常在第 34–42 步越界，之后 action/gripper 分量误差明显上升。

因此下一项转向“动作 head/数据覆盖”诊断：在不访问新评估状态的前提下，对同一 causal support/query 数据做按时间与动作分量的条件覆盖审计，并比较冻结视觉/GRU、只训练 action head 与全 Policy 训练的 expert-state 误差下界。若只训练 head 已接近全 Policy，瓶颈主要是 head/监督信号；若 head 下界仍高，需检查观测到动作的多模态、数据覆盖和 task relation 条件。该诊断仍不选择 checkpoint、不更新 SPT/Knowledge、不注册 Module。

## action head 与表示学习分解结果（2026-10-24）

为检验动作误差是否主要来自最后的动作 head，固定 causal pre-action support/query 数据、三组 replicate seed、同一初始 Policy、同一 batch 顺序和 500 次 optimizer update，比较两臂：`full_policy` 全部训练；`action_head_only` 冻结 `visual_encoder`、`fusion` 和 `recurrent`，只训练 1,799 个 action-head 参数。两臂只访问 meta-support 训练和 meta-query 离线评估，没有访问新的 online、official candidate、validation、held-out 或 qualification 状态。

- 配置：[action-head representation diagnostic](../configs/libero_action_head_representation_diagnostic_v1.json)，SHA-256 `a083aa85a1a7d829526f807941af83bf2415033b2510ccf999b402687800af31`。
- 结果文件：`/tmp/libero-action-head-representation-20261024-v1.json`，SHA-256 `e8e2ded5d7595c870c4b5c3507b2f68b95a435a2e1b60757e19b50d40572526e`。
- 运行提交：`ab412e397d90acc7d23c3030efe9da97bbbbdbf7`；CUDA `cuda:1` / RTX 4090；耗时 `184.27s`。full-policy 三个最终哈希均与已有 causal baseline 锁定值一致。

| 角色 | Policy | MSE 均值 | translation MSE 均值 | rotation MSE 均值 | gripper MSE 均值 | gripper sign accuracy 均值 |
|---|---|---:|---:|---:|---:|---:|
| support | full policy | 0.07597 | 0.09424 | 0.00361 | 0.23827 | 0.9201 |
| support | action head only | 0.22631 | 0.20847 | 0.00381 | 0.94731 | 0.6057 |
| query | full policy | 0.08919 | 0.10105 | 0.00376 | 0.30986 | 0.8947 |
| query | action head only | 0.22818 | 0.21094 | 0.00373 | 0.95328 | 0.5726 |

head-only 与 full-policy 的 query 差值为 MSE `+0.13900`、translation MSE `+0.10989`、gripper MSE `+0.64341`，rotation MSE 接近零。固定随机表示后，head 可以拟合一部分旋转输出，却无法恢复平移和夹爪行为；当前失败不能由最后的线性 head 单独解释，视觉/融合/时序表示学习是必要条件。

同一 causal 数据的动作覆盖审计显示，support/query 前 60 步均有完整有效动作样本，总有效步分别为 `2,925/2,143`；两者前段平移、旋转和 gripper 绝对动作分布同量级，没有发现 query 完全缺失某个早期阶段的简单覆盖断裂。这不等于条件覆盖充分，图像目标位置、task relation 和动作之间的多模态仍未被分离。

条件覆盖审计已完成，结果见 [LIBERO 条件覆盖审计](libero-conditional-coverage-audit-2026-10-24.md)。五个 source-SPI 任务的 support/query 覆盖相同的四种 relation、四种 source geometry 和五个 relation×geometry 组合；三个动作阶段均有样本，但阶段比例仍有任务条件偏移。目标相对几何没有越过 Policy 边界被读取，因此覆盖审计不能解释 RGB 是否遗漏了目标关系。随后完成的 [分层误差诊断](libero-stratified-error-diagnostic-2026-10-25.md) 显示 task 4 的 open-container 条件在 approach/transport 阶段 joint drift 和 eye-in-hand RGB 误差最高，task 1 query transport 的 translation 误差偏高，release 阶段在有样本的任务中夹爪误差放大；同时 task 3/task 7 各有一条 support 专家 replay 未成功。下一项先复核失败专家 replay，再考虑阶段平衡或 gripper loss 的单变量 pilot。该诊断仍不选择 checkpoint、不更新 SPT/Knowledge、不注册 Module。

## 隐藏态模式与闭环成功

每格为 support demo/query demo/official candidate 三组各 3 个初态的成功数：

| 训练视图 | Continuous | 每 10 步 reset | Stateless |
|---|---:|---:|---:|
| stored same-index | 1/3 / 0/3 / 0/3 | 0/3 / 1/3 / 0/3 | 0/3 / 0/3 / 0/3 |
| causal pre-action | 0/3 / 0/3 / 0/3 | 0/3 / 0/3 / 0/3 | 0/3 / 0/3 / 0/3 |

在 same-index 臂中，每 10 步 reset 只是将单个成功从 support demo 0 换到了 query demo 31，总成功仍为 `1/9`；stateless 为 `0/9`。causal 臂三种模式均为 `0/9`。所有官方 candidate 初态都是 `0/3`。因此“短窗口训练、连续 hidden 部署不一致”会改变行为，但单靠 reset 没有提高总成功数，不能作为闭环失败的充分解释。

## 相同示教输入上的 teacher-forced 比较

每个 demonstration 被按自然时间顺序拆成 10 步窗口。Continuous 每个 demo 起点清空 hidden，demo 内跨窗口继续传递；`reset_each_window` 则每个窗口从零 hidden 开始。两种模式使用完全相同的因果对齐 RGB/proprioception/action 输入，按各窗口有效动作步汇总。

| 训练视图 | 角色 | Continuous MSE | Reset 每窗口 MSE | Continuous gripper MSE | Reset gripper MSE |
|---|---|---:|---:|---:|---:|
| stored same-index | support | 0.07253 | 0.08313 | 0.22649 | 0.25687 |
| stored same-index | query | 0.08413 | 0.09318 | 0.29023 | 0.31288 |
| causal pre-action | support | 0.07188 | 0.08771 | 0.24490 | 0.28673 |
| causal pre-action | query | 0.08513 | 0.09683 | 0.31549 | 0.33357 |

连续 hidden 的 teacher-forced MSE 在四组中都低于每窗口清零。也就是说，连续隐藏态在专家强制输入下有利于拟合长程轨迹。可是这一优势没有转化成更高闭环成功率。后段窗口误差有升高，例如 causal-trained query 最后窗口 continuous MSE `0.26983`，reset `0.41326`；但窗口只剩 15 个有效步，且此前窗口并非单调恶化。不能据此说“GRU 长期记忆本身导致失败”。

## 诊断判断

1. **隐藏态实现是行为因素之一，但不是已识别的根因。** 改变 reset 方式改变了夹爪与末端运动轨迹，说明 Policy 确实依赖 recurrent state；总成功数和官方初态成功数却没有改善。
2. **专家分布内的连续记忆表现更好。** teacher-forced 模式下连续 hidden 的误差低于重置模式；因此不能简单把部署改为无记忆或短记忆作为修复。
3. **主要缺口仍在 teacher-forced 到 free-run 的迁移。** causal-trained Policy 在因果 query 上仍有约 `0.085` MSE，但三个 rollout 组全部失败。错误可能来自首批动作小偏差、闭环状态偏移、视觉/动作映射或 BC 平均动作；本轮没有分离这些来源。
4. **时序修正依然必要，但并不充分。** causal loader 的语义修正成立；单 seed 下 causal 训练没有胜过 same-index，也不能由此判断因果配对无用。

## 后续诊断

这些仍是行为误差传播诊断，不是改 GRU 架构或扩大训练预算。相机贡献排序已经在三个 seed 上复核；单变量 eye-in-hand 平移增强 pilot 已完成，但跨 seed 结果不稳定。基础单 SPI 未稳定前不启动 A/B/C 或 Module qualification。
