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

这些仍是行为误差传播诊断，不是改 GRU 架构或扩大训练预算。下一项把同一受控 free-run 分叉拆成 RGB 与本体感知输入的模态对照：保持 recurrent hidden 和另一输入模态不变，分别替换为专家观测与自由观测，测量两种输入对动作偏移的相对贡献。它仍不访问正式评估状态，也不进行 SPT/FOMAML、Knowledge 更新或 Module qualification。只有在模态归因完成后，才决定是否进行单变量修复 pilot；基础单 SPI 未稳定前不启动 A/B/C 或注册 Module。
