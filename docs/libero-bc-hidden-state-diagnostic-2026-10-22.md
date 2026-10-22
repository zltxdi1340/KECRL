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

下一项做局部单步反事实诊断：固定当前 causal Policy 和六个已开放 demo，在专家轨迹的 `0/10/20/30/40/50` 步及首次夹爪分歧处，从相同专家前缀重放到同一 pre-action 状态，再分别执行专家动作和 Policy 预测动作，比较紧邻的一步 joint/gripper 与 RGB 变化。先用重复专家前缀确认状态重放一致。它将估计“在同一状态下，当前预测动作误差会造成多大即时环境偏差”，但仍不用于训练、选模型或资格评估。若即时偏差集中在平移通道，再检查输出尺度/控制频率；若单步影响小而长程自由轨迹仍快速分离，则下一步查视觉观测和闭环状态覆盖。该诊断尚未运行。

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

这些仍是行为误差传播诊断，不是改 GRU 架构或扩大训练预算。单步反事实完成后再决定是否有必要单独试验离散夹爪 head；当前证据不支持先做夹爪头改造。基础单 SPI 未稳定前仍不开始 FOMAML / A-B/C，也不注册 Module。
