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

## 后续动作偏差追踪（已实现，待运行）

为避免把“GRU 长记忆”当成直接解释，runner 另加一个可选配对追踪开关。它只在配置明确启用并校验 causal Policy final hash 后，针对开放的 support/query demonstration 0/1/2 与 30/31/32：

- 沿专家执行轨迹逐步向 Policy 提供 pre-action observation，记录 teacher-forced Policy action 与专家 action 的分量误差及首个 gripper sign mismatch；
- 从完全相同 `init_state` 重新开始，让 Policy 自由执行，再与相同 horizon 的专家 joint/gripper 状态、agentview 和 eye-in-hand RGB 做 10 步分箱比较；
- 不触及 official candidate、validation、held-out 或 qualification 初态，不训练、不选模型。

这一追踪尚未运行，不能预先断言夹爪误差、视觉表示或控制接口是哪一个先造成偏移。代码定向测试 `17 passed`，全量测试 `529 passed, 2 skipped`。配置/结果完成后应回填本节及研究路线。

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

建议下一轮只跑 6 个已使用的 support/query demo 初态，先对同一因果 Policy 逐步记录 expert action 与 Policy action 的 L1/L2 误差、gripper phase、joint/gripper state，并在首次夹爪动作分歧、首次接触阶段、专家成功步数处对齐。自由 rollout 再记录相同 observable 的轨迹偏差。这样能确定偏差是否在第一步就出现、是否集中于夹爪，还是先因视觉/运动偏差逐渐离开专家状态分布。

这应是行为误差传播诊断，不是再改 GRU 架构或扩大训练预算。若证据表明 MSE 动作回归将 `-1/+1` 夹爪命令平均化，再设计离散夹爪 head 的单变量 pilot；若首动作接近专家而状态快速分离，则检查 action repeat、控制频率和图像/状态同步。以上后续干预均需独立冻结配置并保留当前结果。基础单 SPI 未稳定前仍不开始 FOMAML / A-B/C，也不注册 Module。
