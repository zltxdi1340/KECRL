# LIBERO 专家示教重放一致性审计（2026-10-25）

## 目的与范围

分层误差诊断发现 task 3 support demo 1 和 task 7 support demo 0 的专家动作在当前 adapter 中未成功。本次审计固定其 HDF5 `init_state` 和动作序列，加入同任务成功示教作为控制，检查 seed、zero-action stabilization 和重复创建 adapter 是否造成结果变化。

本次只做专家动作重放，不训练 Policy，不访问 validation、held-out 或 qualification 状态，不选择 checkpoint，不更新 SPT/Knowledge，也不注册 Module。

- 配置：[expert replay consistency audit](../configs/libero_expert_replay_consistency_audit_v1.json)，SHA-256 `c27e77fb254ad765878c85f7c221df9d50bcc61b14e9008e4aa3eaf115814620`。
- 结果：`/tmp/libero-expert-replay-consistency-20261025-v1.json`，SHA-256 `a3d8a1482adba274819e80130f9fd19135f70d720938be12df3c786c96e86b2f`。
- 运行提交：`5c0aaafdb2c04af9ea0bdcc1013a1da7a7034e03`；耗时 `213.90s`。
- 每个 case 使用三个 seed（`20261021/22/23`）、`stabilization=0/5`、每个设置重复三次；每次重复新建 adapter，并对完整 joint/gripper/RGB 轨迹计算 digest。

## 结果

| case | stabilization | seed 结果 | 三次 trace digest | 结论 |
|---|---:|---|---:|---|
| task 3 demo 1（此前失败） | 0 | 三个 seed 均 `0/3` | 每个设置均 1 个 | 确定性失败 |
| task 3 demo 1（此前失败） | 5 | 三个 seed 均 `3/3` | 每个设置均 1 个 | stabilization 敏感，全部成功 |
| task 3 demo 0（成功控制） | 0 | 三个 seed 均 `3/3` | 每个设置均 1 个 | 确定性成功 |
| task 3 demo 0（成功控制） | 5 | 三个 seed 均 `3/3` | 每个设置均 1 个 | 确定性成功 |
| task 7 demo 0（此前失败） | 0 | 三个 seed 均 `0/3` | 每个设置均 1 个 | 跨 seed、重复均失败 |
| task 7 demo 0（此前失败） | 5 | 三个 seed 均 `0/3` | 每个设置均 1 个 | 跨 seed、重复均失败 |
| task 7 demo 1（成功控制） | 0 | 三个 seed 均 `3/3` | 每个设置均 1 个 | 确定性成功 |
| task 7 demo 1（成功控制） | 5 | 三个 seed 均 `3/3` | 每个设置均 1 个 | 确定性成功 |

task 3 demo 1 在 stabilization 0 时每次执行完整存储动作 `101` 步后未成功；加入 5 步 zero-action 后每次在 `91` 步成功。task 7 demo 0 在两种 stabilization 下都完整执行 `155` 步仍未成功。成功控制示教的状态稳定，但 task 7 demo 1 的成功步数随 seed 有小幅变化（stabilization 0 为 `112/114/112`，stabilization 5 为 `112/109/112`），说明 seed 会影响成功时刻，却没有把失败 case 变成成功。

## 与 HDF5 标签的矛盾

HDF5 中 task 3 demo 1 和 task 7 demo 0 的 `rewards` 最后值均为 `1`，`dones` 最后索引分别为 `100` 和 `154`；数据标签把两条轨迹记录为成功。当前固定 LIBERO checkout + adapter 的重放却分别得到：

- task 3 demo 1：stabilization 0 失败、stabilization 5 成功；
- task 7 demo 0：stabilization 0/5 都失败。

因此这不是 Policy 的失败证据，而是“演示数据成功标签—当前初态/动作重放—环境成功判定”之间的可复现不一致。重复 digest 排除了普通随机噪声；task 3 的 stabilization 敏感性说明 zero-action 边界会改变轨迹；task 7 的跨设置稳定失败则需要检查更具体的状态或动作语义。

首个 post-action joint-state RMSE 也支持保持 causal 边界：task 3 demo 1 在 stabilization 0/5 分别约 `0.001887/0.008542`，task 7 demo 0 约 `0.001611/0.008520`。5 步 stabilization 改善 task 3 成功率的同时，改变了执行首动作时的状态，不能直接用于 causal pre-action 训练对齐。

## 诊断结论

1. LIBERO adapter 和 MuJoCo 路径在重复运行中是确定的；当前失败不是同进程随机抖动。
2. task 3 demo 1 的 replay 结果对 stabilization 极其敏感，说明初态边界需要明确记录；因果训练仍应保持 stabilization 0，不能用成功率倒推改边界。
3. task 7 demo 0 的失败在三个 seed 和两种 stabilization 下稳定出现，而同任务 demo 1 稳定成功；问题更可能与该 demonstration 的初态/动作序列语义、数据生成轨迹或 task-specific replay 差异有关。
4. 在这两条示教的 replay 一致性解决前，不能把 task 3/task 7 的 Policy 误差或成功率用于正式 source-SPI 门槛，也不能开始 A/B/C。

## 后续审计

状态轨迹差异审计和回放语义冻结审计已经完成，见 [HDF5—当前重放状态对齐审计](libero-replay-state-alignment-audit-2026-10-25.md) 与 [LIBERO 回放语义冻结审计](libero-replay-semantics-audit-2026-10-25.md)。结果显示四条轨迹（包括两个当前 replay 成功控制）都在首个动作后偏离 HDF5 状态；相关 controller 配置和动作边界没有差异，而 HDF5 没有生成时 simulator 版本。此前的失败 demo 不能直接作为 Policy 失败样本，下一步先冻结当前 runtime 的可复现专家轨迹。phase-balanced、gripper-loss 和 A/B/C 仍暂停，直到 replay 兼容性得到可复现解释。
