# LIBERO HDF5—当前重放状态对齐审计（2026-10-25）

## 目的与边界

专家重放一致性审计发现，task 3 demo 1 和 task 7 demo 0 在当前 adapter 中未成功，而 HDF5 的 `rewards/dones` 把它们记录为成功。本次审计检查这是否由状态索引、动作截断或当前环境执行轨迹与 HDF5 记录不一致造成。

审计固定四条 source-SPI 示教：task 3 的失败 demo 1 与成功控制 demo 0，task 7 的失败 demo 0 与成功控制 demo 1。每条示教使用 HDF5 `init_state` 和完整存储动作，在 `stabilization=0/5` 下各重放一次；只读取 `get_sim_state()` 做取证，不把私有状态送入 Policy 或 Knowledge。

- 配置：[replay state alignment audit](../configs/libero_replay_state_alignment_audit_v1.json)，SHA-256 `4a6a1895720091f76acf17352cfc055b444ccf9388742830b0c1289cfd0b1b41`。
- 结果：`/tmp/libero-replay-state-alignment-20261025-v3.json`，SHA-256 `5a8ac5887752f450fe6c120539bf27eacddbc7c9097f73ca40ebd6febc5ec26c`。
- 运行提交：`97ec8ab6320e43afc9048d4aa18b1c1ac48d904f`；CUDA `cuda:1` / RTX 4090；耗时约 `25.93s`。
- `formal_result=false`；没有训练、checkpoint 选择、validation/held-out/qualification 访问、SPT/Knowledge 更新或 Module 注册。

## 对齐语义

LIBERO 的数据生成脚本先把 `states[0]` 设置为仿真初态，再执行 `action[t]`，并把执行后的仿真状态用于检查 `states[t+1]`；同一执行后的 observation 被保存为 `obs[t]`。因此本审计采用：

- replay 初态 `sim_state[0]` 对比 HDF5 `states[0]`；
- replay 执行第 `t` 个动作后的 `sim_state[t+1]` 对比 HDF5 `states[t+1]`；
- replay 执行第 `t` 个动作后的 observation 对比 HDF5 `obs[t]`。

这个时间关系避免把已知的 post-action observation 错当成 pre-action 输入，也保留了 causal Policy 的边界。

## 结果

下表为 `stabilization=0`，因为它保持 HDF5 初态和第一条存储动作之间的原始边界。`state` 是完整 MuJoCo 状态的 L∞ 误差，`joint/gripper` 是对应 policy observation 的 L∞ 误差，`eye` 是 eye-in-hand RGB 的最大 MAE。

| case | HDF5 标签 | 当前 replay | 首步 state | 首步 joint | 首步 gripper | 全轨迹最大 state | 全轨迹最大 joint | 全轨迹最大 eye |
|---|---|---|---:|---:|---:|---:|---:|---:|
| task 3 demo 1 | success | failed | .0985 | .00267 | .00529 | 3.4620 | .04235 | 28.32 |
| task 3 demo 0 | success | success | .1228 | .00438 | .00529 | 3.0871 | .01136 | 9.73 |
| task 7 demo 0 | success | failed | .1328 | .00381 | .00527 | 4.4251 | .19948 | 28.53 |
| task 7 demo 1 | success | success | .1328 | .00302 | .00527 | 6.5691 | .05447 | 24.83 |

四条轨迹的 `sim_state[0]` 与 HDF5 `states[0]` 均精确一致；但四条轨迹都在第一步后出现状态偏差，包括两个当前 replay 成功的控制示教。因而“当前 replay 失败”不能作为 task 3 demo 1 或 task 7 demo 0 独有的 Policy 失败证据。HDF5 的 `states/rewards/dones` 仍保持原始记录：四条示教的 reward sum 均为 `1`，最后一个 `done` 均为 `1`。

加入 5 步 zero-action stabilization 后，初态本身相对 HDF5 `states[0]` 已产生约 `0.25` 的状态偏差，并且 task 3 demo 1 的 replay 从失败变为成功；task 7 demo 0 仍失败。因此 stabilization 会改变因果边界，不能用它来修正 HDF5 对齐，也不能据此选择训练设置。

作为补充取证，读取每条 demo 保存的 episode `model_file`，将旧路径映射到当前安装的 LIBERO/robosuite 资产后重放首个动作，首步 joint 与 HDF5 observation 仍存在约 `0.0033–0.0040` 的误差，未消除上述差异。这说明仅把 canonical BDDL 换成保存 XML 不能解释全部偏差；更可能还涉及数据生成时的仿真/控制器版本或动作执行细节。该补充检查不是新的正式协议。

## 诊断结论

1. HDF5 状态索引语义可以由 LIBERO 数据生成脚本确定，当前审计使用了正确的 state/observation 对齐关系。
2. 当前环境在固定初态下的动作重放是确定的，但不能逐步复现 HDF5 的历史状态和 observation；偏差在成功控制示教中也存在。
3. task 3 demo 1 的 stabilization 敏感性和 task 7 demo 0 的稳定失败仍然是真实的当前 replay 行为，但它们不能直接解释为 Policy、视觉表示或 action head 的根因。
4. 此前分层误差诊断中把当前 replay 成功的专家轨迹作为条件是一个环境侧筛选边界；在 HDF5—当前环境语义没有解决前，不能把失败 demo 排除后得到的 phase 均值升级为正式 Policy 误差结论。

## 后续审计

回放语义冻结审计和当前 runtime 专家轨迹审计已经完成，见 [LIBERO 回放语义冻结审计](libero-replay-semantics-audit-2026-10-25.md) 与 [LIBERO 当前运行时专家轨迹审计](libero-current-runtime-replay-audit-2026-10-25.md)。相关 controller 字段、control frequency、动作边界和 adapter 校验均与 HDF5 配置一致；当前运行时仍在首步产生 qvel/qpos/观测偏差，但重复 sim-state/observation digest 稳定。下一步把历史 HDF5 observation 与当前 runtime observation 分成两条协议，先比较它们对 Policy 输入的影响。

在这个边界有可复现解释前，暂停 phase-balanced、gripper-loss 和 A/B/C 正式候选实验。
