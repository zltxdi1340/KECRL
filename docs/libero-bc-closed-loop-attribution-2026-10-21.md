# LIBERO 单 SPI BC 闭环失败归因（2026-10-21）

本诊断沿用 task 0、seed `20261021` 和 500 次 Adam 更新，不增加训练预算，检查上一轮“独立 query MSE 明显下降、官方初态闭环 0/3”的原因。结果排除了动作尺度或模拟器接口损坏，也表明官方初态分布差异和 GRU 长时状态都不是唯一原因；更关键的发现是，当前 HDF5 的同索引 observation/action **不是因果控制配对**。

本次仍是单 source SPI、单 seed 的失败归因，不是学习 pilot、架构对照、SPT 更新、Module qualification 或正式结果。

## 固定边界与 provenance

- 运行代码 commit：`c3eab6f`（`增加LIBERO闭环失败归因诊断`），运行时 tracked worktree clean。
- 固定配置：[`configs/libero_spatial_bc_closed_loop_attribution_v1.json`](../configs/libero_spatial_bc_closed_loop_attribution_v1.json)，SHA-256 `5c1f367f9bc14b60cddff2d9b1fa526a51e58440b2ec737914498ff89707dc11`。
- 入口：[`experiments/run_libero_bc_closed_loop_attribution.py`](../experiments/run_libero_bc_closed_loop_attribution.py)。
- 训练仍只用 source task 0 的 `meta_support` 0--29：500 updates、37,534 valid-step-updates、batch 8、序列长度 10、`lr=3e-4`。
- 初态组：`meta_support` demo 0/1/2、`meta_query` demo 30/31/32、官方 `candidate_nonregression` 0/1/2。
- demo 初态使用 0 个稳定步，以便专家动作原样重放；官方初态沿用 benchmark 的 5 个零动作稳定步。
- Policy 分别采用连续 GRU、每 10 步重置隐藏态、逐步 stateless 三种推理方式；三者共享同一组 500-update 权重，不做 checkpoint 选择。
- validation、held-out、qualification 均未访问；SPT 未更新，Module 未注册。
- 结果文件：`/tmp/libero-bc-closed-loop-attribution-20261021-v1.json`，SHA-256 `b743c34d343e59ffea8599ba2d543063d9bbac76cb70898e48a2b4938ee9c0fa`。
- `cuda:1` / RTX 4090，运行 187.78 秒，峰值 CUDA allocation 286,148,096 bytes。

训练 loss 与上一轮逐项复现：首 batch `0.243631`、末 batch `0.075395`、均值 `0.141713`，说明以下差异来自闭环检查，不是重新训练漂移。

## 专家重放与视觉约定

| 初态来源 | 专家成功 | 成功步数 |
|---|---:|---|
| meta_support 0/1/2 | 3/3 | 82、73、79 |
| meta_query 30/31/32 | 3/3 | 114、69、82 |

六条保存动作都能从各自 HDF5 `init_state` 在当前固定 LIBERO revision 中触发成功。这排除了动作维度、范围、控制器配置和示教初态不可复现等接口问题。

第一步重放 observation 与 HDF5 图像的 agentview MAE 为 `3.07--4.70`，垂直翻转后的 MAE 为 `54.81--55.89`；关节 RMSE 也很小。因此训练数据与在线 adapter 使用同一图像方向，当前失败不是上下翻转造成的。

## Policy 闭环结果

| 初态来源 | 连续 GRU | 每 10 步重置 | stateless |
|---|---:|---:|---:|
| meta_support 0/1/2 | 1/3 | 0/3 | 0/3 |
| meta_query 30/31/32 | 0/3 | 1/3 | 0/3 |
| candidate_nonregression 0/1/2 | 0/3 | 0/3 | 0/3 |

连续 GRU 在 support demo 0 于 86 步成功；每 10 步重置在 query demo 31 于 233 步成功。其余 25 个 rollout 全部在 300 步超时。默认连续 GRU 从训练初态也只有 1/3，因此不能把官方初态 0/3 单独归因为 demonstration/evaluation 初态分布差异。

GRU 状态处理会改变轨迹，但不是充分修复：stateless 在九个初态全部失败，平均 arm action norm 约 `0.37--0.38`，低于专家的 `0.84--0.89`，并且 demo 六条 stateless rollout 的夹爪始终输出负号、零次切换。连续与周期重置能恢复部分动作幅值和夹爪切换，却仍不稳定。demo 初态首动作相对对应专家的 L2 误差均值为 support `0.402`、query `0.314`；误差从第一步已经存在，不只是 rollout 后期累积。

这两个成功 episode 说明 Policy 并非绝对无法驱动任务，但 `2/18` 个 demo-start/mode 组合不能构成稳定 SPI，更不能形成 Module。

## 关键发现：同索引 observation/action 时序错位

固定 LIBERO revision 的 `scripts/create_dataset.py` 先执行 `action[t]`，再把环境返回的 observation 写到 HDF5，同时仍把 `action[t]` 写在相同索引。经过开头 5 步裁剪后：

- HDF5 `init_state` 是执行首个保留动作前的状态；
- `obs[t]` 是执行 `action[t]` 后的 observation；
- 当前 loader 训练的是 `obs_after(action[t]) -> action[t]`，而在线控制需要 `obs_before(action[t]) -> action[t]`。

因此上一轮下降到 `0.08830` 的 query MSE 是独立轨迹上的真实拟合结果，但它拟合的是带动作后果信息的反向配对，不能证明学会了因果闭环动作映射。这能同时解释“离线 query 很好”和“闭环几乎全失败”。该问题来自数据时序语义，不是 HDF5 文件损坏；专家动作按正确顺序重放仍为 6/6。

## 当前结论与下一步

1. 动作接口、模拟器 revision、示教初态和图像方向已通过；不需要更换环境或扩大训练预算来检查这些项。
2. 官方初态分布可能仍增加难度，但 Policy 在 support/query demo 初态也不稳定，当前证据不支持把它列为首要原因。
3. GRU 长序列与长度 10 训练窗口之间存在部署差异，但三种隐藏态方式都不稳定；应先修复更基础的监督时序定义。
4. 当前最先处理的是因果数据视图：为每条示教从 `init_state` 重建首个 pre-action observation，此后用 `stored_obs[t] -> action[t+1]`，从而得到与原轨迹等长的 `pre_action_obs[t] -> action[t]`。
5. 下一诊断固定同一模型、seed 和 500-update 预算，对比现有 same-index 与因果对齐数据视图，并分别报告 causal support/query loss 和三组闭环结果。只有因果对齐后基础 SPI 稳定执行，才进入多 source pilot 和 SPT/FOMAML。

