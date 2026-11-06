# LIBERO task 4/7 局部 open-transition gripper loss 配对诊断（2026-11-01）

本项已完成 **12 个模型 × 500 updates** 的配对诊断，六个 baseline 均精确复现历史权重。局部 2× gripper loss 未通过冻结的停止规则：两个任务的局部 MSE 都只在 2/3 seed 改善，task 4 平移均值增加 7.03%，两个任务七维 MSE 分别增加 4.19%/6.63%。**停止该候选分支，保留原 baseline，不追加预算、不搜索权重、不进入该候选闭环。** 本项不是 SPT 更新、正式 SPI、Knowledge Evolution 或 Module qualification。

## 冻结实现与训练协议

- 损失实现：[libero_bc_local_gripper_loss.py](../src/algorithms/libero_bc_local_gripper_loss.py)。新增模块，不改动旧 Policy、数据集或 baseline loss 实现。
- 执行配置：[libero-task4-task7-local-gripper-loss-diagnostic_v1.json](../configs/libero-task4-task7-local-gripper-loss-diagnostic_v1.json)。
- runner：[run_libero_task4_task7_local_gripper_loss_diagnostic.py](../experiments/run_libero_task4_task7_local_gripper_loss_diagnostic.py)。
- 候选来源：[专家目标动作误差诊断](libero-task4-task7-expert-action-error-diagnostic-2026-11-01.md)及其固定的[设计稿](../configs/libero-task4-task7-open-transition-gripper-weighting-proposal_v1.json)。设计稿保留历史状态，新的执行授权由本轮配置记录。

两臂为 `source_baseline` 和 `open_transition_gripper_weight_2`，覆盖 task `4,7`、seeds `20261022/20261023/20261024`，共 12 个模型。每模型固定 500 Adam updates，总预算 6,000；只保存最终 500 checkpoint，不挑中间 checkpoint、不调整系数或增加 updates。Adam lr `0.0003`、batch `8`、sequence length/stride `10/10`、gradient clip norm `1`、workers `0`；CNN-only Full Policy、相同初始化及 shuffled batch 顺序，不加 jitter 或其他增强。实际 optimizer defaults 和每步 loss/gradient norm 记录入结果。

设 `I_open(t)` 为单条完整 support 示教内全部专家 gripper 从非负切到负的事件前后各 5 步窗口之并集，限制 `t>=1`，不跨 demo、重叠不重复加权。候选权重 `w(t)=1+I_open(t)`，只用于第七维：

```text
L = Σ_valid_t [ Σ_d=0..5 (prediction-target)² + w(t) × gripper_error² ]
    / (7 × valid_step_count)
```

mask 在完整 meta-support 动作序列上计算，再按原训练窗口切片；padding 不参与损失，真实 `t=0` 的权重保持 1。baseline 直接调用原 `masked_behavior_cloning_loss`。单位权重或空窗口时，新函数精确返回原损失；梯度测试核验其他六维不变，只有指定有效步的夹爪梯度加倍。部署时不输入专家命令或 phase 标签。

### 数据角色与 source 首帧预算

训练只用 source `meta_support` 示教 `0..29`，query 只用同任务 `meta_query 30..49`，每个任务所有六个 support fit 完成后才创建 query dataset。数据文件 SHA-256、角色、causal 视图均沿用 baseline。

训练保持原完整 causal 输入，包含真实首帧，不采用上一项只读评分时排除 `t=0` 的覆盖范围。首帧按原训练 helper 临时读取 **100 条已授权 source 示教的 `init_state`** 重建：两个 task 各 support 30 / query 20，合计 100 次 adapter reset、4 个 adapter，固定 seed `20261022`，每角色重建独立 adapter 并按 demo 顺序递增，native horizon `300`、stabilization `0`。

这些 source 示教初态只用于重建四类公开观测，不作为 Policy 特征、Knowledge 证据或输出 artifact；不读取 live private simulator state。新增保存 100 个公开首帧的 RGB/joint/gripper 与 demo ID，方便以后纯离线复核，不保存其私有初态数值。这里的示教编号与 official evaluation-state 编号属于不同数据角色。

**Policy rollout、candidate/official evaluation-state 读取预算均为 0**。未访问 qualification、validation、held-out 或 final evaluation，未注册 Module。

### 重现、配对与资源门槛

配置固定 26 个源文件/结果 hash，包括损失、训练/core、数据访问、Policy、SPT 初始化实现及 native reset 代码。baseline 初始和最终 500 权重必须精确匹配历史 budget-sweep 的 500 点。两臂的初始权重、500 个 batch 的 demo/start 顺序、首批输入/目标摘要及 valid/局部标记步数必须完全相同。

baseline 完整 causal support/query 指标还需在原 `1e-10` 绝对容差内匹配参考，权重 hash 门槛不使用浮点容差。后续 `t>=1` CPU 分解的输入摘要与此前专家目标分析逐批精确对齐，baseline CPU 预测也要求逐元素一致。完整含 `t=0` 的 CUDA 指标与排除首帧的 CPU 分解分别记录。

训练配置为 RTX 4090 `cuda:1` / EGL device 1，torch threads 16；启动时 GPU 1 占用须不超过 1,024 MiB 且 utilization 不超过 10%，否则不启动。训练全部完成后再用 CPU 单线程、batch 32 做分量/命令窗口诊断；不因占用其他作业而切换到 GPU 0。已做零优化器步的初始化检查，三个 seed 在 CPU threads 1/16 下的初始权重均匹配参考；真实 CUDA 训练仍保留最终 hash 检查。

## 冻结停止与后续诊断规则

技术错误立即停止：输入 hash/角色不符、初始或最终 baseline hash 不符、两臂初始化或 batch 顺序不同、loss/gradient 非有限、SPT 初始化对象被修改。保留失败记录与已生成 artifact，不读后续 query，不以容差放宽或换 checkpoint 继续。

性能规则在全部 12 个固定 500 endpoints 完成后检查，不据 query 中途停某个 seed。对 **两个任务分别**要求：

1. query `t>=1` 的 open 窗口 gripper MSE 在三个 seed 中均下降；同窗口 gripper sign error 的三 seed 均值不增加。
2. query 全步 translation MSE 的三 seed 均值候选/baseline 比值 `<=1.02`，每个 seed 比值 `<=1.05`。
3. query 全步 rotation MSE 的三 seed 均值比值 `<=1.05`，七维 MSE 均值比值 `<=1.02`。

任一规则失败即停止该候选分支，不追加 updates、搜索其他权重或挑选 seed。全部通过也只支持另行冻结一次 candidate-only 闭环诊断；本轮不自动启动闭环，更不构成 Module 资格。上述规则是预先声明的描述性诊断门槛，没有显著性检验，不是正式 qualification 阈值。

## 结果与 provenance

### 执行与配对检查

12 个模型均完成固定 500 updates，共 **6,000 updates**，没有失败重试或额外更新。六个 baseline 的初始与最终权重均匹配参考；六组 baseline/候选的初始参数、500 批次顺序、首批输入/目标摘要、有效/局部标记步数完全一致。六个 baseline 的完整 causal support/query 指标也匹配历史参考；SPT 初始化对象全部保持不变。

实际有 **461,482 次有效示教步呈现**；候选中局部加权的呈现次数为 **18,045**。四个 source adapter 重建 **100 个公开首帧**，stabilization 和 Policy action steps 均为 0。没有 Policy rollout、candidate observation 或 official evaluation-state 读取。

### Query 目标动作误差

下表是同一 causal 专家输入、`t>=1`、三个 seed 的 MSE 均值。所有分量按自身维数归一化，七维总 MSE 沿用 uniform 权重，不能直接用训练时的加权 loss 比较两臂质量。

| task | arm | 平移 MSE | 旋转 MSE | 夹爪 MSE | 七维 MSE |
|---|---|---:|---:|---:|---:|
| 4 | baseline | `0.040584` | `0.002516` | `0.140425` | `0.038532` |
| 4 | 局部 2× 候选 | `0.043438` | `0.002588` | `0.142945` | `0.040146` |
| 7 | baseline | `0.040363` | `0.002142` | `0.172075` | `0.042798` |
| 7 | 局部 2× 候选 | `0.040926` | `0.002153` | `0.190212` | `0.045635` |

完整包含真实 `t=0` 的 CUDA 指标另存于 `full_causal_support_metrics/full_causal_query_metrics`，CPU 分量/命令窗口结果存于 `expert_target_evaluations`；停止规则只使用预先声明的 `t>=1` CPU 指标，没有混用覆盖范围。

### 局部收益与 seed 差异

open 切换窗口的三 seed 平均 gripper MSE 在 task 4 从 `0.981996` 降至 `0.933688`，task 7 从 `1.091029` 降至 `1.052506`；窗口 sign error 分别从 `35.05%` 降至 `34.58%`、从 `40.29%` 降至 `37.54%`。但逐 seed 不一致：

| task | seed | baseline open 窗口 gripper MSE | 候选 | 是否下降 |
|---|---:|---:|---:|---|
| 4 | 20261022 | `0.755648` | `0.735989` | 是 |
| 4 | 20261023 | `1.118160` | `1.201052` | 否 |
| 4 | 20261024 | `1.072178` | `0.864022` | 是 |
| 7 | 20261022 | `1.160182` | `1.059983` | 是 |
| 7 | 20261023 | `1.096252` | `1.062955` | 是 |
| 7 | 20261024 | `1.016652` | `1.034581` | 否 |

专家 open/close 条件之间存在取舍：query 专家 open 时误发 close 的均值在 task 4 从 `7.39%` 降至 `6.70%`、task 7 从 `11.98%` 降至 `9.92%`，但专家 close 时漏 close 分别从 `2.97%` 升至 `4.03%`、从 `3.35%` 升至 `4.99%`。整体 gripper MSE 及整体 sign error 都上升。它们仍是专家命令条件下的预测误差，不是环境实际抓取或 release 成败。

### 按冻结规则停止

| 检查 | task 4 | task 7 |
|---|---|---|
| open 窗口 gripper MSE 三 seed 均下降 | 失败：2/3 | 失败：2/3 |
| open 窗口 sign error 均值不增 | 通过 | 通过 |
| 整体平移 MSE 均值比值 `<=1.02` | 失败：`1.070327` | 通过：`1.013941` |
| 每 seed 平移 MSE 比值 `<=1.05` | 失败：最大 `1.093928` | 通过：最大 `1.022498` |
| 旋转 MSE 均值比值 `<=1.05` | 通过：`1.028589` | 通过：`1.005178` |
| 七维 MSE 均值比值 `<=1.02` | 失败：`1.041885` | 失败：`1.066286` |

两任务均未通过；结果状态为实验正确执行完成，候选后续门槛为失败，`decision=stop_candidate_branch`。不能因局部均值下降或 task 4 某一个 seed 的整体 MSE 下降就保留该候选。没有追加更新、换系数、挑 checkpoint 或继续闭环。

本结果只否定了本次固定 2×、±5 步、500-update 候选在这些诊断条件下作为后续方案的依据，不能推广为所有局部损失无效，也不能把失败唯一归因于某个组件。梯度裁剪触发次数已记录：候选在部分模型中更频繁触发 clip；本轮没有据此改动 clipping 或其他超参数。

### 实际成本与可复现信息

- 总 elapsed **`387.79s`**：训练循环合计 **`68.41s`**、source 首帧重建 **`102.84s`**、完整 CUDA 评价 **`8.95s`**、CPU 分量评价 **`206.11s`**，其余为初始化、hash/I/O 等开销。短训练耗时对应实际 6,000 个 Adam updates，不是环境学习步数或收敛声明。
- RTX 4090 `cuda:1`，启动时 `30 MiB / 49,140 MiB`、utilization 0%，GPU UUID/driver 已记录。peak CUDA allocated **`286,280,192 bytes`**；训练 torch threads 16、CPU 评价 threads 1。
- Python `3.10.22`、PyTorch `2.4.1+cu124`、CUDA `12.4`；cuDNN deterministic=true、benchmark=false、allow_tf32=true，matmul allow_tf32=false。实际 Adam defaults：betas `(0.9,0.999)`、eps `1e-8`、weight_decay `0`、amsgrad=false、foreach/fused 为库默认 `None`，其余见各模型 `actual_optimizer`。
- 配置 SHA-256：`8c0d1cf573aed89b4ba8572b0fa23647197d9b95cb4ce712a9985df1551cee2d`。
- runner SHA-256：`7da1127b15c25bc1c5248e608f34771faf3e52cc5322a399b1ae33886f3428d9`；损失实现 SHA-256：`0ad7c1afa68ba3c988ed7712d45ecc54b6695b93713b14e29d816cab87b21027`。
- 结果：`/tmp/libero-task4-task7-local-gripper-loss-20261101-v1.json`，SHA-256 `c231ced84aa4d35e573e0ad33f0923bc91e7d7bb6ef0bbd615743505efa4082a`。
- Artifact 根目录：`/tmp/libero-task4-task7-local-gripper-loss-artifacts-20261101-v1`，含 12 个最终 checkpoint、4 份公开 source 首帧 archive、4 份专家目标/预测 archive 和 `events.jsonl`。全部保持在仓库外。
- 独立只读复核：`/tmp/libero-task4-task7-local-gripper-loss-independent-audit-20261101-v1.json`，SHA-256 `359a3f656c0421c714aa0f083a32659e1bf9e654dcc39e9d20f4ba48033c3174`。
- 日志：`/tmp/libero-task4-task7-local-gripper-loss-20261101-v1.log`，SHA-256 `5722d4df494971218a49f88b6c93d1331b2aea13fe9e8211e484da3e7991cad0`。

### 独立复核与元数据边界

复核了全部 12 份 checkpoint 的文件/权重 hash、六组初始化和批次顺序、全部 6,000 次 update 的记录、461,482 个有效训练步呈现及 18,045 个候选加权步呈现。100 个公开首帧的数组/digest 均核验，archive 不含私有初态；505 个专家输入批次与上一项精确对齐，baseline CPU 预测逐元素一致。重算 168 个模型×命令分层指标，独立浮点求和最大差 `2.66e-15`；目标、索引、mask 和 hash 对齐全部精确，不放宽时序门槛。停止判定另按配置阈值逐项核对一致。

旧 `_capture_role_dataset` 返回的 `access` 是**用于读取示教初态的索引 dataset 快照**，因此 raw result 的 `source_role_captures.access.observation_action_alignment` 会显示 `stored_same_index`。同一 helper 随后将重建的首帧传入 `_role_dataset(initial_observations=...)`，实际返回并用于训练的 dataset 为 `causal_pre_action`。审计分别记录了索引快照与实际训练视图，依据冻结 helper/data code、100 个真实公开首帧及六个 baseline 权重/完整指标复现；没有把索引字段改写成训练语义，也没有改写原始结果。后续新版本可明确分开这两类 access record。

专项新增 **24 tests**；相关损失/Policy/数据访问/源 query 检查 **41 passed**；在 `CUDA_VISIBLE_DEVICES=''` 下排除 `test_crafter_*`、`test_spatial_gather_spt.py`、`test_smoke.py` 后，92 个测试文件共 **433 passed**。可选 Crafter/smoke 组未运行，实际 CUDA 路径由本轮全部模型和精确 baseline 复现验证。`py_compile`、环境检查及 `git diff --check` 通过。

执行命令：

```bash
nice -n 10 env -u CUDA_VISIBLE_DEVICES \
  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 OMP_NUM_THREADS=16 \
  OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=16 \
  conda run --no-capture-output -n libero-kecrl python \
  experiments/run_libero_task4_task7_local_gripper_loss_diagnostic.py \
  > /tmp/libero-task4-task7-local-gripper-loss-20261101-v1.log 2>&1
```

## 决定与下一步

保留原 CNN-only uniform-loss、500-update baseline，关闭本次局部加权候选分支，不为它追加闭环预算。既有 baseline/jitter `7/120 / 6/120` 闭环记录保持不变。下一步先收束已完成的输入对齐、表示、jitter 与损失诊断证据，明确仍未被分离的机制，再冻结不同的单变量诊断；不自动搜索邻近权重或改训练预算。SPT/Knowledge/Module 的实验边界不变。

文档核对附记：历史 2026-10-25 的全局 6× gripper 臂实际是 `uniform_phase_gripper_6x_action_mse`，没有叠加 inverse-frequency phase-balanced，且分母为总动作权重 `12×valid_steps`。此前文字将它描述为 phase-balanced 背景不准确，已按原配置/runner 更正；历史结果和训练代码未改变。它与本轮局部权重、固定 `7×valid_steps` 的实现分开解释。
