# LIBERO task 4/7 固定 500-update Policy 的专家目标动作误差（2026-11-01）

本项已完成 task 4/7 × 三个 seed × baseline/jitter 的 **12 个现有 500-update checkpoint** 专家目标误差对照。jitter 在两任务 query 的平移 MSE 上均为三 seed 下降，但夹爪收益不一致；所有 24 个 task × role × seed × arm 组合都存在 open 切换窗口的夹爪误差峰值。下一训练候选限定为该窗口内的局部 2× gripper MSE，不把 jitter 升为默认、不采用全局大权重。新增 optimizer updates 和环境 episodes 均为 **0**；新训练尚未运行。

## 运行前冻结协议

配置：[libero-task4-task7-expert-action-error-diagnostic_v1.json](../configs/libero-task4-task7-expert-action-error-diagnostic_v1.json)。分析脚本：[analyze_libero_task4_task7_expert_action_error_diagnostic.py](../experiments/analyze_libero_task4_task7_expert_action_error_diagnostic.py)。复用现有 `resolve_demonstration_access` 和 Policy 实现，不修改共享训练或数据集代码。

- 任务 `4,7`；training seeds `20261022/20261023/20261024`；两臂为 `source_baseline/public_state_jitter`。所有 checkpoint file/weight hash 与 budget-sweep 的固定 `500` 点核对，不加载或选择 1000/2000 点。
- source `meta_support` demonstrations `0..29`、`meta_query` demonstrations `30..49`，分别汇报，不触及其他 task 或锁定数据角色。
- 仅评估 `t>=1`：`HDF5 obs[t-1] → expert action[t]`。每条示教的 action 0 只允许用于专家命令切换标签，不做预测或计入误差；首帧不重建、不伪造，不调用会构造环境的 `_capture_role_dataset`。
- 100 条示教原有 14,590 个动作；排除 100 个 `t=0` 后，评估 14,490 个独立专家输入。每任务六个固定模型，共 86,940 行预测；CPU 重复前向验证最多额外 768 行，单独记录。
- CPU 单线程、batch size 32、analysis RNG seed `20261022`、CNN-only stateless；每批观察张量供同任务六个模型共同使用，逐模型检查输入未被修改，首批重复前向须精确一致。推理不加 jitter。
- 平移 `0:3`、旋转 `3:6`、夹爪 `6:7` 的 MSE/MAE/RMSE 按各自维数归一化；总七维 MSE 为 `(3×平移 MSE + 3×旋转 MSE + 夹爪 MSE)/7`。平方误差构成是代数分解，不能称为环境因果贡献。
- 分层：all、每条示教首个可评价动作 `t=1`、专家 open/close/neutral 命令、close/open 切换前后各 5 步。命令由专家 gripper action 正负号定义，不冒充真实 approach/transport/release 阶段。
- 另记 gripper sign confusion、漏 close、专家 open 时误发 close。空分层或零分母记 `null`，不以 0 伪造完美结果。不做未冻结的显著性检验。
- 每 task × role 保存一个 NPZ，包含目标动作、demo/action 索引、命令 masks、前一个专家 gripper 命令和六个模型预测；不复制 RGB、checkpoint 或数据集。逐批输入摘要、每条示教覆盖范围、模型 hash 和实际成本写入结果。

| task | role | 评估 `t>=1` | 排除 `t=0` |
|---|---|---:|---:|
| 4 | meta_support | 4,442 | 30 |
| 4 | meta_query | 2,987 | 20 |
| 7 | meta_support | 4,188 | 30 |
| 7 | meta_query | 2,873 | 20 |

元数据 preflight：`/tmp/libero-task4-task7-expert-target-preflight-20261101-v1.json`，SHA-256 `d3c4abc420be3e1370d56e27cc6b26f834516abbc343be1c836f93d7ba225f3e`。该 preflight 只读取允许角色的 shape 元数据，没有读取动作/观察数值、private simulator state 或示教初态属性。

## 解释边界

本指标覆盖范围排除了首帧，且为固定 CPU batch 的离线诊断，不能称为原完整 source-query MSE 的精确重算。它衡量专家状态下的动作误差，不能直接推出自由运行成功率或本体状态漂移的因果机制。support/query、task 和 seed 分开汇报；已有 baseline `7/120`、jitter `6/120` 的闭环诊断不因离线误差变化而改写。

本轮只打开两个已授权 source HDF5 的动作和四类公开 Policy observation；不读取 `states`、`robot_states`、`init_state`、official evaluation states、qualification、validation、held-out 或 final evaluation。

这里的“专家”指原 HDF5 示教记录，本轮没有重新确认每条示教在当前 runtime 中的 replay 成功资格，也没有为 candidate 自由运行状态获得专家动作标签。

## 结果：各任务和数据角色分别汇报

表内先在各角色的全部有效步上计算逐维 MSE，再对三个 seed 取均值。平移/旋转分别按 3 维平均，夹爪按 1 维平均。均值/中位数、MAE/RMSE、逐维误差及每条示教结果均保存在 JSON。

| task | role | arm | 平移 MSE | 旋转 MSE | 夹爪 MSE | 七维 MSE |
|---|---|---|---:|---:|---:|---:|
| 4 | support | baseline | `0.036519` | `0.002446` | `0.127169` | `0.034866` |
| 4 | support | jitter | `0.035343` | `0.002411` | `0.125568` | `0.034118` |
| 4 | query | baseline | `0.040584` | `0.002516` | `0.140425` | `0.038532` |
| 4 | query | jitter | `0.038891` | `0.002473` | `0.136764` | `0.037265` |
| 7 | support | baseline | `0.036578` | `0.001849` | `0.155026` | `0.038615` |
| 7 | support | jitter | `0.036715` | `0.001868` | `0.153814` | `0.038509` |
| 7 | query | baseline | `0.040363` | `0.002142` | `0.172075` | `0.042798` |
| 7 | query | jitter | `0.039981` | `0.002176` | `0.174854` | `0.043046` |

这补齐了此前“同观察 Policy 输出不同，但不知道哪个更接近专家”的缺口：在相同专家输入上，jitter 对不同任务和动作分量有不同影响，不能笼统判断它改善了控制。

### Query 配对 seed 差值

以下为 `jitter − baseline`，负值表示该离线指标下降。

| task | seed | 平移 MSE 差 | 旋转 MSE 差 | 夹爪 MSE 差 | 七维 MSE 差 |
|---|---:|---:|---:|---:|---:|
| 4 | 20261022 | `-0.004116` | `-0.000142` | `-0.006578` | `-0.002765` |
| 4 | 20261023 | `-0.000496` | `-0.000081` | `-0.000797` | `-0.000361` |
| 4 | 20261024 | `-0.000467` | `+0.000095` | `-0.003610` | `-0.000675` |
| 7 | 20261022 | `-0.000213` | `+0.000026` | `+0.004848` | `+0.000613` |
| 7 | 20261023 | `-0.000383` | `+0.000051` | `-0.002643` | `-0.000520` |
| 7 | 20261024 | `-0.000550` | `+0.000025` | `+0.006133` | `+0.000651` |

task 4 的 query 平移/夹爪 MSE 均为 `3/3` seed 下降，但旋转不是；task 7 平移为 `3/3` 下降，夹爪 `2/3` 上升，旋转 `3/3` 上升。这些是描述性配对差值，没有显著性或因果结论，也不改变已有闭环 baseline `7/120`、jitter `6/120` 的结果。

### 专家命令切换窗口

| task / role | 整体夹爪 MSE baseline / jitter | close 切换 ±5 步 MSE | open 切换 ±5 步 MSE | open 切换窗口夹爪 sign error |
|---|---:|---:|---:|---:|
| 4 support | `0.127169 / 0.125568` | `0.611407 / 0.610167` | `0.931612 / 0.940594` | `33.53% / 35.54%` |
| 4 query | `0.140425 / 0.136764` | `0.635286 / 0.630420` | `0.981996 / 0.988784` | `35.05% / 36.14%` |
| 7 support | `0.155026 / 0.153814` | `0.599793 / 0.603981` | `1.010128 / 0.985909` | `36.49% / 35.05%` |
| 7 query | `0.172075 / 0.174854` | `0.545213 / 0.538006` | `1.091029 / 1.083474` | `40.29% / 38.55%` |

窗口有效步数分别为：task 4 support close/open `341/333`、query `231/214`；task 7 support `374/349`、query `264/230`。不是只在 seed 均值上成立：**24/24** 个 task × role × seed × arm 组合都满足 `open 切换夹爪 MSE > close 切换夹爪 MSE > 整体夹爪 MSE`。

baseline query 整体夹爪 sign error 为 task 4 `4.82%`、task 7 `5.98%`，明显小于切换窗口内的错误率。首次可评分动作 `t=1` 的夹爪 sign error 在所有模型中为 0；该样本并不是被排除的 `t=0`，也不是自由运行 candidate 的首步。没有证据支持“在专家状态下完全不会 close/open”。

两种错误方向也有取舍：task 4 query 的 jitter 漏 close 从 `2.97%` 降至 `2.59%`，但专家 open 时误发 close 从 `7.39%` 升至 `7.66%`；task 7 则分别从 `3.35%` 升至 `3.72%`、从 `11.98%` 降至 `10.83%`。整体 MSE 或整体 sign error 不能替代这两种条件错误。

在全角色的七维 MSE 中，夹爪虽只有一维，仍占 baseline query 平方误差约 task 4 `52%`、task 7 `57%`；平移也占较大部分。因此不能仅凭夹爪逐维 MSE 较高就再次全局放大它。这里的百分比来自可加的平方误差代数分解，不是对失败机制的因果贡献。

专家 open/close 是命令标签。open 切换窗口两侧同时包含 open 前后的命令，不能将表中错误率直接称为环境中的“release 失败率”，也不能认定它是低闭环成功率的唯一原因。

## 训练修改决定

保留原 baseline。当前结果不支持把 public-state jitter 升为默认，也不支持增加 updates、再次全局加大 gripper 权重或改动视觉输入。结合三 seed、两个 task、support/query 的一致误差窗口，下一项只检验一个局部损失候选：

> 当专家 gripper action 从非负变为负时，把该切换前后各 5 步的 gripper MSE 权重设为 2，其余时间和其他六维仍为 1。覆盖每条 demo 内全部该类切换，重叠窗口取并集，不重复累加权重。分母继续使用原 `7 × valid_step_count`。

候选设计已记录在 [open-transition gripper weighting proposal](../configs/libero-task4-task7-open-transition-gripper-weighting-proposal_v1.json)。它是**待执行候选设计**，不是完整可执行训练协议或已完成实验，`execution_enabled=false`。权重 2 为一次固定诊断预设，不是本轮估计的最优值，不搜索多个系数。

这与历史 [phase/gripper loss 诊断](libero-phase-gripper-loss-diagnostic-2026-10-25.md) 中采用 uniform phase 的全局 6× gripper 臂不同；旧实现分母为 `12×valid_steps`，没有叠加 phase-balanced（此前文字描述已按代码更正）。旧臂伴随平移误差上升，协议和模型也不同，不能混合统计或当作本轮的因果结果。局部候选仍可能损害平移拟合，必须配对验证。

下一步需把该局部 loss 接入独立 runner 并冻结执行配置：task 4/7 × 三 seed × baseline/新候选，均为 500 updates；保持初始化、data order、完整 causal BC 数据、优化器、CNN-only 观测和全部其他设置不变，不使用 jitter。baseline 重训必须匹配既有 500-update 权重 hash。权重标签仅由 meta-support 的完整动作序列在切窗前计算，不跨 demo，不作为部署时输入，`t=0` 仍按原真实初帧协议训练且权重为 1。

在新训练前明确 source 首帧重建预算、checkpoint 规则、非回归与停止规则，并检查 GPU 占用；后续 candidate 闭环评价仍需保留全部状态顺序和原 reset 历史，不能只挑成功案例。先检验局部误差是否下降以及平移是否退化，再做有界闭环诊断；不因离线指标下降就宣称修复成功。当前仅作出候选选择，**尚未训练新臂或重新评价成功率**。

后续执行更新：用户授权后，该候选已使用新的冻结执行配置完成 12 模型 × 500 updates；六个 baseline 精确复现，但局部收益未覆盖所有 seed，且整体误差非回归失败，已按规则停止该候选分支。没有继续闭环或权重搜索；原设计稿及本报告的离线结果均未改写。实际预算、停止判断和完整 provenance 见 [局部损失配对诊断](libero-task4-task7-local-gripper-loss-diagnostic-2026-11-01.md)。

## 实际预算、provenance 与验证

- 实际覆盖：100 条 source 示教、14,490 个 `t>=1` 专家输入、86,940 行模型预测，另做 768 行 CPU 重复前向检查；全部通过。新增训练/环境交互均为 0。
- CPU 单线程、batch 32、seed `20261022`，elapsed **`219.16s`**；Python `3.10.22`、PyTorch `2.4.1+cu124`、NumPy `1.22.4`、h5py `3.11.0`。这是固定模型离线前向的耗时，不是 500-update 训练耗时。
- 配置 SHA-256：`9051738cf37d7cc6f3b1fd09a01e93755ff314cda2cb719e83db98f50ff95cef`。
- 脚本 SHA-256：`bd44981057bbb09ecc485379ca4ad13563b3f770d892dbd9c73ce3e1af557425`。
- 结果：`/tmp/libero-task4-task7-expert-action-error-20261101-v1.json`，SHA-256 `11bed14d3624144553d11e9395b66ee4621e69b5284e740c75117a60e7d7273e`。
- 四个预测/目标 archive：`/tmp/libero-task4-task7-expert-action-error-archive-20261101-v1/task{4,7}-{meta_support,meta_query}.npz`；各自 file hash、数组形状/类型与逐批输入摘要在结果内。RGB、checkpoint 和数据集未复制入仓库。
- 独立复核：`/tmp/libero-task4-task7-expert-action-error-independent-audit-20261101-v1.json`，SHA-256 `8a2e6ed375680c88b2fabcc0d7b165096f87e806ef20d7ac42b188f39cbf3bf5`。
- 日志：`/tmp/libero-task4-task7-expert-action-error-20261101-v1.log`，SHA-256 `4b01c40893a630121e418cf15f16670441805a8fb0a35699fa2631ffcfde8ffe`。
- 独立重读允许的 HDF5 动作/公开观察，精确核对 100 条 demo 的目标索引、masks、505 个共享输入批次 hash 和 12 个 checkpoint file hash；重算 168 个模型×命令分层单元及三 seed 的均值/中位数/配对差值。目标、索引、mask 和输入摘要全部精确匹配，浮点度量用独立求和次序复算的最大差为 `2.10e-15`，只属算术舍入，不放宽数据时序检查。MSE 分量恒等式最大残差 `5.55e-17`。
- 新专家动作误差测试 **`21 passed`**；连同之前相关诊断共 **`136 passed`**。`py_compile`、环境检查与 `git diff --check` 通过。未运行含可选 Crafter 依赖的全量 pytest；本轮未修改共享数据访问、Policy、环境或训练实现。
- 保存 11 个 pinned 输入、12 个 checkpoint file/weight hash、两份数据文件 SHA、起止时间、Git HEAD/工作区状态、原始预测和实际预算。历史结果及权重均未覆盖或删除。

执行命令：

```bash
nice -n 10 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  conda run --no-capture-output -n libero-kecrl python \
  experiments/analyze_libero_task4_task7_expert_action_error_diagnostic.py \
  > /tmp/libero-task4-task7-expert-action-error-20261101-v1.log 2>&1
```

再次执行需指定新的 output/archive 路径，不能覆盖本轮 artifact。结果仍是 diagnostic，不是正式 SPI、SPT 更新、Knowledge Evolution 或 Module qualification。
