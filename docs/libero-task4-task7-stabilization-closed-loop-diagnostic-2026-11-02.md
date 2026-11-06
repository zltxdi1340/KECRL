# LIBERO 固定 Policy 的 stabilization 0/5 闭环成功率对照（2026-11-02）

用户提出直接比较两种设置的闭环成功率，执行优先级因此从 RGB/本体分解调整到本项。前序[source 配对诊断](libero-task4-task7-source-stabilization-control-diagnostic-2026-11-02.md)只回答输入和模型响应是否变化；本项直接检验相同 Policy 在两种 reset 流程下的闭环结果，不重训。

执行更新：device1与原device0的历史完整复现均在同一task7长轨迹停止，旧试验继续保持failed；原device0本次仅新增7条，零步臂未执行。两次当前分叉轨迹彼此精确相同，设备切换未解决历史差异。为完成用户要求，另行冻结[两臂均fresh的当前环境对照](libero-task4-task7-stabilization-contemporary-closed-loop-diagnostic-2026-11-02.md)，不复用本项episode或旧成功率，不把本项失败改判通过。

后续独立对照已完成并审计：两臂各7/120，维持原5步；主比较完整240条均fresh。该结果不改变本报告两个历史完整复现尝试的failed状态。所有额外预算、差异与下一步以新报告为准。

## 执行前冻结的设置与预算

[配置 v2](../configs/libero-task4-task7-stabilization-closed-loop-diagnostic_v2.json) SHA-256 `72fbc906fd6f6aaea2bfed8643d7db39c5f989db1757693816d670565065b34f`；[runner](../experiments/run_libero_task4_task7_stabilization_closed_loop.py) SHA-256 `fed40bd93ef152b00915b56d4317a332f0981fbef2a6f1b1b38f721de5255912`。预运行 v1 没有执行 episode；v2 增加拒收轨迹先落盘和峰值资源记录，数值、状态集、预算及精确门槛不变。资源门槛在新 episode 之前修订为有界推理共享：GPU1 初始 utilization<=50%、已占显存<=4GiB；不训练、不停止其他作业，记录启动快照与实际峰值。

| 项目 | 固定设置 |
|---|---|
| task / 训练 seed | 4、7 / 20261022、20261023、20261024 |
| Policy | 六个既有 CNN-only baseline，500 updates 终点；同一模型用于两臂 |
| 数据与初态 | 只用已反复使用的 candidate_nonregression 0..19，不能称 unseen |
| 两臂 | stabilization=5 与 0；其余环境、相机、权重、动作接口一致 |
| 环境 seed | 20261022，每 task×seed×arm 单独 adapter，state0..19 递增 |
| 时长 | 两臂 native horizon=305、wrapper 最多300个 Policy 动作 |
| 最大 episode | 240：每臂120，每 task×seed×arm 20 |
| 最大 Policy 动作 / 固定零动作 | 72,000 / 600 |
| 新训练 / backward / 示教读取 | 0 / 0 / 0 |
| 设备 | CUDA_VISIBLE_DEVICES=1 的 cuda:0；EGL device1，torch/OMP/BLAS 单线程 |

builder 对两臂均以 max_episode_steps=300、stabilization=5 构造，保持 native horizon=305；随后只改变 reset 的稳定化次数。Policy 动作预算相同，绝对仿真推进次数因五步前缀不同，这是本项干预的一部分。不能把结果解释成纯相机或纯输入置换效应。

## 复现和数据边界

先重跑五步臂全部120条，每条 reset 摘要、逐步 action/公开 joint/gripper 轨迹、success、terminal reason 和长度均须精确匹配原 Full Policy 记录。全部通过后才启动零步臂。任一不匹配即停止，保留已完成 episode，不改容差、不换 checkpoint 或 seed。历史基准为 task4 `3/60`、task7 `4/60`、合计 `7/120`，当前仍需实际复现，不能把它预填成新结果。

不调用普通 `torch.load` 加载整份 initial-state 数组。针对 SHA 固定的 ZIP_STORED NumPy protocol2 格式，先验证 container hash 与193字节元数据头，再仅解码前20×92×8个候选数值字节；后续 qualification/reserved 数值不解码、不做全数组 finite 检查、不传 adapter。完整文件校验涉及原始字节哈希，区别于数值行访问。只在内存使用40个 candidate初态，记录向量 hash，不保存原始值或 live private simulator state。

所有 checkpoint 文件/权重与原 baseline500 hash 核对；不创建 HDF5 dataset，不读 query、validation、held-out 或 final evaluation，不更新 SPT/Knowledge、不注册 Module。

## 比较方式与停止规则

报告 task×seed×arm 的成功数/分母、episode 长度和终止原因。按相同 task/seed/state 配对，完整报告 both_success、zero_only_success、five_only_success、both_failure，以及零步减五步的差值；仅三 seed 全部严格增加才记该任务的一致零步改善线索。不得只汇报净增长、单 seed 或有利初态，不做未经协议允许的显著性推断，也不把重复场景当独立最终测试样本。

完整240条结束后才形成总体配对比较；技术中断则 status=failed，不把部分分母当全组成功率。即使零步更好，也只是本诊断 reset 流程的条件效果，不选择 checkpoint、自动改默认设置或声称 Module qualification。

## 启动前验证与输出

49 项专项及相关测试通过，覆盖候选前缀读入、锁定行的 NaN 不被访问、非法 UTF-8/压缩格式拒绝、精确复现、拒收 episode 保留和双向配对统计。另用一个已有 task4/seed22/state0 的首帧作 CUDA 技术预检，动作逐元素匹配原记录，最大差0；该预检只增加1次已有帧前向、0环境/初态读取，不是新成功率。预检记录为 `/tmp/libero-stabilization-cuda-prefix-preflight-20261102-v1.json`。

本次运行输出（diagnostic）：`/tmp/libero-task4-task7-stabilization-closed-loop-20261102-v2.json`，逐 episode JSON/公开状态和事件日志存对应 artifacts 目录，旧输出拒绝覆盖。运行资源与退出记录分别保存，不复制 checkpoint 到仓库。实际计数、结果、复核与下一决定将在运行后补充。

## 五步复现中断与原设备续跑（结果读出之前冻结）

device1 运行完成67条五步 episode，前66条精确匹配；task7/seed20261022/state6 从第149步出现长轨迹分叉，reset摘要、前148步、终止类型和最终 success 均一致。首个动作差最大 `7.79e-6`，公开 joint 差最大 `7.45e-9`。这不满足原精确门槛，所以 v2 已以 **failed** 保存，零步臂未启动，不能形成总体成功率比较。失败结果 SHA-256 `077e546e63164f39a23ea7771190e1062a084cdf55e3c7469f67cfda46d94f84`；完整失败 episode、日志及所有旧记录保留。当前不能唯一归因于 CUDA 推理或 EGL 渲染。

另冻结[原设备续跑配置](../configs/libero-task4-task7-stabilization-closed-loop-device0-diagnostic_v1.json)，SHA-256 `2bcf92aaeaba19c0e78023859196c916d6dced0c6730bcdc5f950b1822b486cf`；[续跑入口](../experiments/run_libero_task4_task7_stabilization_closed_loop_device0.py) SHA-256 `117eb35775c13fd653fda3b1310a14b72812a87513c8ddc1aeb60c5e219b571e`。改回原 CUDA/EGL device0，不改原权重、状态、horizon、动作预算或精确数值门槛。

只复用 device1 下已经**完整且逐步精确匹配原 device0 记录**的 task4 三个五步组共60条；task7 原尝试7条全部排除主比较，不因其中6条通过而挑选有利部分。新运行在 device0 上完整执行 task7 五步60条，全部精确通过后，再运行两任务零步120条。即使最终比较完成，也须保留 task4 五步复用的 device1 来源及其原 device0 精确等价证据。

新阶段上限180 episodes、54,000 Policy动作、300固定零动作、9个adapter、40个candidate初态数值读取。前阶段实际67 episodes、19,761 Policy动作、335固定零动作。累计最大247次 episode尝试、73,761 Policy动作、635固定零动作、80次candidate初态数值读取（唯一初态仍为40个），训练为0。最终主比较仍为240条，额外7条只作技术复现开销，不能混入成功率分母。资源启动门槛保持 utilization<=50%、已占显存<=4GiB，仅改物理设备0；续跑测试与原组件相关36项通过。

续跑输出 `/tmp/libero-task4-task7-stabilization-closed-loop-device0-20261102-v1.json` 及独立 artifacts 目录。任何新的复现失败继续停止，不放宽容差，不访问 qualification/reserved/validation/held-out，不追加训练。
