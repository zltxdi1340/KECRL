# LIBERO 有界夜间诊断与晨间汇总（2026-11-02）

用户授权睡眠期间按已有诊断和后续安排继续推进，早上查看设置、结果与结论。本项仍为 diagnostic，保留 500-update baseline，不访问 qualification、reserved、validation、held-out 或 final evaluation，不更新 SPT/Knowledge、不注册 Module。

## 晨间可直接查看的结论

**完成五项有界诊断及独立复核，没有得到足以启动本轮候选训练的完整证据。** 这次没有新训练或 rollout，是预设机制门槛失败后的停止决定，不是等待授权或训练时间不足。历史 baseline `7/120`、jitter `6/120` 保持原记录，本夜没有新增成功率。

| 完成的检查 | 得到什么 | 反证／判断边界 | 当前决定与证据 |
|---|---|---|---|
| 跨示教条件可分性 H2 | open 窗口的固定近邻中仍有命令混合 | fusion→projection 没有三 seed 一致的预设下降；编码器已见过全部 support | 不指定 projection 为根因；[详细结果](libero-task4-task7-condition-separability-diagnostic-2026-11-02.md) |
| 同一前向的 tanh 饱和分解 | 错误符号中强 squash 抑制比例 task 4 为 1.54%–2.41%，task 7 为 8.33%–12.40% | 全部低于事先冻结的 20% 条件，近邻优于 Policy 的条件也未在三 seed 同时成立 | 停止 BCE 条件训练及其闭环分支；不是宣布所有 BCE 无效 |
| 专家观察上的命令时序 | matched open 命令中位数迟 1.5–5 步，六模型最低误差 lag 均为负 | 边界删失较多；命令不是物理 release；不改变 causal 对齐 | 保留偏迟线索；[完整事件与对照](libero-task4-task7-expert-command-timing-diagnostic-2026-11-02.md) |
| 已有闭环输入覆盖 H3 | 相对 source 前 20 步，初始差异已在第一个动作前出现 | task 4 换全时序 source 参照后初始超线比例只有 0/0.05/0；runtime t0 bank 的规模/时相/稳定化有混杂 | 不能称全局 OOD 或统一 drift 根因；[覆盖结果](libero-task4-task7-cached-input-coverage-diagnostic-2026-11-02.md) |
| source 有标签首步控制 | task 4 两 seed 首步平移误差偏高，task 7 两 seed 首步优于全时序平均 | 两任务均未通过三 seed 一致 1.5× 条件；source 标签不能评价 candidate 动作 | 不启动首步加权；[首步结果](libero-task4-task7-source-t0-target-error-diagnostic-2026-11-02.md) |
| 两维公开夹爪输入替换 | 固定首个已观测夹爪状态后，open MSE 三 seed 均值下降 task 4 4.68%、task 7 2.10% | 每个 task 各有一个 seed 的 sign error 增加；全零输入的 MSE 六模型均变差；混合观察可能不真实 | 不直接删输入或训练；[替换、导出恢复及结果](libero-task4-task7-gripper-feedback-diagnostic-2026-11-02.md) |

最有用的新收束是：**open 命令偏迟值得保留，但“tanh 强饱和”“统一 projection 区分下降”“去掉公开夹爪输入即可修复”均没有通过本轮完整判断条件。初始覆盖差异必须按时相和参照库解释。** 这些局部诊断没有唯一确定低闭环成功率的原因，也没有完成 SPT A/B/C、FOMAML 或 Knowledge Evolution。

总边界先冻结在[夜间计划](../configs/libero-task4-task7-overnight-diagnostic-plan_20261102_v1.json)。每一项计算/训练/闭环仍需自己的完整配置、来源 hash、测试和停止规则；总计划不是直接跳过这些门槛的运行入口。

## 固定顺序与最大预算

1. H2：六个 baseline500，source-support `t>=1` 的 8,630 个独立输入。按示教留一，比较视觉、fusion、逐帧 projection 与 proprio 的固定 5NN 命令标签一致性；不让同 demo 相邻帧成为近邻，不读取 query dataset。
2. 同一前向记录中的夹爪饱和/纠错梯度诊断：比较错误符号、切换窗口与 `1-a²`，解析计算 squash 对梯度的影响，不调用 optimizer。既有轨迹中的动作饱和比例不等于有专家目标的错误输出梯度诊断。
3. 同一缓存中的专家命令切换时序分解：固定窗口与 lag，检查先/后偏移和缺失切换，不更改 causal 数据对齐，不用未来标签训练。
4. 仅当 source-support 证据达到预先冻结的机制条件，才允许一次新变量：gripper 的 logit BCE 与原 MSE 对照；其他六维、容量、数据、初始化/批次顺序及 500 updates 不变。最多两任务×三 seed×两臂，合计 6,000 updates。公共 source 首帧只用已保存且 hash 固定的 cache，不重建初态。
5. 仅当新的离线候选通过另行冻结的全 seed 改善/非回归门槛，才准备 candidate-only 闭环；最多 120 episodes、36,000 Policy actions、600 stabilization actions，保留 task/seed 内 reset `0..19` 顺序与原环境 seed。既有 candidate 已反复使用，不称 unseen 或 qualification。

不搜索多个损失系数，不追加 updates，不按 query 选 checkpoint/seed。训练只用 support；每任务全部六个 support fit 完成后才创建 query dataset；任何前置门槛失败即停止该分支。GPU 忙时不抢占其他作业。若没有新训练依据，则完成只读阶段并明确报告证据不足，不为赶进度强行运行候选。

## 运行记录

开始时已核对 git 与 `AGENTS.md`，保留全部既有未提交工作以及用户的 `.vscode/`、`AGENTS.md`。上轮 H1 主指标未形成一致整体冲突，task 7 projection 的次级负向线索保留。本轮首先实现 H2，后续实际门槛、成本、结果与下一决定在此逐项补记。

H2 已完成并独立复核：两个任务均无三 seed 一致的 projection agreement 下降；错误符号的强 squash 抑制比例 task 4 为 1.54%–2.41%，task 7 为 8.33%–12.40%，均不足以通过事先冻结的 20% 条件。BCE 训练与其闭环分支停止，不改变阈值，不另搜系数。H2 主分析 900 次前向加 6 次重复、elapsed 95.09s；独立复核 6.49s，新增学习/交互为 0，详见[条件可分性诊断](libero-task4-task7-condition-separability-diagnostic-2026-11-02.md)。

后续计算预算另冻结为[夜间计划 v2](../configs/libero-task4-task7-overnight-diagnostic-plan_20261102_v2.json)，原 v1 不改写。继续做零前向的命令时序分解，并利用已存在且 `120/120` 前缀精确匹配的闭环 archive 做输入覆盖诊断。增加的只是至多 90 次固定模型前向（2,580 主行加 192 重复行），使用现有 candidate 2,400 帧和 source-support 公共 t0 cache；新训练、初态文件读取和环境交互均为 0，不恢复已失败的训练分支。

## 在读结果前冻结的命令时序分解

[时序配置](../configs/libero-task4-task7-expert-command-timing-diagnostic_v1.json)与[脚本](../experiments/analyze_libero_task4_task7_expert_command_timing.py)只读取 H2 的 source-support 预测缓存，不构造模型或读取 HDF5。固定 lag `-5,-2,-1,0,1,2,5`，对每条 demo 使用同一内部区间，合计 168,630 个 gripper 标量误差比较。定义是 `prediction[t]` 对 `expert[t+lag]`，负 lag 更合适表示延迟，正 lag 更合适表示超前；最低误差 lag 只作描述，不变更 causal 训练对齐。

专家/预测符号切换在 ±10 步内按同方向、最近且不可重复使用规则匹配。缓存首帧不虚构为 transition，超出可观测边界的窗口标记 censored；offset=`predicted-expert`，正值为迟，负值为早。未匹配仅表示窗口内未找到，不称全程缺失或物理 release 失败。结果另记，不进行显著性或因果推断。

## 增补首步的有标签控制

为防止将覆盖距离直接当动作正确性，另冻结[首步目标误差控制](libero-task4-task7-source-t0-target-error-diagnostic-2026-11-02.md)。只增加读取 60 条 source-support `action[0]` 标签，复用覆盖诊断已有的 180 行公共 t0 预测；不增加模型前向、环境或优化器预算。与 source t1/all-t>=1 分开报告，只用于判断 source 记录上的首步拟合，不冒充 candidate 正确动作或据此自动训练。

## 最后一次有界输入检查

上述首步指标并未形成同任务三 seed 一致的 1.5× 缺口。随后另冻结[预算 v3](../configs/libero-task4-task7-overnight-diagnostic-plan_20261102_v3.json)及[夹爪反馈诊断](libero-task4-task7-gripper-feedback-diagnostic-2026-11-02.md)，将源专家观察上的偏迟命令线索拆到两维公开 gripper 输入。保留缓存视觉/joint，只作原输入、同 demo t1 锚定、全零锚定的固定权重读出，不选最优锚点。

新增计算最多 2,706 次部分 Policy 读出链（77,862 行），不是新的 CNN/完整 Policy 前向；原有 996 次完整前向预算不增。所有学习和环境预算仍为 0，训练与闭环分支保持停止；完成本项后整理全部结果和未决问题，不再展开新候选。

## 实际设置与预算账本

共同设置：task `4,7`；固定三个已有 baseline500（训练 seeds `20261022/20261023/20261024`）；CNN-only、原 uniform 七维 action MSE；不选择 checkpoint、不改模型。只用 source-support demos `0..29` 的输入/标签、旧 source 公共 t0 cache，以及已使用 candidate `0..19` 的公开 archive。训练步数 500 是原 checkpoint 的 provenance，本夜新增 optimizer updates 为 **0**。

| 阶段 | 实际新增计算／读取 | 主程序 elapsed |
|---|---|---:|
| H2 与饱和读出 | 900 主前向＋6 重复；25,890＋192 模型行；720 folds、103,560 近邻行 | 95.0921s |
| 命令时序 | 缓存上 168,630 个 lag 标量比较；0 模型调用 | 0.0275s |
| 输入覆盖 | 84 主前向＋6 重复；2,580＋192 模型行；72 组 bank/scaler | 14.1171s |
| source 首步目标 | 60 条 source `actions[0]`，复用 180 行预测；0 模型调用 | 0.7420s |
| 夹爪反馈 | 2,700 主链＋6 重复；77,670＋192 行，仅 fusion→projection→head | 1.8542s |
| 导出恢复 | 从完整预测缓存重建指标；0 新模型调用 | 0.1818s |

合计 **996 次完整 Policy 前向、28,854 模型行（含重复）**，另有 **2,706 次部分读出链、77,862 行（含重复）**；新增 backward、optimizer updates、环境 reset、Policy actions、stabilization actions 均为 **0**。主分析 elapsed 合计约 `111.83s`，缓存导出恢复和独立复核另计；不能把编程、审计、写报告的墙钟时间算作训练耗时，也不能把有限的固定权重前向耗时解释为 500-update 训练已经收敛。

CPU、`torch_threads=1`，`CUDA_VISIBLE_DEVICES=''`、OMP/OpenBLAS/MKL 均为 1；Python `3.10.22`、PyTorch `2.4.1+cu124`、NumPy `1.22.4`。没有占用 GPU 训练。`scripts/check_environment.py` 的本轮输出保留于 `/tmp/libero-overnight-environment-check-20261102-v1.json`；其中 CUDA available=false 来自本轮主动隐藏 GPU，不是 GPU 不可用结论。

H2 主程序于北京时间 `2026-11-02 01:32:19–01:33:54` 执行；反馈读出于 `02:28:37` 完成，缓存导出恢复于 `02:40:14` 完成。原始 JSON 保留 UTC 时间、实际设备、全部配置/代码/权重 pin；本报告不改写这些运行时间。

## 结果与审计索引

下列均为 `/tmp/` 下的独立 artifact，没有复制 checkpoint、RGB 或预测 NPZ 到仓库。各主结果记录完整输入路径、文件 hash、权重 hash、角色、运行设置与输出清单；细节在对应 dated report 中。

| 文件名 | SHA-256 |
|---|---|
| `libero-task4-task7-condition-separability-20261102-v1.json` | `66682c55850acde665ca09c84ffcffe9ba5f0a0e9ae6c743b2390394ddb2a51e` |
| `libero-task4-task7-expert-command-timing-20261102-v1.json` | `fce047f957e50cd34aa89b917a680ebc7875208aea245b5176b9b4ffd043e90e` |
| `libero-task4-task7-cached-input-coverage-20261102-v1.json` | `0009843ce72f6fa802da4c9538e7ede206074b3201f5cd32d614861c8d056488` |
| `libero-task4-task7-source-t0-target-error-20261102-v1.json` | `2e39434deb173af9af8ab8e9e8c5cbe820ba2d334aa11ca44bba237d367b57cc` |
| **有效** `libero-task4-task7-gripper-feedback-20261102-v2.json` | `437f065755ef47b13a2ab972ee060a46697ab9625e6285d5eb02a9dec678f53e` |
| `libero-task4-task7-condition-separability-independent-audit-20261102-v1.json` | `a698a1b6423a11d4f9b24130c3310929cd7966708b64067d67ed15bb0d397894` |
| `libero-task4-task7-cached-input-coverage-independent-audit-20261102-v1.json` | `373ef8b6576ea611595a490692cbd3447789d3edb12a7c3513818cdc1c1aabe1` |
| `libero-night-last-stages-independent-audit-20261102-v1.json` | `1b98445e7dd3445af32b3e2d7ac125068534cd6171c90999616f51ef7c2b00cc` |

独立审计分别耗时 `6.4882s`、`2.8049s`、`1.0647s`，没有新增模型/学习/环境计算。最后一项重算全部 168,630 个 lag 比较、387 条模型—专家事件记录（含窗口内未匹配项）、126 个目标误差分层及全部判断，标量重算差为 0；另重读同一 60 个 source 首动作，不扩大数据角色。H3 距离算术重算最大差 `1.12e-8`，全部来源与判断通过；原 `120/120` 闭环前缀仍要求精确匹配。

反馈 v1 在模型计算和 NPZ 写完后因 `numpy.bool_` JSON 导出失败，失败文件及原代码快照保留。只修复布尔类型导出，并通过独立恢复配置重建 v2，不重跑模型、不改阈值；详细原执行／修复／恢复 provenance 见[反馈报告](libero-task4-task7-gripper-feedback-diagnostic-2026-11-02.md)。不得引用不完整 v1 JSON 作为有效完成结果。

## 验证与改动范围

新增五个离线诊断脚本、对应冻结配置和测试，以及一个缓存导出恢复脚本/配置/测试；全部位于 `experiments/`、`configs/`、`tests/`。没有改共享 Policy、dataset、adapter、SPT 实现。同步更新本组 dated reports、README、roadmap 与 500-update 后续计划；保留此前未提交工作、`.vscode/` 和用户的 `AGENTS.md`，没有自动提交或推送。

相关专项测试、`py_compile` 与最终 CPU 回归完成。最终回归 **`532 passed in 15.05s`**，命令为：

```bash
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  conda run --no-capture-output -n libero-kecrl python -m pytest -q \
  --ignore-glob='tests/test_crafter_*.py' \
  --ignore=tests/test_spatial_gather_spt.py --ignore=tests/test_smoke.py
```

该环境缺少可选 Crafter 依赖，故不声称全量 pytest 完成；上述排除项和历史 CPU smoke 边界保持显式。研究角色、结果范围与测试依赖问题分开解释，不把它写成 LIBERO 代码失败。

文档本地链接 214 项、四个共享源文件 pin、相关文件空白及 `git diff --check` 已通过。最终来源/代码/文档 hash、工作区快照、实际计算总数与已观察测试结果另记录于 `/tmp/libero-overnight-final-verification-20261102-v1.json`；该清单用于追溯本轮交付，不是额外实验或模型运行。

## 下一步应该做什么

优先准备 **固定 Policy 的 source 稳定化匹配对照**，先分离本轮 runtime_t0 参照的一个已知混杂。采用同一 source 示教初始条件、同样 bank/reference 数量和首帧时相，只比较 stabilization `0` 与 `5` 的公开观察、feature 距离和 Policy 响应；保留 task 4 的原 reset 历史。训练仍为既有 500 updates，不同时改 loss、视觉增强、序列长度或数据权重。

下一项先实现角色限制和复现检查，再冻结独立配置、两臂预算和停止规则；**尚未采集或执行，不属于本夜结果**。拟议上限为两任务×30 source-support demos×两设置，120 个 source reset、300 个固定零动作稳定化步、0 Policy 控制步、0 optimizer updates，六个现有 checkpoint 共享相同公开观察。只允许角色授权的 source 初始条件用于重建，禁止加载 official qualification/validation/held-out 初态；不导出 private simulator trace。stabilization=0 的公共观察若不能精确复现已有 source cache，先停止并处理来源问题。

稳定化后的观察没有新的专家正确动作，不能继续拿原 `action[0]` 当其真值；这里只测观察与模型响应。结果若没有三 seed 一致线索，就保留反证，不恢复当前已停止的 BCE 或夹爪输入候选。后续只有在单一机制形成足够证据后才另冻一次 baseline/候选的 500-update 训练；若要回到 LIBERO SPT A/B/C 主实验，仍需先形成可靠闭环 Policy 并冻结独立资格与统计协议。
