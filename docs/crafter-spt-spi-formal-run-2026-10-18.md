# Crafter SPT→SPI 正式运行状态

建立日期：2026-10-18；更新至 2026-10-19。v1 因跨角色 RGB 碰撞未通过审计；v2 的 E1/E2 CUDA 训练均已完成，当前正在完成审计器勘误后的独立审计与分析。训练完成不等于通过原封存分析器的正式验收。

## v1 冻结方案与失败记录

- 协议：[formal v1 配置](../configs/crafter_gather_spt_transfer_protocol_formal_v1.json)，30 个独立 seed（101--130）。
- E1：wood→coal 直接迁移，最大交互预算 6,846,720 步。
- E2：三个 leave-one-binding-out fold，唯一主比较为每个 seed 内等权平均三折的 B−C 归一化 query-success AUC，最大预算 20,540,160 步。
- E1+E2 合计上限 27,386,880 步。功效规划点为 AUC 差 0.10、seed 级标准差假设 0.20；这是显式规划假设，不是 pilot 方差估计。
- 配置、协议、关键源码及运行依赖由独立 manifest 和 detached SHA-256 封存。E1 manifest SHA-256：`8fb6b93f2f23fa448a3fd72225b19a0e51edf8ad44af8ec0571a3e574bcdff35`；E2 manifest SHA-256：`ab5efd52d3cdeee3dd0c05224176ad0b468a04be1879972789084089164fd3dc`。Git 基线为 `29898dd83b15b9ce40540a6353a3b82de63e05a1`，运行相关文件哈希由 manifest 约束。

## 运行位置

| 实验 | 输出目录 | GPU | 进程 PID | 启动日志 |
| --- | --- | ---: | ---: | --- |
| E1 | `results/crafter_gather_spt_transfer_e1_formal_v1/` | 0 | 37374 | `/tmp/crafter_gather_spt_transfer_e1_formal_v1.log` |
| E2 | `results/crafter_gather_spt_transfer_e2_formal_v1/` | 1 | 37375 | `/tmp/crafter_gather_spt_transfer_e2_formal_v1.log` |

训练现已结束：E1 写出 90 份结果（3 臂 × 30 seed），E2 写出 270 份结果（3 fold × 3 臂 × 30 seed）；训练进程正常退出，GPU 空闲。

E1 独立 comparability audit 已结束，但 `valid=false`、`provenance_valid=false`，共 270 条错误，全部为跨角色初始 RGB 哈希重复：target support/evaluation 90 条、source/target support 60 条、source/evaluation 60 条、source 内 support/query 60 条。规范化场景记录交集为 0；例如 B 臂 seed 101 的 source support/query 有 24 个重复 RGB，source/target support 有 22 个重复 RGB。冻结审计将这类视觉状态复用视为角色隔离失败，因此不得把 E1 作为通过审计的正式效应报告。审计明细在 `results/crafter_gather_spt_transfer_e1_formal_v1/analysis_verified/comparability.json`。

E2 的逐 fold provenance/RGB 审计和 seed-cluster 主分析现已结束，但三折均未通过严格 comparability audit。每折各有 270 条错误，构成为：target support/evaluation 90 条、source/target support 60 条、source/evaluation 60 条、source 内 support/query 60 条；规范化场景记录交集仍为 0，重复来自初始 RGB 哈希。E2 analyzer 返回 `valid=false`、`formal_result=false`、`candidate_statistics.status=invalid_input`，`primary_effect=null`，因此没有可报告的正式 B−C 数值。验证明细在 `results/crafter_gather_spt_transfer_e2_formal_v1/analysis_verified/verification.json`。

因此，本轮 E1 和 E2 的训练账本均已完成，但两项都未通过冻结的角色隔离门槛；本轮不能回答 FOMAML 更新后的 SPT 是否改善新 SPI 学习效率。下一步应修复场景池生成/去重规则，加入“canonical scene + initial RGB”双重跨角色去重，在新协议版本中重新冻结后再重跑；当前原始结果只保留作失败诊断，不用于方法效果结论。

正式结果只回答受控 gather 族中 FOMAML 更新后的 SPT 是否改善新 SPI 学习效率；不代表自然 Crafter Module 资格、完整 KECRL、Knowledge Evolution 或持续任务链已验证。

## v2 修复与重跑

- GitHub 已先推送 v1 失败记录，提交 `1b82804`；修复版本提交 `56c1796` 已推送至 `origin/main`。
- 根因是不同 layout seed 会生成重复材料坐标，导致规范化记录不同但初始 RGB 完全相同；不是 GPU 训练异常。
- v2 保留 v1 生成器供历史结果复现，新增可逆唯一布局索引；材料距离改为 Manhattan 1--5 格，并限制在 Crafter 原生 9×7 世界观察范围内，因此这是单独版本化的任务分布更新。
- v2 使用四个互斥布局分区：source-support 32,768、source-query 32,768、target-support 16,384、evaluation 2,500。完整穷举了所有 84,420 个计划使用布局，canonical scene 与 native initial RGB 均为 84,420 个唯一值，碰撞为 0。
- Runner 启动前校验预检报告与 scene-generator SHA、角色分区容量及 formal manifest；独立 analyzer 还会逐 episode 验证 seed 与角色 schedule 一致，并复建 RGB。任一检查失败即拒绝运行/报告。
- E1/E2 seeds、优化设置、训练交互上限与 primary analysis 保持 v1 冻结值不变；场景分布更改由 `formal_v2` 配置和 manifest 单独封存。训练启动后在本节追加 GPU、PID 与日志信息。
- v2 的第一次进程启动触发了 target block 初始化 seed 被误解析为角色 seed 的保护错误；两边在写出任何 result 前退出，未产生可分析的正式运行。修复已加入回归测试并通过全量回归；失败尝试输出目录和日志保留，retry 使用新的输出目录。
- 修复后的 manifest 已封存并提交：E1 SHA-256 `47a136f6aa537cff06528c99ede01d82264990d5a3de3246697b201817b967e7`，E2 SHA-256 `f15b2c0ab59f23a6920c25e81625eda26d7ac933b14c493965b473d748ed8541`；二者绑定代码提交 `5f3d8ac93526c472b3b8e006fde0f71b7352ba22`。retry 计划分别使用 GPU 0/1 和 `results/crafter_gather_spt_transfer_e1_formal_v2_retry1/`、`results/crafter_gather_spt_transfer_e2_formal_v2_retry1/`。

### 修复后的重跑状态（2026-10-18）

- 修复后 E1/E2 已分别在 GPU 0/1 启动，使用上述 retry 输出目录；当前进程 PID 为 E1 `392561`、E2 `392792`。`nvidia-smi` 已确认两个训练进程分别占用约 0.9 GiB 和 0.8 GiB 显存，GPU 利用率采样约 11% 和 17%。CUDA 确实用于策略计算/优化；Crafter 环境步进主要受 CPU 限制，因此 GPU 利用率不会持续接近满载。
- 首个完成的 A 臂、seed 101 是 coal 目标，E1 与 E2 的逐检查点结果一致：support steps `0/1,024/4,096/8,192/16,384` 对应独立 100-episode 成功数 `7/7/14/18/15`，成功率 `0.07/0.07/0.14/0.18/0.15`。该结果显示这个 seed 在 8,192 步检查点暂时达到最高值，末点略有回落；它只是同一配置的单 seed 早期观测，不是三臂比较或正式效应结论。
- 状态采样时 E1 已写出 2 份 `result.json`、E2 已写出 1 份，两个训练进程仍在运行；目标训练实际完成 `16,384` 个 support interaction steps。过程文件记录的 `formal_result=true` 只表示单个配置运行结果符合结果格式，不能替代全 seed 的 provenance/comparability 审计和主分析。
- 输出与启动日志保留在本机：`/tmp/crafter_gather_spt_transfer_e1_formal_v2_retry1.log`、`/tmp/crafter_gather_spt_transfer_e2_formal_v2_retry1.log`。本节的进程号和文件计数是运行中快照，后续状态以最新进程与结果目录为准；正式结论须待全部计划 seed/arms/folds 完成后再运行独立审计及冻结分析。

### v2 训练完成与审计勘误（2026-10-19）

- retry1 训练已结束：E1 为 90 份 `result.json`（3 臂 × 30 seeds），E2 为 270 份（3 folds × 3 臂 × 30 seeds），均保存在上述独立输出目录。运行记录确认使用 CUDA / RTX 4090；没有因本次审计修复重新训练或选择 checkpoint。
- 账本中的 source/target 训练交互分别为 E1 `3,932,160 / 1,474,560` 步、E2 `11,796,480 / 4,423,680` 步；独立评估分别为 `45,000 / 135,000` episodes，评估交互另计。aggregate 记录的 E1/E2 墙钟分别约 `2.65 / 7.91` 小时；两个作业分卡并行，不能把二者相加当作总等待时间。
- 首轮 v2 审计暴露的是分析器错误：C 臂的 `source_ordinary_block` 是 support/query 的合并摘要，没有独立的角色 schedule；旧分析器却先要求合并摘要具有单独 schedule，再审计其两个 subblocks，产生 `frozen v2 role schedule was not supplied to auditor` 假报错。E1 旧审计的 86,822 条错误全部为该类型。该问题与 v1 已确认的真实 RGB 碰撞不同；旧失败输出继续保留。
- 修正后通过两个 subblocks 分别检查冻结角色 schedule、交互预算、材料绑定、动作 RNG、canonical scene 和初始 RGB；合并摘要仍须满足 episode manifest 拼接、步数、episode 数和角色计数的一致性检查。初次修复已推送至 `2ff7f9f`；`ea9cdc9` 清除了误纳入 Git 的本地工作区文件，保留其本机副本。
- RGB 审计复用原生环境的渲染资源，每条记录仍执行 `configure`、`reset` 和哈希重算。回归测试检查其与新建环境的 RGB 一致，且错误哈希仍被拒绝；C 臂回归改为实际 v2 injective schedule。
- 原封存 manifest 与原 E2 统计实现保持原样。由于修正后的审计器哈希与原封存值不同，原 `verify_formal_manifest` 仍应拒绝当前源码；不能静默替换旧 manifest，或用元数据检查替代完整 RGB 审计。
- 勘误后的分析另行绑定原 manifest、输入 aggregate、完整 RGB 审计及修正代码的 SHA-256，并在临时历史 checkout 中调用原验证器复核原始封存版本。仅允许 comparability 审计器的指定哈希变化，任何训练源码、协议、配置、主统计实现变化均拒绝。入口为 [`analyze_crafter_gather_spt_transfer_erratum.py`](../experiments/analyze_crafter_gather_spt_transfer_erratum.py)。该入口始终输出 `formal_result=false` 和 `post_training_audit_correction`，明确标注事后审计勘误。
- 勘误记录须在查看 B−C 主效应前封存并提交；随后继续使用原冻结的 E2 三折 seed 内等权 AUC、30 个配对 seed 和原 bootstrap 设置。此时不调参、不挑选检查点、不新增训练，也不把结果用于 Module 注册或 Knowledge 更新。
- 本次验证：相关聚焦测试 `43 passed`；全量回归 `461 passed`。原封存版本在独立历史 checkout 中通过 E1/E2 两份 manifest 校验；当前工作区的原验证器仅拒绝已修正的 comparability 文件哈希，没有其他封存文件差异。`scripts/check_environment.py` 确认两张 CUDA 设备可用。
