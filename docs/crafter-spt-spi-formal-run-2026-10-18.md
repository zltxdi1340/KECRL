# Crafter SPT→SPI 正式运行状态

建立日期：2026-10-18；更新至 2026-10-19。v2 的 E1/E2 CUDA 训练、勘误后的完整 RGB 审计及 E2 主分析均已完成。当前实现的 B−C AUC 差为 −0.0774，95% 配对 bootstrap 区间为 [−0.1108, −0.0463]，不支持预期迁移收益；另发现 v2 角色池存在明显几何分布偏差。分析明确标注事后审计勘误，不能冒充原封存分析器直接验收的正式结果，也不代表 SPT/SPI 理论被否定。

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

### 勘误后的完整审计与分析封存

| 审计对象 | arm/seed 运行数 | 初始 RGB 复建数 | 完整审计 | 错误 / 警告 |
| --- | ---: | ---: | --- | ---: |
| E1 wood→coal | 90 | 254,944 | 通过 | 0 / 0 |
| E2 holdout coal | 90 | 232,189 | 通过 | 0 / 0 |
| E2 holdout stone | 90 | 234,557 | 通过 | 0 / 0 |
| E2 holdout wood | 90 | 234,624 | 通过 | 0 / 0 |

四项均为 `reconstruct_scene_and_initial_rgb`，`valid`、`provenance_valid`、`strict_source_scene_pool_equal` 全部为 true，共复建 `956,314` 条记录。此处验证的是逐条 RGB 重建、角色隔离、预算及来源配对，尚未解释方法效果。

修正代码已在 `13a0cd9` 推送。独立[分析勘误记录](../configs/crafter_gather_spt_transfer_analysis_erratum_v1.json)和 detached SHA-256 在查看 B−C 主效应前另行封存；记录文件 SHA-256 为 `0177f0af24b49ea71bd882ef38c8c2a99f0c263103c0804efbef7e0ca57d2706`。它绑定两份原 manifest、全部输入 aggregate、四项完整审计及分析源码。原训练 manifest 未改写，统计实现仍为原冻结文件。

### 勘误后的效应结果

勘误记录在 `4356645` 提交并推送后，才首次计算主效应。以下为原冻结统计方案的重算结果，输出标记为 `post_training_audit_correction`、`formal_result=false`；不是将事后修改伪装成训练前冻结的分析器。E1 是预设次要直接迁移，三个 E2 fold 的单独数值用于描述；唯一主比较仍为 30 个 seed 内先等权平均三个 E2 fold，再计算 B−C。

| 实验 / 目标 | A：初始 SPT 平均 AUC | B：FOMAML SPT 平均 AUC | C：普通预训练平均 AUC | B−C |
| --- | ---: | ---: | ---: | ---: |
| E1 wood→coal | 0.0847 | 0.0658 | 0.0681 | −0.0023 |
| E2 holdout coal | 0.0847 | 0.0581 | 0.1455 | −0.0875 |
| E2 holdout stone | 0.0619 | 0.0223 | 0.1260 | −0.1038 |
| E2 holdout wood | 0.0272 | 0.0132 | 0.0543 | −0.0411 |
| **E2：seed 内三折等权平均** | **0.0579** | **0.0312** | **0.1086** | **−0.0774** |

- 主效应精确值为 `−0.0774479167`；seed 级样本标准差 `0.0926415019`；95% 配对 seed-cluster percentile bootstrap 区间为 `[−0.1108442708, −0.0462565972]`（10,000 resamples，analysis seed 20261015，linear percentile）。30 个 seed 中 21 个 B−C 为负，9 个为正。
- 三折的平均 B−C 均为负。此结果只支持“在当前 v2 场景构造、CNN/context-conditioned FOMAML 实现和冻结预算下，B 的学习曲线差于 C”，不支持预期的 SPT 更新收益；不能据此推广到其他 SPT/SPI 实现、自然 Crafter、Knowledge Evolution 或完整 KECRL。
- 差距在目标适应前已出现。E2 目标 0-step 成功率，A/B/C 在 coal 为 `0.0930/0.0510/0.1393`，stone 为 `0.0840/0.0083/0.0820`，wood 为 `0.0433/0.0037/0.0507`。这将下一步诊断指向来源训练后的初始化和场景分布，同时尚不能断言元梯度、表示、head 或奖励中的哪一项是因果根源。
- E1 各臂 30 个 seed，以及 E2 各臂 90 个 fold/seed 组合，均没有在计划检查点达到候选 query-success 0.8。该检查点阈值本身也不是完整 Module 契约。
- 另用既有优化器检查函数核对全部 360 份运行记录：B 的 template/generator 实际学习率均为 `0.001`，C 为 `0.0003`，inner 为 `0.0003`，均与各自冻结配置一致，没有发现再次被默认值覆盖。该检查不意味着这些超参数已经合适。
- 实际独立评估交互为 E1 `1,367,775` 步、E2 `4,158,608` 步；连同训练，实际总交互为 `27,153,263` 步，低于冻结上限 `27,386,880`。

完整主分析输出：`results/crafter_gather_spt_transfer_e2_formal_v2_retry1/analysis_verified/erratum_analysis_v1.json`，SHA-256 `fa8fae990036d79d59693ee5663a84bc470342680c240a66119144665a6f972e`。复算命令如下；入口拒绝覆盖已有输出，复算时须指定新输出路径。

```bash
python -m experiments.analyze_crafter_gather_spt_transfer_erratum \
  --erratum configs/crafter_gather_spt_transfer_analysis_erratum_v1.json \
  --output results/crafter_gather_spt_transfer_e2_formal_v2_retry1/analysis_verified/erratum_analysis_recheck.json
```

### 新发现的角色几何偏差

在主分析之后进行的只读结构诊断发现：v2 使用材料坐标排列的字典序表，再按连续索引区间划分角色。seed 的置换只发生在各自区间内部，不能消除区间之间的材料位置偏差。因此“完整场景 / 初始 RGB 不重叠”和“几何分布相匹配”是两个不同条件；本次原 comparability gate 只验证了前者。

| 角色池 | 布局数 | 树的唯一位置数 | 树距玩家平均 Manhattan 距离 |
| --- | ---: | ---: | ---: |
| source support | 32,768 | 4 | 4.7129 |
| source query | 32,768 | 4 | 3.8613 |
| target support | 16,384 | 3 | 4.9805 |
| final evaluation | 2,500 | **1** | **5.0000** |

最终评估池的树全部在 `(30,29)`；source-support 与 target-support 的树位置集合交集为 0，位置分布 total variation 为 1。石头也不均衡：前三个池覆盖 50 个位置，最终评估仅覆盖 14 个。朝向计数和各材料位置 / 距离分布已完整记录。

这是已确认的环境分布构造限制，尚未通过干预实验证明它导致 B 的劣势；B/C 仍共享同一冻结任务分布与配对来源池，因此上述数值仍是对该具体设置的有效条件性比较。不能将它解释成已消除分布因素后的通用元学习结论，也不能把这个安静受控 fixture 的失败归因于自然世界中的大量死亡。

诊断脚本：[`audit_crafter_material_gather_role_geometry.py`](../experiments/audit_crafter_material_gather_role_geometry.py)。输出：`results/crafter_gather_spt_transfer_e2_formal_v2_retry1/analysis_verified/role_geometry_diagnostic_v1.json`，SHA-256 `57fb000d83fef2247414671de85340c9a0efcf8087a1074791b3783add239795`。该脚本解码全部冻结角色布局，不读取策略表现、不改写场景、不重训策略。

### 下一步诊断顺序

1. 单独设计 v3 **候选**场景划分：先全局打散布局，再分配互斥角色池；同时检查材料位置、距离、朝向和目标可达步数分布。保留 canonical scene / RGB 跨角色隔离，不能继续按排列字典序连续切分。v2 保留，不能就地修改后合并结果。
2. 在新的训练 / 验证 scenes 和 seeds 上，以小预算重新验证同一 CNN + PPO 的可学性，确认独立验证曲线确有改善。v2 final evaluation 不作调参或 checkpoint 选择集；新分布上的可学性尚未验证，暂不追加正式预算。
3. 在可学性通过后，固定场景和来源交互预算，诊断 B 的 support/query 梯度范数、方向一致性、actor 熵、template/generator 更新量及目标 0-step 初始化；逐项判断是分布偏差、元更新幅度 / 方向还是上下文生成导致收益丢失。这些目前仍是待检验解释。
4. 上述边界稳定后，再冻结新的独立确认方案，复验 SPT 更新是否帮助新 SPI 对应的 Policy 初始化和适应。自然 Crafter 生存、完整 Module 资格和持续任务链仍是后续独立问题。

新增几何诊断的聚焦测试为 `2 passed`；包括全部新代码的最终全量回归为 `463 passed`。本轮所有训练 / 分析进程均已结束；原始结果、旧失败审计、修正审计和 checkpoint 均保留在本机，未纳入 Git。
