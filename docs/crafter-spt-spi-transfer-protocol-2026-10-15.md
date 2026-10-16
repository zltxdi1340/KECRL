# Crafter 采集族 SPT/SPI 迁移实验协议草案

日期：2026-10-15。状态：可审阅草案，未冻结；正式 E1/E2 未运行，非正式集成和修错 pilot 已完成。设计清单见
[JSON 兼容 YAML 配置](../configs/crafter_gather_spt_transfer_protocol_candidate_v1.yaml)。
该文件是实验设计清单，不是已有 runner 的可执行配置。

## 1. 研究问题与本轮路线调整

本协议直接检验：同一采集族的学习经验经过 SPT 更新后，能否改善后续 SPI 的有限预算学习；这种改善是否超出普通预训练。
最近的 wood 表示、动作头和状态覆盖实验提供后端诊断，未检验这一主张。
不再把 wood 成功率达到候选 0.8、所有资源逐项学会或完整生存机制修通作为开展本项研究的前置条件。
此前 wood 等样本量覆盖对照保留为未完成草案，暂不作为默认下一轮实验。

这是 Skill Evolution 的受控组件实验。即使协议冻结后取得正式结果，也不等于证明完整 KECRL、Knowledge Evolution、自然 Crafter 持续任务链或 Module 复用有效。
本协议不替换原先受控离散环境的阶段协议，也不改变 PROJECT_BIBLE 的 SPT/SPI/Module 定义。

## 2. 固定技能族与 SPI

沿用候选任务族标识 `gather`，语义对应理论中的 Mine Template。
固定的是“在声明的启动条件下，通过原生动作增加目标材料库存”的族结构，不在线创建或改变族语义。

| SPI | 材料绑定 | 输出物品 | 原生工具条件 | 本实验统一提供 |
| --- | --- | --- | --- | --- |
| `gather_wood_delta1` | tree | wood | 无 | 木镐 1 把 |
| `gather_stone_delta1` | stone | stone | 木镐至少 1 把 | 木镐 1 把 |
| `gather_coal_delta1` | coal | coal | 木镐至少 1 把 | 木镐 1 把 |

三个 SPI 的绑定包含目标材料、输出物品与增量 1。初始 wood/stone/coal 均为 0；成功以目标物品相对初始库存增加 1 判定。
工具条件是实验契约的已知前提，本项不研究怎样发现该机制或怎样制作工具。
统一工具库存避免工具显示与目标绑定相关；多余木镐不改变原生砍树规则。
地图、朝向和采样 seed 的变化是 episode 变化，不能计为新增 SPI。

每个实例必须形成实际 `SPI` 对象并记录 SPT id/version、参数绑定、状态转移请求、实现契约、程序规格、生成初始化和适应规格。
调用链必须为 `SPT -> Instantiate -> SPI -> support adaptation -> independent evaluation`。
手工加载一个 wood checkpoint 再训练 coal 不能自动标记为完整 SPT 路径。

## 3. 环境与信息边界

候选环境是已实现并用于 pilot 的受控 Crafter fixture，基于本机 Crafter 1.8.3，版本名为 `quiet-material-gather-v1`。
仅 reset、昼夜和实体生成设置受控；移动、朝向、采集、工具检查、资源更新、材质和 RGB 渲染复用原生实现，不修改安装包。

- 每个场景含 tree、stone、coal 各一个，材料位置和初始朝向均变化；目标外的材料作为干扰项。
- 候选局部布局为玩家周围 Manhattan 距离 1--3，三种材料不重叠，目标有可达的相邻可通行格；场景构造器保证条件，不向策略提供坐标或路径。
- 白天、无敌人、无水或熔岩，初始 health/food/drink/energy 均为 9。运行中不重置生命资源，保留原生资源更新。单集最多 32 步。
- 策略逐步输入为原生 64x64 RGB。目标元数据来自公开的 SPI 请求，通过初始化生成器使用；不输入语义地图、坐标、朝向标签、有效动作集合或几何 oracle。
- 固定动作 allowlist 为 0--6，三个 SPI 和所有臂一致；不使用状态依赖 mask、动作替换、冻结 do 概率或监督蒸馏。
- 训练奖励仅为首次目标库存增量的 +1，随后结束 episode；其他步为 0。目标外材料不奖励，不加入距离 shaping、原生成就奖励或死亡惩罚；原生奖励可单独记录但不进入损失。
- 目标库存检测只服务奖励、成功判定与日志，不反向提供推荐动作。所有失败、原生死亡和外部截断均保留；实现错误单列并使受影响运行失效，不静默补成成功。

同一布局必须在不同目标绑定下都能出现，防止“总是采最近或唯一材料”替代目标条件化学习。
留出材料可以作为来源 episode 的视觉干扰项；本实验留出的是目标绑定的训练经验，不是从未见过的像素或新材料概念。
这些受控 episode 不作为自然 Crafter 的 wood>=3 资格结果。

## 4. A/B/C 对照与 SPT 参数化

每个独立 seed 从同一份随机初始化复制三臂。不得从历史 100k wood checkpoint、监督动作头、旧采集数据或旧 optimizer 开始。

| 臂 | 来源阶段 | 后续目标 SPI 初始化 | 解释 |
| --- | --- | --- | --- |
| A：更新前 SPT | 不接触来源训练 | 起始 SPT 生成 | 新增来源经验的参照，来源成本为 0 |
| B：SPT 更新 | 来源 SPI 的 support 内循环、独立 meta-query 和 context-conditioned FOMAML | 锁定候选 SPT 生成 | 检验元更新后的迁移 |
| C：普通预训练 | 相同来源任务、上下文及总交互步，普通 PPO 连续训练 | 相同结构的生成接口导出 | 检验收益是否超出普通预训练 |

候选 backbone 沿用已审计的 CNN-only 结构：通道 16/32、embedding 128、actor 17 输出、critic 1 输出；不加 GRU或此前的监督空间头。
为使普通预训练臂也能表达不同目标的空间动作选择，三臂共享同一生成器容量：

`G(theta, W, z) -> policy clone; actor_parameters = base_actor_parameters + reshape(W @ z)`。

`z` 是按 tree/stone/coal 顺序编码的公开目标 one-hot；`W` 为 actor.weight 与 actor.bias 的完整残差生成矩阵，起始全零。
encoder/critic 初始值来自共享 theta；生成出的 actor 与其他 Policy 参数在 SPI 内共同适应。
不只对 B 增加参数，也不让 C 仅使用无法表达目标差异的固定动作偏置。
B 更新 theta 与 W 的元参数；C 对相同 theta/W 做普通目标条件化 PPO。各臂目标适应时均先生成 Policy，再使用全新的同配置 optimizer，仅更新该 Policy。

在目标绑定留出的一折中，W 的对应列没有来源训练梯度，须保持起始值零；记录并校验。
因此本设计主要检验共享初始化与适应先验，不声称 one-hot 生成器学会了未见目标的语义推理。
若要主张上下文编码器本身的贡献，需要另设等容量消融，不由 A/B/C 自动得出。

## 5. 两个实验及可主张范围

### E1：直接回答 wood -> coal

来源只有 `gather_wood_delta1`，后续目标为 `gather_coal_delta1`；按 A/B/C 比较。
来源阶段多条地图轨迹仍属于同一个 SPI，不把它们包装成多个 SPI。
此项回答单向经验迁移，不能单独支持“多个 SPI 共同演化模板”的一般结论。

### E2：主实验，三个绑定轮流留出

| fold | 来源更新 SPI | 后续留出 SPI |
| --- | --- | --- |
| holdout_coal | wood + stone | coal |
| holdout_stone | wood + coal | stone |
| holdout_wood | stone + coal | wood |

每折从独立封存的起始状态训练，不继承其他折或 E1 的参数、数据、optimizer 或选择结果。
三折计划同时固定，禁止看某折结果后修改其他折。每折内目标绑定不参与来源更新。
三个材质绑定仅支持这个有限基准内的留出迁移结论，不代表大规模技能族或新环境泛化。

E1 的 wood 来源与 E2 的 wood+stone 来源不是等价证据；不能把 E2 的组合收益全部归于砍树。
若进一步追问 wood 的边际贡献，应另行预注册“wood+stone 对比等预算 stone-only”的来源消融，本版不默认扩展。

## 6. 候选预算、更新和成本

下表是供成本核算的候选值，不是已经冻结或已经执行的设置。正式冻结前可根据唯一一次集成/成本 pilot 修订，变更必须形成新版本。

| 项目 | 候选值 |
| --- | --- |
| 独立训练 seed | 正式 101--110，共 10；pilot 11/12，不合并 |
| B/C 每个来源训练总步数 | 65,536，包括 B 的 support 与 meta-query |
| B 来源外循环 | 64 轮；每轮 2 个任务槽，各 256 support + 256 meta-query 步 |
| E2 来源分配 | 每轮两个槽分别对应两个来源 SPI |
| E1 来源分配 | 每轮两个槽都是 wood SPI，使用不同 episode |
| 后续 SPI support 步数 | 16,384，各臂相同 |
| 独立 evaluation checkpoint | 0 / 1,024 / 4,096 / 8,192 / 16,384 support 步 |
| 每 checkpoint evaluation | 100 个配对场景/动作 seed，各臂独立执行 |
| PPO 候选参数 | lr=3e-4，gamma=.99，GAE=.95，clip=.2，value=.5，entropy=.01，4 epochs |
| FOMAML 外循环候选 | Adam lr=1e-3，每轮一次 query 一阶更新 |

每 256 步块可包含多个自然结束的 episode。交互步数精确封顶；目标成功和死亡终止不 bootstrap，时间/块截断按同一规则 bootstrap。GAE 不跨 reset。
B 的每个内循环从生成的全新 Policy 与全新 Adam 状态开始，使用 256 support 步做 4 epoch PPO；在该 adapted Policy 下新收集 256 meta-query 步。
两个任务槽使用相同外循环起点，query 梯度按槽平均后更新候选 theta/W；省略内循环二阶项，但保留生成器到 actor 参数的一阶映射。
query 的 old log-prob、value/return 和梯度必须与实际采样策略一致，不能用未适应策略的轨迹冒充 adapted query。

C 使用同样来源任务槽及场景池，两个 256 步块都可供普通 PPO 学习，持续保留来源 optimizer；不在槽间做 meta reset。
v2 pilot 的 C 使用一个 512 步块，曾与 B 的两个 256 步块在 scene pool 上不一致。v3 校准为 B/C 每个 slot 共享 disjoint support/query candidate pools；C 分别收集两个 256 步块后合并做一次普通 PPO 更新。实际 episode 仍可因成功/截断而不同，manifest 必须区分声明 pool 与实际访问子集。
不强制 B/C 的动作轨迹一致；只匹配来源目标分配、场景分布及总交互成本。
二者优化步骤结构不同，须分别报告梯度次数、wall time、峰值显存与实际步数，不宣称计算成本完全相同。
A 的来源步数为零，不能把 B 对 A 的优势单独当作元算法优势。

后续目标适应三臂统一使用 256 步块和上述 PPO，不调用 meta 更新，不复用来源 optimizer。
evaluation 在 Policy 和 optimizer 冻结、训练 RNG 保存并恢复的副本上运行；评价不能改变后续训练轨迹。
全部方法都运行到最终 support 预算，不按评价结果提前停止或挑最好 checkpoint。

按最大 32 步估计，E2 总交互上限 6,846,720，E1 上限 2,282,240，合计 9,128,960，不含 pilot 和纯复核回放。
其中来源 B/C 成本、目标 support 成本、评价成本、回放成本分别计数；不能只报告目标适应成本而隐藏元训练成本。

## 7. 数据、版本与独立性

正式运行前生成并封存 manifest，至少区分 calibration、source_support、source_meta_query、target_support、final_evaluation。
每条记录包含实验/E1或E2、fold、SPI、replica、角色、预算检查点、scene id、environment seed、action seed 和生成器版本。
同一比较中的 A/B/C 共享 scene id 与动作 seed，并各自独立执行；不能据相同 seed 声称后续轨迹相同。
源 meta-query 参与梯度，属于训练数据，不能同时称为独立最终测试。

按完整 scene/episode 分角色；初始布局的 canonical key 与 RGB digest 用于检查角色之间重复，不能只检查整数 seed。
同一角色内部允许自然状态重复并如实报告；不承诺不同动作链从未经过相同中间画面。
不同臂的配对场景重复是设计要求；不同 replica 共享哪些固定评价场景须写入 manifest，episode 不能充当独立模板训练重复。
已查看的历史 wood 数据只作设计背景，不进入新训练或最终测试。最终评价一旦查看，就不能继续据它选配置或选择模板。

每个来源阶段结束后直接封存固定预算末端候选，不按最终测试挑选版本。
本组件比较不自动切换活动 SPT 指针、不注册 Module、不更新 Knowledge；系统级 SPT 接受和 Module 资格使用各自独立协议。

## 8. 预先指定指标与分析

本 Crafter 组件实验的主指标设为 support-budget query 曲线的归一化梯形面积：

`AUC = sum((b[k+1]-b[k]) * (success[k+1]+success[k])/2) / 16384`。

它在 [0,1] 内，使用全部固定检查点，避免所有臂未达到 0.8 时失去可比较信息。
这是新组件协议的主指标选择，不追溯改写历史实验的主指标。
唯一主比较为 E2 的 B-C：先对每个 seed 的三个 fold 等权平均 AUC 差，再对独立 seed 汇总；禁止把 30 个 fold 当成 30 次完全独立训练。

报告每个 seed/fold 的完整曲线、均值、样本标准差与配对差。
候选区间方法为按 seed 成组 bootstrap 10,000 次的 95% percentile CI，重采样时同 seed 的三折和全部臂共同保留。
10 个 seed 的区间仍有有限样本限制，不宣称已具备特定统计功效；正式样本数在成本/方差核算后冻结，不能因 B 未获胜才追加 seed。
E1 的 B-C、E1/E2 的 B-A/C-A 及各 fold 分项作为预先声明的次指标，不逐项挑选显著结果作为主结论。

次指标包括固定终点成功率、目标外采集率、首次目标增量步数、死亡/截断、契约检查和资源/计算成本。
首次达到候选 query 成功率 0.8 的 support 检查点只作次指标；未达到者按右删失记录，步数不填 0、不删去、也不简单以最终预算冒充真实达标步数。
0.8 不用于选择 epoch、不成为启动实验条件，也不替代独立 Module 资格。

解释约束：B>A 支持来源经验经该更新路径有帮助；B>C 才进一步支持超出本普通预训练基线的收益。
B 与 C 相当或更差属于可报告结果。所有臂无学习信号或起点即饱和时，本设定缺乏区分能力，应在冻结前修订；不能自动认定 SPT 或环境无效。

## 9. 一次集成 pilot 与正式准入

pilot 仅用于查接口、梯度、可学性、成本和方差；不决定方法是否“值得发表”。
候选范围是 E1、两个独立 seed、B/C 各 8,192 来源步、三臂各 4,096 目标 support 步，检查点 0/1,024/4,096，每点 40 个评价 episode。
pilot 数据永不合并进正式结果；明确披露使用 coal pilot 来定预算，因此 coal 是来源训练留出绑定，不是研究者从未接触的任务。
只做这次集成/成本 pilot，不默认接续新的 head/奖励/表示消融。若发现实现错误则修复并记录失效运行；需要改变研究设计时先更新协议版本。

已发现 v1 将 B 声明的外循环学习率 0.001 覆盖为实际 0.0003；旧账本审计未覆盖该项。v2 按原预算修复并记录实际学习率，v2 run2 数值复现一致但严格 scene pool 审计失败。v3 增加 paired support/query candidate pools，场景池审计通过；终点为 A 15/80、B 7/80、C 19/80，仅为两个 seed 的校准信号，不冻结方法优势。详见[动作链与修错报告](crafter-gather-execution-chain-2026-10-15.md)。

进入正式组件实验前应具备：

1. 实际 SPT->SPI->生成->适应路径和 CNN 的 FOMAML 更新实现，且 A/B/C 参数、工具、奖励和预算匹配。
2. 新场景的采集/工具规则、目标外材料负对照及配对 reset 检查；固定动作正对照只作环境审计，不进入训练或方法成绩。
3. candidate/active 隔离、生成器梯度、目标冷列、optimizer 重置、GAE/bootstrap、query 不参与目标训练和评价 RNG 不扰动训练的检查。
4. 短独立进程重复及适当回放审计通过；manifest、代码/依赖 hash、数值设置和所有候选数值冻结。
5. 任务在预定范围内存在可辨别的学习信号、预算与样本数成本可接受。无需 B 胜过 A/C，也无需任何策略先达到 0.8。

这是受控组件的准入条件；完整自然 Crafter 系统实验仍需独立解决其任务链、资格、Knowledge 和 Pipeline 验证。

## 10. 实现核对与当前缺口

| 项目 | 已有依据 | 尚需工作 |
| --- | --- | --- |
| SPT/SPI/Module 对象 | `src/skills/models.py`、`src/skills/spatial_gather_spt.py` | gather 契约、三种材料绑定、实际实例化和训练调用已接通；Module 注册仍留在正式系统协议 |
| 上下文 FOMAML | `src/skills/spatial_gather_spt.py` | 已实现 CNN-only 的完整 actor 残差生成器、adapted query 一阶更新和生成器梯度；早期 MLP 接口不作为本协议后端 |
| CNN/PPO 后端 | `src/algorithms/spatial_crafter_policy.py`、`spatial_gather_spt.py` | 已支持 RGB rollout、episode 边界 GAE/bootstrap、source meta-update 与目标 PPO 适应 |
| episode batch | `SpatialPPORollout` | 已明确 `[N,3,64,64]` RGB 形状、向量对齐和固定 action allowlist；更大规模的 scene manifest 仍需冻结 |
| 受控场景 | `experiments/crafter_material_gather_scene.py` | 三材料、原生工具规则、配对 reset 和 RGB 场景已通过 pilot；不把旧相邻树结果冒充新训练环境 |
| 确定性/回放 | `run_crafter_gather_spt_transfer_pilot.py` | CUDA 确定性设置、RNG 恢复、步数账本和 cold-column 审计已在两个独立 pilot 运行中检查 |
| manifest/计量/分析 | pilot runner、E2 runner、candidate preflight 与[集成记录](crafter-gather-spt-transfer-pilot-2026-10-15.md) | v3 E1 小 pilot、E2 三 fold smoke 和 candidate 成本/provenance 预检已通过；正式 E1/E2 的 hash 封存、统计版本和按 seed 分析仍需冻结 |

当前已完成非正式 E1 wood→coal 集成、动作链审计、学习率修错、paired scene-pool 校准 pilot、独立进程复现、E2 三 fold runner smoke 和 candidate 成本/provenance 预检（seed 11/12 的 E1 pilot；seed 11 的 E2 smoke），结果见[集成记录](crafter-gather-spt-transfer-pilot-2026-10-15.md)。v3 解决了当前 pilot 的 source pool 可比性缺口并通过了复现，E2 runner 和 candidate preflight 已接通但正式 hash、统计版本和启动边界仍未冻结。正式 E1/E2 尚未启动，活动 SPT 指针、Module 和 Knowledge 均未改变。
