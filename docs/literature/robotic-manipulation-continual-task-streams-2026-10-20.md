# 机器人操纵与持续任务流基准调研

调研日期：2026-10-20
目的：判断哪些机器人操作环境适合检验 KECRL 中的 SPT/SPI、Module、Knowledge Evolution 和持续任务流，而不是把“机械臂”“长任务”或“多任务成功率”直接当作理论证据。

## 一、先给结论

你的判断基本正确，但需要把两个问题分开：

1. Crafter 的强项是长时序依赖、资源前置条件、风险与探索、状态记忆和因果链。它可以作为 Knowledge Evolution 环境；但当前 `wood >= 3` 的自然 RGB 采集实验没有形成有辨识度的 Skill Family，不能作为 SPT/SPI 的主验证环境。
2. 机器人操作更容易构造 SPT/SPI 的实例族：相同的操作结构，换物体、目标、姿态、场景、机器人或组合顺序，就得到不同 SPI。这样才能测试“来源实例经验是否改善未来实例”，而不是只测一个任务是否成功。
3. 现成基准通常只覆盖其中一部分：Continual World 和 LIBERO 直接测持续迁移、遗忘与知识类型迁移；CALVIN、FurnitureBench、RoboCasa 和 BEHAVIOR-1K 测长时程组合或复杂环境；Meta-World、RLBench、ManiSkill3 和 RoboTwin 2.0 更适合技能覆盖、视觉/实体泛化或高吞吐控制。
4. 到目前为止，没有一个现成基准同时冻结了“SPT/SPI 技能族学习 + 在线任务流 + 显式因果知识更新 + Module 资格检查”。因此应选择一个主基准，再补一个严格的 KECRL 协议，而不是宣称某个基准天然验证完整理论。

## 二、如何判定基准是否真的适合 SPT/SPI

根据 [`PROJECT_BIBLE.md`](../../PROJECT_BIBLE.md)，SPT 是预定义 Skill Family 的实例生成器和元学习状态，SPI 是一个具体绑定和状态转移请求下的学习规格，Policy 是控制器，Module 是经过独立资格检查的可执行技能。一个合适的实验基准至少应提供以下结构：

| 检查维度 | 需要看到什么 | 不足时会发生什么 |
|---|---|---|
| 可重复技能族 | 同一操作结构有多个目标/物体/姿态/场景实例 | 只能测单任务策略，不能测 SPT 更新 |
| 新实例评价 | 训练实例和 held-out SPI 有清晰划分 | 训练集记忆会被误认为迁移 |
| 任务流 | 有固定顺序、阶段预算和历史任务重评估 | 多任务并行训练会被误称持续学习 |
| 长时程组合 | 后一个动作需要前一个动作改变的状态 | 5 个独立指令串联不能说明因果链 |
| 环境机制 | 物体状态、接触、开关、容器或可达性改变可被观测/干预 | 成功率无法区分动作模仿与机制理解 |
| 独立资格 | held-out episode 上检查启动、保持、输出和资源约束 | 一个高分 checkpoint 会被直接注册成 Module |
| 成本可控 | 能记录每个 SPI 的交互步数、演示量和 GPU 成本 | 无法比较“学习更快”还是“数据更多” |

这里的“因果”要谨慎使用：改变物体、目标或前置状态后能够调整动作，只能说明行为上的机制泛化；若要主张 Knowledge Evolution，还必须记录干预前后的 Capability、机制证据和更新边界，不能从 Policy 的成功率反推 Knowledge Bank 已正确更新。

## 三、最直接的持续学习基准

### 1. Continual World（CW10/CW20）

**论文与年份：** Wołczyk 等，*Continual World: A Robotic Benchmark for Continual Reinforcement Learning*，NeurIPS 2021。论文与代码：[NeurIPS 页面](https://proceedings.neurips.cc/paper/2021/hash/ef8446f35513a8d6aa2308357a268a7e-Abstract.html)、[官方仓库](https://github.com/awarelab/continual_world)。

**环境和玩法：** 它从 Meta-World 的 Sawyer 单臂操作任务构造固定任务流。常用 CW10 序列为 `hammer-v1`、`push-wall-v1`、`faucet-close-v1`、`push-back-v1`、`stick-pull-v1`、`handle-press-side-v1`、`push-v1`、`shelf-place-v1`、`window-close-v1`、`peg-unplug-side-v1`；CW20 将该序列重复一次。每个阶段有明确的环境步数预算，训练阶段切换后重新评价历史任务。

**测量内容：** 新任务学习曲线、旧任务保持、forward transfer、backward transfer、遗忘，以及第二次遇到同一任务时的恢复速度。它的价值是把“持续任务流”定义得很清楚，而不是只提供一个 MT10 多任务分数。

**SPT/SPI 适配：** 很适合做第一层控制实验：同一操作族内比较更新前 SPT、使用来源 SPI 更新后的 SPT、同预算普通预训练。它不适合直接证明复杂因果知识，因为任务大多是独立短技能，且任务流主要测参数/策略的保留与迁移。需要检查其具体实现的 task head、任务标识和随机化方式，不能把默认算法设置直接当作我们的协议。

### 2. LIBERO

**论文与年份：** Liu 等，*LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning*，NeurIPS 2023；[论文](https://proceedings.neurips.cc/paper_files/paper/2023/hash/8c3c666820ea055a77726d66fc7d447f-Abstract-Datasets_and_Benchmarks.html)、[官方仓库](https://github.com/Lifelong-Robot-Learning/LIBERO)。

**环境和玩法：** Franka 机械臂模拟桌面操作，提供程序化任务生成、人类遥操作示范和固定初始状态。四个套件共 130 个任务：`LIBERO-Spatial`、`LIBERO-Object`、`LIBERO-Goal` 和 `LIBERO-100`。前三个套件分别控制空间关系、物体知识和目标/程序变化；`LIBERO-100` 包含 100 个混合知识任务，并划分为 `LIBERO-90` 预训练和 `LIBERO-10` 下游终身学习测试。官方 README 还明确指出默认是稀疏完成奖励，主要实验以 lifelong imitation learning 为主，而不是直接用稀疏奖励 RL。

**测量内容：** 声明性知识、程序性知识或两者混合的迁移，任务顺序鲁棒性，策略结构和预训练影响。原论文报告顺序微调在 forward transfer 上可以超过一些终身学习方法，并且没有一种视觉编码器在所有知识迁移类型上都最好；朴素监督预训练还可能伤害后续终身学习。

**SPT/SPI 适配：** 目前最容易将“SPT 更新”与“知识类型变化”分开的基准。可以定义例如 `Pick(object, pose)`、`Place(object, receptacle)` 或 `Open(container, handle)` 的操作族，按 Spatial/Object/Goal 组合构造 SPI，再测新的绑定。限制是它默认的示范/模仿学习和固定初态不等于我们当前的在线 PPO；如果采用 LIBERO，应将算法、输入模态、示范量和阶段预算重新冻结，并保留独立 Module qualification。

## 四、长时程和组合任务环境

### 3. CALVIN

**论文与年份：** Mees 等，预印本 2021，IEEE RA-L 2022；[论文](https://arxiv.org/abs/2112.03227)、[官方仓库](https://github.com/mees/calvin)。

**环境和玩法：** Franka Panda 模拟桌面环境，A/B/C/D 四种场景配置，每个环境有约 6 小时遥操作数据。智能体接收视觉/本体感知和自然语言指令，连续完成由 5 条指令组成的长任务链；只有当前子任务完成后才进入下一条指令。常见数据划分是 `D -> D`、`ABC -> D` 和 `ABCD -> D`，用于测试新环境、物体和语言指令上的泛化。论文强调任务序列、动作空间和语言条件比既有视觉语言数据集更复杂。

**适合什么：** 长时程状态衔接、技能组合、失败对后续任务的连锁影响、语言条件的 SPI 绑定。

**不适合直接推出什么：** 标准 CALVIN 主要是长时程模仿和零样本泛化，不是“任务 1 学完、更新 SPT、再学习任务 2”的阶段式持续 RL；连续完成 5 条指令也不自动证明因果知识演化。需要另加阶段任务流、历史任务重测和机制干预。

### 4. FurnitureBench / FurnitureSim

**论文与年份：** Heo 等，*FurnitureBench: Reproducible Real-World Benchmark for Long-Horizon Complex Manipulation*，RSS 2023；[论文](https://arxiv.org/abs/2305.12821)、[项目页](https://clvrai.github.io/furniture-bench/)。

**环境和玩法：** 真实 Franka Panda 家具装配，同时提供 FurnitureSim。任务要求识别和抓取零件、移动、对齐、插接或连接，并处理长时程接触和失败。论文提供 200 多小时预采集数据、5,000 多条示范、可 3D 打印家具模型、环境搭建指南和系统化初始化；这是离线 RL/IL 与真实机器人评测，不是默认在线持续学习基准。

**SPT/SPI 适配：** “抓取—对齐—插入—固定”具有明显程序结构，可定义装配操作族和不同零件绑定；真实接触也能暴露只记动作序列的策略。但任务数量集中、数据和硬件成本高，适合后期真实验证，不适合作为第一阶段大规模 SPT 更新实验。需要先使用 FurnitureSim 证明机制，再以有限真实任务做外部验证。

### 5. RoboCasa（RSS 2024）

**论文与年份：** Nasiriany 等，RSS 2024；[论文与项目](https://roboticsproceedings.org/rss20/p050.html)。

**环境和玩法：** 面向厨房家庭任务的仿真框架，提供可交互家具、电器、150 多类物体和 100 个评测任务。任务含原子操作和组合活动，支持人类示范、自动轨迹以及跨场景/物体评估。

**SPT/SPI 适配：** 适合用较小规模先验证 `open/close`、`pick/place`、`turn/press`、`put-into` 等原子族，再组合为料理或整理任务。相比 RoboCasa365，它更容易控制变量和调试；但论文原生重点是通用机器人任务和泛化，不提供完整的在线 Knowledge Evolution 协议。

### 6. RoboCasa365（ICLR 2026）

**论文与年份：** Nasiriany 等，ICLR 2026；arXiv 首次提交 2026-03-04；[论文](https://arxiv.org/abs/2603.04356)、[会议页面](https://proceedings.iclr.cc/paper_files/paper/2026/hash/a05003fdb1e9562ab0c0a9719ea4de10-Abstract-Conference.html)。

**环境和玩法：** 扩展 RoboCasa 到 365 个厨房日常任务和约 2,500 个程序生成环境，包含 65 个原子任务与 300 个组合任务；组合任务最长可包含多个连续子任务。论文把预训练、目标任务微调、机器人基础模型和 lifelong learning 放在同一框架中，并区分 atomic、composite-seen 和 composite-unseen 任务。其 lifelong 方案按任务复杂度分阶段，完成一个阶段后重新评价此前阶段，用来观察迁移和遗忘。

**SPT/SPI 适配：** 在候选环境中最完整地覆盖“原子技能族 -> 复合任务 -> 新任务泛化 -> 任务流遗忘”。它仍不是自动的 KECRL 证明：任务复杂度、演示规模、基础模型、移动导航和多物体变化很多，必须先固定一个小的 task family 和输入/动作接口，否则无法把收益归因于 SPT。

## 五、广覆盖、高吞吐和跨实体平台

### 7. Meta-World

**论文与年份：** Yu 等，CoRL 2019（PMLR 2020）；[论文](https://proceedings.mlr.press/v100/yu20a.html)、[官方基准说明](https://metaworld.farama.org/benchmark/benchmark_descriptions/)。

**环境和玩法：** 50 个 Sawyer 操作任务。`MT10/MT50` 是同时学习固定任务集合；`ML1` 在一个任务内用 50 个物体/目标位置训练、10 个留出位置测试；`ML10/ML45` 在 10/45 个训练任务上适应 5 个留出任务，物体和目标位置随机且不提供 task ID。

**SPT/SPI 适配：** 很适合先做参数绑定、目标位置和结构相似任务的少样本适应；不等同于持续任务流，因为 MT 设置并不规定先后阶段，ML 设置也不是 SPT 版本更新。Continual World 正是把其中一部分任务重新组织为持续流。

### 8. RLBench

**论文与年份：** James 等，预印本 2019，IEEE RA-L/ICRA 2020；[论文](https://arxiv.org/abs/1909.12271)、[项目页](https://sites.google.com/view/rlbench/)。

**环境和玩法：** CoppeliaSim/PyRep 上的 Franka Panda，100 个手工设计任务，从目标到达、开门到打开烤箱并放入托盘等多阶段任务。提供 RGB、深度、分割和本体感知；任务创建时定义 waypoint，运动规划器可以生成大量示范。

**SPT/SPI 适配：** 任务广度、视觉模态和可扩展演示适合定义操作族、少样本迁移和多任务对照；但默认是任务集合和 few-shot/IL 基准，不是持续任务流。应自行建立任务顺序、适应预算和 holdout SPI。

### 9. ManiSkill3

**论文与年份：** arXiv 2024，RSS 2025；[论文](https://arxiv.org/abs/2410.00425)、[RSS 页面](https://roboticsconference.org/2025/program/papers/21/)。

**环境和玩法：** GPU 并行仿真与渲染，支持接触丰富的物理、多种机器人、点云/体素和视觉输入，覆盖 12 个领域并提供来自运动规划、RL 和遥操作的数百万示范帧。论文报告部分环境可达到 30,000+ FPS，重点是高吞吐和泛化。

**SPT/SPI 适配：** 适合作为自定义任务流的底座：大规模运行不同 SPI、扫描预算曲线、测试视觉和物体泛化。它不规定 Knowledge Bank、SPT 更新或任务顺序，所以所有理论协议都需要由我们明确实现。

### 10. RoboTwin 2.0

**论文与年份：** 2025 预印本；[论文](https://arxiv.org/abs/2506.18088)、[任务目录](https://robotwin-platform.github.io/doc/tasks/)。

**环境和玩法：** 双臂操作、约 50 个任务、多个机器人 embodiment，使用杂乱、光照、背景、桌面高度、语言指令等强域随机化，目标是鲁棒双臂操作和 sim-to-real。它强调跨实体和场景变化，以及自动生成/检查任务数据。

**SPT/SPI 适配：** 适合检验同一任务目标在不同身体能力、物体和场景下是否需要不同策略，尤其可用于测试技能参数化和跨 embodiment 适应；现有协议重点是鲁棒性和数据生成，不是终身任务流或知识演化。

### 11. BEHAVIOR-1K / OmniGibson

**论文与年份：** BEHAVIOR-1K 原始工作发表于 CoRL 2022；本文引用的完整 arXiv 版本提交于 2024；[论文](https://arxiv.org/abs/2403.09227)、[官方概念文档](https://behavior.stanford.edu/getting_started/important_concepts.html)。

**环境和玩法：** 1,000 个日常活动，落在 50 个家庭、花园、餐馆、办公室等场景，超过 9,000 个有丰富物理和语义属性的对象。OmniGibson 支持刚体、可变形物体和液体。任务用 BDDL 和一阶逻辑谓词描述初态与目标状态，常包含多步家庭活动和前置条件。

**SPT/SPI 适配：** 很适合研究复杂任务链、对象状态、前置条件和环境语义；可从 BDDL 的公开状态中构造 Capability 和 Knowledge 证据。问题是变量太多：导航、移动操作、柔体/液体、任务规模和模拟器误差会同时影响结果，难以作为第一篇工作的主因果实验。适合后期综合验证。

### 12. Franka Kitchen / D4RL Kitchen

**论文与年份：** Fu 等，*D4RL: Datasets for Deep Data-Driven Reinforcement Learning*，2020；[D4RL 任务说明](https://github.com/Farama-Foundation/D4RL/wiki/Tasks)、[Minari 数据说明](https://minari.farama.org/v0.5.1/datasets/D4RL/kitchen/)。

**环境和玩法：** 模拟 Franka 机械臂在厨房中操作微波炉、水壶、灯开关和柜门。`kitchen-complete-v0` 的目标是按指定轨迹完成四个子目标；`partial` 和 `mixed` 数据提供只完成部分目标或包含混合行为的离线轨迹。典型输入是关节/物体状态，扩展设置也可加入第三人称和腕部 RGB 观测。

**SPT/SPI 适配：** 它是很好的长时程子技能组合和离线数据基线：可以把 `open microwave`、`move kettle`、`flip light`、`open cabinet` 视为可组合的 SPI。但它的场景和目标配置相对固定，默认不是在线持续任务流，也不测旧任务保持。因此，它能回答“已有原子技能能否按顺序组合”，不能单独回答“SPT 在新任务到来时是否演化”。

### 13. FMB（Functional Manipulation Benchmark）

**论文与年份：** Luo 等，IJRR 2024（预印本 2024）；[论文](https://arxiv.org/abs/2401.08553)、[项目页](https://functional-manipulation-benchmark.github.io/)。

**环境和玩法：** 真实 Franka Panda 操作 3D 打印、程序生成的零件。数据包含 54 个单物体装配对象、3 个装配板和 22,550 条轨迹；使用两个全局视角和两个腕部 RGB-D 视角。单物体任务约 20–40 秒，要求抓取、在固定装置上重新定向/重新抓取、移动到装配板并插入；多物体任务由 4 个互锁零件顺序装配，通常超过 100 秒。

**为什么和 SPT/SPI 特别接近：** FMB 的抓取不是“抓到即可”，而是要求抓取姿态有利于后续重定向和装配。这给出一个可检验的功能依赖：上游 `Grasp` SPI 的输出不只是物体被拿起，还应满足下游 `Reorient` 或 `Insert` 的启动条件。训练对象与测试对象在形状、尺寸和颜色上变化，可以构造同族 holdout SPI。FMB 仍然不是标准持续学习基准，需要自行添加任务顺序、预算和历史任务重测，但它是很好的 SPT/SPI 机制验证和真实物理外部检查环境。

### 14. VLABench

**论文与年份：** Zhang 等，2024 预印本；[论文](https://arxiv.org/abs/2412.18194)、[官方任务页](https://vlabench.github.io/)。

**环境和玩法：** 语言条件机器人操纵基准，包含 100 个任务类别和 2,000 多个物体，任务中有原子与组合类别，并对网格/纹理、空间关系、语义指令、物理规律、常识和长时程推理进行强随机化。它同时评价动作策略和语言/视觉模型。

**SPT/SPI 适配：** 适合研究一个操作族在新物体、新语言表达和新空间关系上的泛化，也适合把常识/物理关系作为 Knowledge 侧干预。但它的原生重点是多任务视觉语言动作学习，不要求任务按时间流到达，不报告传统遗忘。因此若采用，必须自行建立“只看当前任务数据、阶段切换、历史重测”的持续协议，并把语言模型先验作为显式实验因素。

## 六、已有真实持续任务流的实验范式（不是现成统一基准）

### 15. LEGION：真实机器人持续强化学习案例

**论文与年份：** Meng 等，*Preserving and combining knowledge in robotic lifelong reinforcement learning*，Nature Machine Intelligence 2025；[论文](https://www.nature.com/articles/s42256-025-00983-2)。

**实验设置：** 论文用模拟训练和真实 KUKA iiwa + RealSense 验证一个连续任务流。任务按 `reach → push → pick-place → door open → faucet open → drawer close → button press → peg unplug → window open → window close` 顺序到达，每个任务训练 1M 步，并周期性重测全部历史任务；方法报告了 forgetting 和 forward transfer。冻结参数部署时，还展示了通过重组既有知识完成“清理桌面”等长任务。

**对本项目的意义：** 这是一个很接近“SPI 经验累积后组合新任务”的实验范式，说明机械臂任务流可以在仿真和实体机器人上被具体化。但它是某一篇论文的算法/任务设置，不是像 LIBERO 或 Continual World 那样的独立标准基准；论文使用任务编码、语言嵌入和其自身的非参数知识空间，不能直接当作 SPT/FOMAML 的证据。我们可以借鉴任务流、阶段预算、全历史重测和指标，而不照搬其内部知识定义。

### 16. IsCiL：持续模仿学习的技能检索案例

**论文与年份：** Lee 等，NeurIPS 2024；[论文页面](https://proceedings.neurips.cc/paper_files/paper/2024/hash/1f0832859514e53a0e4f229fc9b3a4a2-Abstract-Conference.html)。

**实验设置：** 该工作在 Franka Kitchen 和 Meta-World 上研究持续模仿学习，把示范映射到状态嵌入，增量学习可检索技能，并测试任务适应、样本效率和任务遗忘。它更接近“技能库/检索”方法，而非新的环境。

**对本项目的意义：** 可作为 Module Library 和“经验是否成为可复用技能”的对照文献；但它的 adapter/skill retrieval 与当前 SPT 生成初始化、SPI 适应和独立资格规则不同，不能混为同一算法。

## 七、把“长时程”“持续学习”“知识演化”区分开

| 维度 | 真正的实验要求 | 代表基准 |
|---|---|---|
| 长时程执行 | 一个 rollout 内多个子目标，后一步依赖前一步状态 | CALVIN、FurnitureBench、RoboCasa、BEHAVIOR-1K |
| 持续任务流 | 任务分阶段到来，有阶段预算，历史任务重测 | Continual World、LIBERO、RoboCasa365 |
| 技能/策略泛化 | 新目标、物体、姿态、场景或 embodiment 上的同族实例 | Meta-World、LIBERO、RLBench、ManiSkill3、RoboTwin 2.0 |
| 因果机制泛化 | 干预关键状态或物体关系，策略和后续计划按结果改变 | 通常需自定义，BEHAVIOR-1K 的 BDDL/OmniGibson 最容易提供状态基础 |
| Knowledge Evolution | 公开环境证据经验证后更新 Capability 关系，且与 Policy 性能分流 | 没有现成基准直接提供，需要自定义 KECRL 协议 |
| Module 资格 | 独立试验检查启动/保持/输出/资源/范围契约 | 需要在任何基准上新增，不是基准默认指标 |

因此，CALVIN 的“连续完成 5 项指令”不能直接叫持续学习；Meta-World 的 MT50 不能直接叫任务流；RoboCasa365 的 lifelong 阶段结果也不能直接叫 Knowledge Evolution，除非我们单独实现证据更新和任务级边界。

## 八、机器人环境中真正适合定义的 SPT/SPI 例子

### 1. 低层操作族

- `Pickup(object, grasp_region, pose_range)`：抓取不同物体和姿态。
- `Place(object, receptacle, relation)`：将物体放入、放上或对齐到不同容器。
- `Open(container, handle_type)`：拉开抽屉、打开柜门或翻盖。
- `Turn(control, direction, target_state)`：旋钮、阀门或开关。
- `Insert(part, socket, tolerance)`：插入不同零件、孔位和容差。

这些实例共享动作结构，但参数绑定和环境范围不同，最适合检验 SPT 是否让新 SPI 更快形成合格 Module。

### 2. 组合任务族

例如 `PrepareCoffee` 可以由 `Open`、`Pickup`、`Place`、`Turn` 组合；新的厨房任务只改变物体、目标容器或顺序。此时 SPT/SPI 测的是“已学原子技能是否能在新状态转移请求中复用”，而长时程/Knowledge 测的是“前一步是否真的产生了后一步的启动 Capability”。两者必须分开记分。

### 3. 装配族

FurnitureBench 的“抓取—对齐—插接—固定”是连续的技能程序候选。可以把不同家具型号或零件规格作为 SPI 绑定，把前一阶段输出的几何/连接状态作为下一阶段的启动条件，并在真实或仿真环境中检查失败恢复。

## 九、建议的 KECRL 实验协议

### 阶段 A：先验证 SPT/SPI 核心，不加入开放世界复杂性

优先使用 **LIBERO 的一个受控套件** 或 **RoboCasa 的小型原子任务子集**。固定一个操作族，例如 `Pickup` 或 `Open`，建立：

1. 训练 SPI 集合和独立 holdout SPI；
2. A：原始 SPT；
3. B：来源 SPI 经验更新后的 SPT；
4. C：同来源和同预算的普通策略预训练；
5. 三臂使用同一个 Policy 后端、输入模态、适应预算、随机种子、评价初态和环境步数；
6. 记录独立 query 成功率、达到目标的交互步数、Module qualification 通过率、失败类型和 GPU 时间。

B 优于 A 才说明来源实例对共享 SPT 有传递作用；B 优于 C 才说明传递不是普通预训练收益。两者都不成立时，不能直接否定 SPT，先检查实例族是否真的共享可复用结构。

### 阶段 B：加入持续任务流和遗忘

使用 **Continual World 或 LIBERO-90 -> LIBERO-10** 的阶段顺序。每个阶段只训练当前 SPI/任务；在阶段边界重测历史任务和未见 SPI。报告：

- forward transfer；
- backward transfer；
- 历史 Module 保持率；
- 新 SPI 形成所需预算；
- SPT 版本接受/拒绝和回滚；
- 任务顺序变化的敏感性。

### 阶段 C：加入长时程组合

使用 **CALVIN、RoboCasa 或 FurnitureSim**，把原子 Module 组合到复合任务。对每个阶段保存任务开始时的 Knowledge Bank 和 SPT 版本；任务内产生的新证据延迟到边界更新。分别评价：原子 Module 成功率、组合任务成功率、状态衔接成功率和失败后的恢复。

### 阶段 D：加入 Knowledge Evolution 干预

在 **BEHAVIOR-1K 或 RoboCasa** 中选一个小闭环：改变物体可达性、容器状态、插槽匹配或任务前置条件，并用符号/公开状态记录 Capability 变化。要求智能体：

- 对干预后的机制证据更新 Knowledge Bank；
- 让规划层查询到新关系；
- 由 Skill Library 重新请求或适应 SPI；
- 不把一次 Policy 成败直接写成 Knowledge 事实。

这一步才是“操作环境中的 Knowledge Evolution”，不能由普通操纵成功率替代。

## 十、当前推荐选型

| 研究问题 | 首选 | 第二选择 | 原因 |
|---|---|---|---|
| SPT/SPI 同族实例迁移 | FMB、LIBERO-Spatial/Object/Goal 或 RoboCasa 小子集 | Meta-World ML1/ML10 | FMB 有功能性下游依赖；其他设置变量可控，容易做 holdout SPI |
| 持续任务流、遗忘和前向迁移 | Continual World；LIBERO-90 -> LIBERO-10 | RoboCasa365 | 有明确阶段、历史任务重测或终身学习设置 |
| 长时程技能组合 | CALVIN | FurnitureSim、RoboCasa | 子任务连续，状态衔接清晰 |
| 真实接触和错误恢复 | FurnitureBench、FMB | FurnitureSim | 真实装配压力大，但成本和任务范围有限 |
| 语言/常识/物理关系泛化 | VLABench | LIBERO-Plus（若作为扩展） | 需要显式区分语言模型先验和操作策略 |
| 大规模视觉/动作泛化 | ManiSkill3 | RLBench、RoboTwin 2.0 | 高吞吐、丰富任务或跨实体变化 |
| 因果/环境机制综合验证 | BEHAVIOR-1K/OmniGibson | RoboCasa365 | 公开状态、前置条件和复杂对象机制更丰富 |

**推荐最小组合：** 以 LIBERO 或 RoboCasa 小子集先做 SPT/SPI；用 Continual World 做持续学习对照；用 CALVIN 或 FurnitureSim 做组合执行；最后以 BEHAVIOR-1K 做 Knowledge Evolution 的外部验证。RoboCasa365 适合作为规模化后续主环境，不建议一开始就在 365 个任务、移动操作和基础模型上同时展开。

## 十一、对 Crafter 的最终定位

Crafter 不需要被判定为“错误环境”。它更适合作为 Knowledge Evolution 侧的环境：资源、工具、危险和科技树构成状态依赖，适合测能力谓词、机制关系、长时序记忆和探索。当前 `wood` 诊断的问题是任务被压缩成了单一收集行为，而且成功率被 RGB 导航、生存和动作 head 混合影响；这使它不适合作为第一篇 SPT/SPI 技能族迁移的唯一证据。

机器人操作也不是自动的“正确环境”。只有在任务族有多个参数化实例，并设置独立新实例、任务顺序、旧任务重测、机制干预和 Module 契约时，才能分别回答：

- SPT 是否改善未来 SPI 学习效率；
- Policy 是否形成可执行 Module；
- 任务流中是否出现迁移和遗忘；
- Knowledge Bank 是否从公开环境证据中正确演化。

## 十二、参考文献

1. Yu et al. (2019/2020), [Meta-World](https://proceedings.mlr.press/v100/yu20a.html).
2. Wołczyk et al. (2021), [Continual World](https://proceedings.neurips.cc/paper/2021/hash/ef8446f35513a8d6aa2308357a268a7e-Abstract.html).
3. James et al. (2019/2020), [RLBench](https://arxiv.org/abs/1909.12271).
4. Mees et al. (2021/2022), [CALVIN](https://arxiv.org/abs/2112.03227).
5. Liu et al. (2023), [LIBERO](https://proceedings.neurips.cc/paper_files/paper/2023/hash/8c3c666820ea055a77726d66fc7d447f-Abstract-Datasets_and_Benchmarks.html).
6. Heo et al. (2023), [FurnitureBench](https://arxiv.org/abs/2305.12821).
7. Nasiriany et al. (2024), [RoboCasa](https://roboticsproceedings.org/rss20/p050.html).
8. Fu et al. (2020), [D4RL / Franka Kitchen](https://github.com/Farama-Foundation/D4RL/wiki/Tasks).
9. Luo et al. (2024), [FMB](https://arxiv.org/abs/2401.08553).
10. Zhang et al. (2024), [VLABench](https://arxiv.org/abs/2412.18194).
11. Li et al. (2024), [BEHAVIOR-1K / OmniGibson](https://arxiv.org/abs/2403.09227).
12. Tao et al. (2024/2025), [ManiSkill3](https://arxiv.org/abs/2410.00425).
13. RoboTwin team (2025), [RoboTwin 2.0](https://arxiv.org/abs/2506.18088).
14. Meng et al. (2025), [LEGION / robotic lifelong reinforcement learning](https://www.nature.com/articles/s42256-025-00983-2).
15. Lee et al. (2024), [IsCiL](https://proceedings.neurips.cc/paper_files/paper/2024/hash/1f0832859514e53a0e4f229fc9b3a4a2-Abstract-Conference.html).
16. Nasiriany et al. (2026), [RoboCasa365](https://arxiv.org/abs/2603.04356).

文献年份说明：文中优先给正式会议/期刊年份；若论文只有预印本或版本跨年，明确写出预印本和正式发表年份，不将预印本的任务统计冒充另一版本的基准设置。
