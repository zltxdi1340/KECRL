# Knowledge Bank 与 Capability 定义决策

> 日期：2026-09-06  
> 状态：基础定义已确认；完整算法由后续决策补充，实验阈值仍待确定  
> 适用范围：KECRL 第一篇工作

本决策取代 `docs/decisions/26-07.md` 中 Decision 001 关于“LLM 仅作为未来可选知识先验”的限定。项目核心仍是 Knowledge Evolution 与 Skill Evolution；变化仅在于第一篇工作明确使用 LLM 生成初始节点与候选关系，LLM 不参与在线规划和动作决策。

后续已确认 [Knowledge Evolution 完整算法](2026-09-06-knowledge-evolution-algorithm.md)，正式流程见 [PROJECT_BIBLE 第 7 节](../../PROJECT_BIBLE.md#7-knowledge-evolution知识演化)。该补充细化验证调度、证据复用、OR 歧义、关系维护与检索，不推翻本文件的基础定义。

## 1. 研究边界

Knowledge Bank 是可迁移的环境结构知识库，与当前 Agent、Policy 和 Module 的性能无关。它保存环境中状态型 Capability 之间的结构关系，不保存动作策略，也不把“当前 Agent 会不会做”混入环境是否允许该转移。

知识库完善后，可迁移到一个没有任何现成技能的新 Agent，以单独检验结构知识迁移的价值。Skill Library 管理 Agent 如何实现状态转移；Knowledge Bank 管理哪些状态条件能够导致目标状态。

第一篇工作的初始化流程为：

\[
Environment\ Description\xrightarrow{LLM}(V_0,M_0).
\]

LLM 在初始化时提出 Capability Schema 和候选因果图。审核完成后冻结节点集合 \(V_t=V_0\)。在线阶段允许更新已有节点之间的关系，但不提出新节点；无法解释的现象记录为 `knowledge_gap`。在线 Concept Formation 与新节点发现留作后续工作。

## 2. Capability 正式定义

Capability 是参数化、可检测、与 Agent 当前 Policy 或 Module 性能无关，并可作为环境机制启动条件、保持条件或目标的最小环境状态谓词：

\[
c_\theta:\mathcal S\rightarrow\{0,1\}.
\]

Capability 的最小粒度为：能够区分环境机制的可达性或参考时间，并完整连接 SPI 输入与输出的最小参数化状态谓词。

### 2.1 准入标准

一个候选 Capability Schema 必须满足：

1. 状态性：描述当前环境状态，而不是动作、Skill 或历史事件。
2. Agent 无关性：语义不依赖当前 Policy、Module 成功率或训练程度。
3. 可判定性：可由对 Agent 公开的观测或符号接口计算真假。
4. 机制相关性：能够成为机制的启动条件、保持条件或目标。
5. 因果可区分性：不同取值可能改变目标可达性或参考时间。
6. 规范表达性：优先采用参数化 Schema，避免实例、数量和同义词导致节点膨胀。
7. 稳定语义性：名称、参数、作用域、取值域和判定函数明确，跨 episode 含义不变。

“可直接干预”不是节点准入条件。Capability 必须可检测和可进行结构验证，但某些状态可能只能通过合法动作间接改变。

### 2.2 不属于 Capability 的内容

- 动作，例如 `open_door`；
- Skill 或 Agent 能力标签，例如 `can_open_door`；
- 由规划器动态推导的 affordance；
- Policy 内部状态与 Module 性能；
- 不保留当前状态信息的纯历史事件，例如 `has_chopped_tree`；
- 不影响任务可达性或参考时间的微观状态。

持久结果只要表示当前环境状态，可以成为 Capability，例如 `door_state(door)=open`、`bridge_state(bridge)=constructed`。

### 2.3 Schema 与运行时事实

Knowledge Bank 持久化参数化 Schema，例如：

```text
hand_occupancy(agent) = item_or_empty
door_state(door) = {locked, closed, open}
inventory_at_least(agent, resource, quantity)
nearby(agent, object)
```

某一局中的 `hand_occupancy(agent_0)=key_1` 是运行时绑定，不产生新的永久节点。

### 2.4 数量、多值状态与负条件

底层环境状态可以是整数、枚举、实数或关系；Knowledge Bank 中的 Capability 统一返回布尔值。数量要求采用参数化约束：

\[
inventory\_at\_least(agent,wood,2)
=\mathbb 1[quantity(agent,wood)\ge2].
\]

只实例化配方或任务真正需要的阈值；数量蕴含关系由约束系统处理，不保存为因果边。多值状态使用统一变量和值表达，例如 `door_state(door)=unlocked`。

负条件不建立无限扩张的否定节点。“木材不足”由数量比较的真假表示，“门未锁”由枚举值表示，“目标格为空”可规范化为 `cell_occupancy(cell)=empty`。

“手为空”是任务相关且可检测的真实状态，应表示为 `hand_occupancy(agent)=empty`，而不是含糊的 `not_holding(key)`。这能揭示 Skill 拼接时的资源接口冲突：开门后仍持有钥匙时，必须先执行释放资源的转移，才能满足拾取球体的空手前置条件。

## 3. 因果机制结构

基本机制表示为：

\[
M=(P_{start},P_{hold},Y).
\]

- \(P_{start}\)：启动时必须成立，执行后允许被消耗或改变的 Capability 集合；
- \(P_{hold}\)：执行期间必须保持成立的 Capability 集合，可为空；
- \(Y\)：成功后的目标 Capability。

Capability 节点本身没有固定的时间类别；时间角色属于它在具体机制或 SPI 契约中的作用。同一 Capability 可以在一个机制中是目标，在另一个机制中是启动条件或保持条件。

角色优先从对 Agent 公开的规则和 SPT/SPI 执行契约中提出；规则不明确时使用分阶段干预：开始前消融用于检验启动条件，启动后消融用于区分启动条件与保持条件。有无该条件都可达而时间不同，则将其记为成本修饰因素。

若条件只在复合技能的中间阶段生效，优先在清晰的状态转移边界拆分 SPI，而不是引入任意时间区间表达。

单个机制内部的源 Capability 为 AND；指向同一目标的多个机制为 OR。不建立独立 AND/OR 节点。采用稀疏父集假设，优先单源和低阶候选，仅在歧义时扩展组合，不全局枚举所有节点子集。

## 4. 结构统计与确认原则

当前只保留两类核心量：

1. 结构可达性；
2. 达到目标的最低或近似最低参考时间。

硬可达性机制和成本修饰分开存储。例如消融 \(A\) 后 \(Y\) 仍可达，但最低参考时间显著增加，则结论是：

```text
Hard mechanism: B -> Y
Cost modifier: A reduces the minimum time under B
```

不能误记为 `A AND B -> Y`。上述简化以更简单硬机制已在相应范围内获得证据支持为前提；仅消融 A 后仍可达，不能排除另一条 OR 机制，不能直接删条件。具体判读与证据复用见 PROJECT_BIBLE 7.6。具体显著性阈值、区间和每个干预组合的样本量属于实验设置，当前不固定。

## 5. 干预与参考验证

采用两种干预模式：

1. 必要性消融 `do_[0,H](A=0)`：在整个验证窗口中维持赋值，主要检验可达性；
2. 可用性干预 `do_t=0(A=1)`：在初始时提供 Capability，主要测量对参考时间的影响。

优先执行必要性消融；其可达性与参考时间数据可直接复用于相应干预条件下的成本判断，仅在数据不足或需要回答不同干预问题时追加可用性干预。全程禁止 A 与仅初始未拥有 A、后续允许获得 A 的时间差含义不同，不得混用。实验组与对照组使用同一个基础世界，干预只修改目标 Capability，并遵循最小修改原则。无效赋值样本排除；不可避免的副作用必须记录并标记为混杂证据，不能单独用于确认关系。

验证流程为：

```text
同一基础世界
-> Environment Adapter 施加并验证干预
-> 检查最小修改、副作用和混杂
-> 状态/动作层 Reference Planner 搜索实际动作路径
-> 返回 FOUND / PROVEN_UNREACHABLE / UNKNOWN 及参考时间
```

`UNKNOWN` 不是反证。Environment Adapter 先做低成本的赋值合法性和局部约束检查；Reference Planner 使用完整环境状态和规则验证实际可达性并提供最低或近似最低参考时间。当前 Module 性能不参与结构判断。

## 6. 显式规则与信息边界

不能默认所有环境都会向 Agent 提供配方数量、动作语义或持续条件。每个环境必须声明合法信息接口：

- `declared_rule`：正常向玩家或 Agent 公开的配方、教程、对象或动作说明；
- `llm_prior`：LLM 根据公开环境描述提出的预测；
- `empirical_transition`：实际执行、观察和干预产生的证据；
- `oracle_only`：源码、完整状态和底层规则，仅供干预验证、Reference Planner 和实验真值使用。

显式配方可直接生成候选机制，再由实际执行验证。隐藏、不完整、冲突或版本变化的规则才需要边界干预。必须区分资源的前置数量和实际消耗量，后者由成功转移前后的状态差识别。

对 Crafter，第一篇工作决定向 KECRL 公开结构化库存接口，使库存数量成为正常观测的一部分，以聚焦 Knowledge Evolution 与 Skill Evolution。完整语义地图、全局坐标、底层配方表和规划器状态仍与学习端隔离。Crafter 的制作是一步原子动作，不存在持续冶炼过程，因此不能用它单独验证复杂保持条件。

## 7. Knowledge Bank 操作与生命周期

最小操作集合为：

\[
\mathcal O_K=\{AddCandidate,RecordEvidence,Refine,Merge,Retire,Retrieve\}.
\]

认知状态：

```text
candidate -> testing -> confirmed / rejected
confirmed -> testing  # 出现系统性反证
```

生命周期状态：

```text
current
superseded
retired
```

`rejected` 表示证据不支持，不删除记录且不用于正向规划；定义或 Scope 改变时创建新 relation；被更简单或更准确关系替代时标记为 `superseded`，而不是 `rejected`。保留 `derived_from`、`merged_from`、`replaced_by` 和 `status_history`。

## 8. SPT、SPI 与 Capability 的接口

SPT/SPI 参考 Schmidt 的运动图式理论：Skill Family/SPT 保存跨实例共享的不变结构、参数化与更新规则，SPI 是面向具体对象和目标的实例，Policy 执行 SPI，多个 SPI 的经验回流以演化 SPT。

例如：

```text
Skill Family / SPT:
CraftTool(source_resources, workstation) -> has_tool(type)

SPI:
CraftStonePickaxe
```

Capability 是 SPT/SPI 的环境状态输入与输出接口。环境规则决定转换是否可行，Skill 决定 Agent 如何实现。Skill 的接口至少需要显式表达输入状态、输出状态、资源消耗和资源释放，避免各 Module 单独成功但组合失败。

## 9. 尚未确定且不得提前固化的内容

- Knowledge Evolution 的具体预算、调度实现与统计更新公式（完整流程已确认，见后续算法决策）；
- 确认关系所需的样本量、显著性阈值与置信区间；
- SPT/SPI 的最终数学与程序表示、SPT 更新规则及 Policy 训练细节（最小调用协议已确认，见后续接口决策）；
- 最终实验环境组合、基线和指标。

Knowledge Evolution 完整流程、Crafter 木镐示例和 [Knowledge Bank 与 Skill Library 最小接口](2026-09-08-knowledge-bank-skill-library-interface.md) 已确认并写入文档。当前按既定顺序进入实验环境/基线/指标讨论。
