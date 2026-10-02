# PROJECT_BIBLE.md

> Version: 1.7
>
> Status: Mathematical Modeling
> 
> This document is the highest-level design specification of the KECRL project. All future algorithms, code implementations, experiments and paper writing must follow the definitions in this document.

---

# 1. Vision（项目愿景）

构建一种能够持续积累世界知识与技能经验的强化学习框架，使智能体不仅能够学习当前任务，而且能够随着环境交互不断演化自身知识，并利用已有经验越来越高效地学习未来任务。

我们的目标不是训练一个能够完成固定任务的 Agent，而是构建一个能够持续成长（Continual Growth）的智能体。

最终，持续学习能力来自于知识和技能模板的不断演化，而不是不断训练新的 Policy。

---

# 2. Scientific Question（科学问题）

How can an embodied continual reinforcement learning agent continuously evolve its world knowledge through interaction, and utilize the evolving knowledge to efficiently acquire and reuse skills across tasks?

中文描述：

具身持续强化学习智能体如何通过环境交互不断演化世界知识，并利用不断演化的知识高效学习、迁移和复用技能，从而实现持续学习？

---

# 3. Core Philosophy（核心思想）

传统持续强化学习主要迁移已经训练好的 Policy。

然而，Policy 通常具有较强的任务依赖性，不容易跨任务迁移，同时容易受到灾难性遗忘的影响。

我们认为：

真正应该持续积累的不是 Policy，而是知识（Knowledge）与技能生成能力（Skill Generation Ability）。

因此，我们提出两个长期演化对象：

- Knowledge Evolution（知识演化）
- Skill Evolution（技能演化）

Knowledge 决定学习什么（What to Learn）。

Skill 决定如何学习（How to Learn）。

两者共同构成持续学习能力。

---

# 4. Core Concepts（核心概念）

## 4.1 Knowledge（知识）

Knowledge 表示与当前 Agent、Policy 和 Module 性能无关的环境结构知识。

Knowledge 不保存具体动作、Policy 或技能成功率，而保存状态型 Capability 之间经过证据支持的因果机制。Knowledge Bank 可迁移给一个尚未掌握任何技能的新 Agent；迁移后，Agent 仍需通过 Skill Library 学会如何实现这些状态转移。

Knowledge 的主要作用：

- 描述环境规律
- 指导技能发现
- 指导技能组合
- 指导持续迁移

Knowledge 以 Knowledge Bank 的形式长期保存，并随着环境交互持续更新。

---

## 4.2 Capability（能力）

Capability 是参数化、可检测、与 Agent 当前 Policy 或 Module 性能无关，并且能够作为环境机制启动条件、保持条件或目标的最小环境状态谓词：

\[
c_\theta:\mathcal S\rightarrow\{0,1\}.
\]

Capability Schema 是 Knowledge Bank 中持久化的节点；某个 episode 中的实体绑定和值属于运行时事实，不产生新的永久节点。

例如：

- `inventory_at_least(agent, wood, 2)`
- `hand_occupancy(agent) = empty`
- `door_state(door) = open`
- `nearby(agent, table)`

数量条件使用参数化阈值谓词，多值状态使用统一状态变量及其取值表示。负条件不建立独立否定节点，而通过数量比较、枚举取值或关系状态表达。

Capability 不包括动作、Skill、affordance、Policy 内部状态、Module 性能或单纯历史事件。Capability 必须可由公开观测或符号接口判定，但不要求能被直接写入环境状态。

Capability 的最小粒度原则是：它应当是能够区分环境机制的可达性或参考时间，并完整连接 SPI 输入与输出的最小参数化状态谓词。

---

## 4.3 Skill Program Template（技能程序模板，SPT）

Skill Program Template（SPT）表示某一预定义 Skill Family 共享的技能实例生成器与元学习器。它不对应具体对象实例，也不是一组具体 Policy 的集合；它保存跨实例共享的固定语义结构与可演化的技能生成状态，用于生成具体 SPI，并利用多个 SPI 的反馈提高后续技能的学习效率。

例如：

Mine Template

Pickup Template

Attack Template

Navigate Template

时刻 \(t\) 的 Skill Family \(f\) 的 SPT 定义为：

\[
T_f^t=\left\langle T_f^{static},\omega_f^t\right\rangle.
\]

其中：

- \(T_f^{static}\) 是固定语义结构，至少规定 Skill Family 标识、参数绑定 Schema、实现契约构造器、程序结构 Schema、实例化接口以及反馈/更新接口；
- \(\omega_f^t\) 是可演化状态，可表示跨实例共享的表征、Policy 初始化先验、目标条件化生成参数、探索先验或适应超参数；其具体参数化形式由 Skill Evolution 算法确定。

当前版本预定义 Skill Family，且不在线创建、合并或改变 SPT 的族语义。因此第一篇工作中：

\[
T_f^{static,t+1}=T_f^{static,t}.
\]

Skill Evolution 主要更新 \(\omega_f^t\)。固定语义结构不表示程序实现永远不可版本化；若人工修改其语义，应建立新版本并保留来源，而不是把在线性能更新伪装成 Skill Family 语义变化。

SPT/SPI 的设计参考 Schmidt 的运动图式理论：多个具体实例的经验回流到共享图式，使后续实例化更高效。SPT 的长期目标是提高未来 SPI 的生成、学习与适应效率，而不是积累越来越多具体 Policy。具体反馈、更新和验证规则见第 9 节。

---

## 4.4 Skill Program Instance（技能程序实例，SPI）

SPI 是 SPT 针对一个具体技能目标、参数绑定和状态转移请求生成的技能程序与学习规格。它说明该具体技能应如何被构造、学习和判定，但不是已经学成的可执行 Module。

例如：

Mine(Wood)

Mine(Coal)

Mine(Iron)

都是 Mine Template 的不同 SPI。

给定 SPT 版本 \(T_f^t\)、参数绑定 \(b_j\) 和状态转移请求 \(q_j\)，实例化过程为：

\[
I_j=Instantiate(T_f^t,b_j,q_j),
\]

其中 SPI 定义为：

\[
I_j=\left\langle
id_j,f,v_f^t,b_j,q_j,\Gamma_j,P_j,\zeta_j^0,A_j
\right\rangle.
\]

各组成部分为：

- \(id_j\)：SPI 标识；
- \(f\)：所属 Skill Family；
- \(v_f^t\)：生成该 SPI 的 SPT 版本；
- \(b_j\)：具体对象与参数绑定；
- \(q_j\)：触发实例化的状态转移请求与合法运行上下文；
- \(\Gamma_j\)：具体实现契约；
- \(P_j\)：实例化后的程序规格，包括观测或特征接口、允许的 Policy 类、训练目标以及成功、失败和终止判定；
- \(\zeta_j^0\)：SPT 为该实例生成的初始化状态；
- \(A_j\)：该实例的学习或适应规格。

程序规格 \(P_j\) 是与实现形式无关的技能规格，不要求采用某一种程序语言或神经网络结构。SPI 不保存最终 Policy、Module 长期性能统计、Module 生命周期或 Knowledge Bank 的机制置信度，也不要求永久保存完整训练轨迹。SPI 是 SPT 与具体 Policy 学习之间的桥梁；其带上下文的学习和执行反馈可以推动对应 SPT 演化。

---

## 4.5 Policy（策略）

Policy 是强化学习真正执行动作的控制器。

Policy 负责：

根据环境状态输出动作。

Policy 不作为持续学习的核心知识保存对象。

Policy 服务于 SPI。

SPI 服务于 SPT。

---

## 4.6 Module（技能模块）

Module 是 Skill Library 中保存和复用具体技能的基本可执行单位。Module 由 SPI 经 Policy 学习或适应并通过 Skill Library 内部资格检查后形成；刚完成实例化但尚未学会的 SPI 不能作为 Module 执行。

Module 定义为：

\[
m_i=\left\langle
id_i,spi_i,\pi_i,\Gamma_i,E_i,\Sigma_i,\mu_i
\right\rangle,
\]

其中：

- \(id_i\)：Module 标识；
- \(spi_i\)：该实现绑定的唯一 SPI；
- \(\pi_i\)：已经形成的可执行 Policy；
- \(\Gamma_i\)：从 SPI 固化到该实现版本的实现契约；
- \(E_i\)：用于技能学习、适应或 SPT 反馈的技能经验；
- \(\Sigma_i\)：由经验派生的 Module 内部统计摘要；
- \(\mu_i\)：来源、版本、生命周期和其他元数据。

一个 SPI 可以因环境范围、观测/动作接口、Policy 版本或资源约束不同而形成多个 Module，但每个 Module 只绑定一个 SPI。Module 与 Policy 不是同义词：Policy 是动作控制器，Module 还包含具体技能语义、实现契约、经验、统计和版本信息。

实现契约定义为：

\[
\Gamma_i=\left\langle
I_i^{start},I_i^{hold},O_i,C_i,R_i,X_i,\Omega_i
\right\rangle,
\]

其中：

- \(I_i^{start}\)：实现启动时要求成立的 Capability；
- \(I_i^{hold}\)：执行期间要求保持成立的 Capability；
- \(O_i\)：Module 声明尝试产生的输出 Capability；
- \(C_i\)：声明的资源消耗；
- \(R_i\)：声明的资源释放；
- \(X_i\)：该实现特有的额外适用约束；
- \(\Omega_i\)：环境、版本、观测接口与动作接口等适用范围。

\(O_i\) 是声明输出而不是环境事实；实际输出只能由 `ExecuteImplementation` 返回的 `TransitionResult` 判定。\(X_i\) 可以包含当前实现特有的限制，且不要求全部成为 Capability Schema；这些限制、Module 性能和普通执行失败都不能自动写入 Knowledge Bank。

Skill Library 必须区分硬资格判断与软性能比较：只有契约、范围、资源和当前执行状态均兼容的 Module 才能成为候选；在候选之间使用 \(\Sigma_i\) 进行性能评分或选择。资格规则见第 9.3 节；具体阈值数值、软评分公式、经验 \(E_i\) 的存储形式和完整 Module 生命周期仍未固定。

---

# 5. System Architecture（系统架构）

整个系统由两个长期记忆组成：

## Knowledge Bank（知识库）

负责保存世界知识。

Knowledge Bank 只持久化状态型 Capability Schema 及其因果机制；affordance 由规划器根据当前状态、机制和环境约束动态推导，不持久化。

基本机制表示为：

\[
M=(P_{start},P_{hold},Y),
\]

其中 `P_start` 是机制启动时必须成立、随后允许被消耗或改变的 Capability 集合；`P_hold` 是执行期间必须保持成立的集合，可为空；`Y` 是成功后的目标 Capability。时间角色属于 Capability 在具体机制中的角色，不属于 Capability 节点本身。

同一机制内的源 Capability 为 AND；指向同一目标的多个机制为 OR，不建立独立 AND/OR 节点。父集采用稀疏假设，优先检验单源和低阶组合，不全局枚举所有节点子集。

Knowledge Bank 同时保存：

- 结构可达性与最低或近似最低参考时间；
- 证据、来源、适用范围和混杂标记；
- 认知状态与生命周期状态；
- 关系的派生、合并和替代历史。

硬可达性机制与成本修饰必须分开。若消融某条件后目标仍可达，但最低参考时间显著增加，该条件是成本修饰因素，不是硬必要条件。

Knowledge Bank 持续演化。

---

## Module Library（技能库）

负责保存已经学习完成的技能。

Module Library 中按照技能族组织。

例如：

Mine

├── SPT

├── Module(Wood)

├── Module(Coal)

├── Module(Iron)

Pickup

├── SPT

├── Module(Key)

├── Module(Box)

Attack

├── SPT

├── Module(Sword)

每个技能族维护一个 Global SPT。

不同 Module 内保存各自对应的 SPI。

---

# 6. Continual Learning Pipeline（持续学习流程）

Continual Learning Pipeline 是已有 Knowledge Bank、Skill Library、Knowledge Evolution、Skill Evolution 与规划层之间的运行时协调流程，不是第三个长期知识库，也不引入新的学习算法。

## 6.1 运行时任务闭环

对每个任务，规划层读取公开观测和目标，将观测映射为运行时 Capability 事实，然后按第 8 节的最小接口循环执行：

~~~text
读取观测与目标
-> RetrieveMechanisms
-> 选择当前可用的机制
-> RequestImplementation
-> ExecuteImplementation
-> 检查 TransitionResult 与实际状态
-> 目标达成则结束；否则继续规划或选择替代机制
~~~

规划层保存当前任务所需的临时上下文和计划，不将环境机制、Module、SPI 或 SPT 复制为自己的长期知识。`unavailable` 表示当前未取得实现，不等于环境不可达；结果不可判定时不当作失败。

## 6.2 双反馈分流

执行及学习产生两类职责不同的反馈：

| 反馈 | 内容与去向 |
|---|---|
| 环境证据 | 公开状态变化、环境范围及合法干预元数据；进入 Knowledge Evolution，并按第 7 节验证与更新 |
| 技能经验 | SPI 上下文、适应反馈和 Module 执行经验；留在 Skill Library，用于 Skill Evolution |

两类信息不得越过既定信息边界。一次成功或失败不会自动确认或否定 Knowledge Bank 中的机制；Policy 学习曲线、梯度和 Module 性能不进入 Knowledge Bank。

## 6.3 演化更新边界

运行时先记录反馈；Knowledge Evolution 和 Skill Evolution 在任务边界或明确的更新边界运行。更新分别受第 7 节和第 9 节规则约束，不要求每次执行后立即完成整轮验证或元更新。

单个任务固定其开始时使用的 Knowledge Bank 与 SPT 版本视图。该任务产生的新知识和新接受的 SPT 版本默认供后续任务使用；更新不原地改写已形成的 Module。此规则用于保持任务内行为可追溯，不要求额外的复杂版本状态机。

## 6.4 任务结果

任务结果保留最小的四类：

- `completed`：目标 Capability 已由实际观测确认；
- `continued`：目标尚未达成，但当前任务可以继续规划和执行；
- `unavailable`：当前没有可用 Module，或 SPI 学习未形成合格 Module；
- `unknown`：关键状态、目标或执行结果无法判定。

`unavailable` 和 `unknown` 均不构成环境不可达的结论。具体任务分段、并行调度、自动重试与规划搜索算法不属于本 Pipeline 的核心定义。

## 6.5 Knowledge Evolution 与 Skill Evolution 联合目标

Knowledge Evolution 与 Skill Evolution 是两类长期演化过程，保持各自的对象、证据来源、更新规则和信息边界。第一篇工作不构造一个强行合并两者的复杂加权损失，而采用“两个局部演化目标 + 一个任务层耦合评价”。

Knowledge Evolution 的局部目标写为：

\[
J_K(K_t)=Q_{\mathrm{structure}}(K_t)-\lambda_K C_K,
\]

其中，\(Q_{\mathrm{structure}}\) 表示知识结构的正确性、可解释性和适用性质量，\(C_K\) 表示知识验证与维护成本。其具体分解不在本节固定。

Skill Evolution 的局部目标写为：

\[
J_S(L_t)=
-\mathbb E_{I\sim\mathcal D_{\mathrm{SPI}}}
\left[\sum_k\alpha_k\mathcal L_I^{query}(\theta_I^k)\right]
+\lambda_Q Q_{\mathrm{module}},
\]

其中，\(L_t\) 表示 Skill Library 及其 SPT 状态，query 损失衡量 SPI 学习效率，\(Q_{\mathrm{module}}\) 表示合格 Module 的质量项。具体权重和资格阈值属于运行或实验设置。

系统层使用任务表现作为两类演化的耦合评价：

\[
J_T(K_t,L_t,\Pi)=
\mathbb E_{\tau}\left[R(\tau)-\lambda_c C(\tau)-\lambda_u U(\tau)\right],
\]

其中，\(\Pi\) 为规划与实现选择过程，\(R\) 为任务结果收益，\(C\) 为任务成本，\(U\) 为不确定性或不可用惩罚。\(J_T\) 主要用于系统级评价和协调，不替代 Knowledge Evolution 或 Skill Evolution 的局部更新算法。

整体设计可表示为：

\[
\max_{\Pi,U_K,U_S}J_T(K_t,L_t,\Pi)
\]

满足：

\[
\mathrm{Integrity}(K_t)\ge\kappa_K,
\qquad
\mathrm{Qualification}(L_t)\ge\kappa_S,
\]

以及独立更新：

\[
K_{t+1}=U_K(K_t,E_t^K;B_t^K),
\qquad
L_{t+1}=U_S(L_t,E_t^S;B_t^S).
\]

其中，\(E_t^K\) 是合法公开环境证据，\(E_t^S\) 是 SPI、Policy 和 Module 技能反馈；\(B_t^K,B_t^S\) 是各自预算。Knowledge Evolution 不直接修改 Policy，Skill 性能不直接写入 Knowledge Bank。

本节只规定对象边界、局部目标与任务层耦合关系，不固定 \(Q_{\mathrm{structure}}\)、任务成本是否进入主评价、\(\kappa_K\) 和 \(\kappa_S\) 的数值，也不要求最终压缩为单一标量目标。

## 6.6 整体 KECRL 数学形式化与有限保证

### 基础环境与可检测 Capability

对任务目标 \(g\)，将基础环境表示为带公开观测接口的折扣决策过程：

\[
\mathcal E_g=(\mathcal S,\mathcal A,\mathcal P,R_g,\gamma,\mathcal O,h),
\]

其中，\(\mathcal S\) 是环境状态空间，\(\mathcal A\) 是动作空间，\(\mathcal P(s'\mid s,a)\) 是状态转移，\(R_g\) 是任务目标相关收益，\(\gamma\) 是折扣因子，\(\mathcal O\) 是对 Agent 公开的观测空间，\(h:\mathcal S\rightarrow\mathcal O\) 是公开观测映射。当 \(h\) 为恒等映射时，该过程退化为完全可观测 MDP；当前框架不要求 Agent 能读取完整环境状态。

Capability Schema 集合为：

\[
V_0=\{c_1,\ldots,c_n\},\qquad
c_\theta:\mathcal S\rightarrow\{0,1\}.
\]

“可检测”要求存在公开判定器 \(\widehat c_\theta\)，使其在声明适用范围内满足：

\[
\widehat c_\theta(h(s))=c_\theta(s).
\]

### 两类长期状态与运行时协调

Knowledge Bank 的最小状态表示为：

\[
K_t=(V_0,\mathcal M_t,\Xi_t),
\]

其中，\(\mathcal M_t\) 是机制集合，\(\Xi_t\) 保存证据、原子命题统计、适用范围和生命周期信息。每个机制仍为：

\[
M_j=(P_{start,j},P_{hold,j},Y_j).
\]

Skill Library 的最小状态表示为：

\[
L_t=(\mathcal T_t,\mathcal I_t,\mathcal U_t),
\]

其中，\(\mathcal T_t\) 是 SPT 及其版本，\(\mathcal I_t\) 是 SPI 与技能反馈，\(\mathcal U_t\) 是合格 Module 及其技能经验和统计摘要。

规划层使用当前公开观测、目标和两库的只读视图进行任务内协调：

\[
\Pi:(o,g,K_t,L_t)\rightarrow(M,q,m),
\]

其中，\(M\) 是选定机制，\(q\) 是状态转移请求，\(m\) 是取得的合格实现。规划器的临时状态不进入 \(K_t\) 或 \(L_t\)，也不形成第三个长期知识库。

在任务或明确更新边界，两类长期状态分别更新：

\[
K_{t+1}=U_K(K_t,E_t^K;B_t^K),
\qquad
L_{t+1}=U_S(L_t,E_t^S;B_t^S).
\]

因此，当前版本的整体 KECRL 定义为：

\[
\mathfrak K=
(\{\mathcal E_g\}_{g\in\mathcal G},V_0,K_t,L_t,\Pi,U_K,U_S).
\]

该定义不把 Policy 参数、梯度、完整训练轨迹或规划器临时状态并入统一超级状态。

### 当前版本可主张的性质

以下性质均是给定接口、适用范围和统计假设下的有限保证，不是全局最优或无条件收敛结论。

1. **Capability Schema 不变性。** 初始化审核后，\(U_K\) 只能在 \(V_0\) 上增删、修正或维护机制，不能创建新 Schema。因此由归纳法有：

   \[
   \forall t\ge 0,\quad V_t=V_0.
   \]

2. **长期记忆边界保持。** 若 \(U_K\) 只接收合法环境证据 \(E_t^K\)，\(U_S\) 只接收技能反馈 \(E_t^S\)，且三项最小接口遵守第 8 节的数据契约，则 Policy 梯度、完整训练轨迹和 Module 性能不会被 \(U_K\) 持久化，环境机制也不会被 \(U_S\) 当作 Policy 参数更新。该性质是类型与接口不变量。

3. **机制调用前提一致性。** 对机制 \(M_j\)，只有在启动时检测到 \(P_{start,j}\) 成立，并能在声明执行区间检查 \(P_{hold,j}\) 时，它才具有结构调用资格。该资格不保证存在合格 Module，也不保证一次执行成功。

4. **条件式契约组合性。** 对机制与 Module 序列，若每个 Module 均通过资格检查、适用范围与资源约束兼容，并且每一步实际观测到的输出满足下一步的启动条件、执行期间满足保持条件，则该序列在契约层可组合；最后一步实际观测到目标 Capability 时，任务目标才视为达成。Module 的声明输出 \(O_i\) 本身不能替代实际观测。

5. **原子命题后验正确性。** 在同一适用范围内，若有效证据可建模为条件独立或可交换的 Bernoulli 观测，且先验为 \(Beta(\alpha_0,\beta_0)\)，则支持数 \(S_{c,t}\) 与反证数 \(C_{c,t}\) 产生共轭后验：

   \[
   p_c\mid D_t\sim
   Beta(\alpha_0+S_{c,t},\beta_0+C_{c,t}).
   \]

   因而第 7 节的确认规则具有条件统计解释；混杂、UNKNOWN 或跨范围证据不满足该保证。

6. **单轮有限预算终止性。** 若 \(B_t^K,B_t^S<\infty\)，每个验证、适应或更新步骤消耗严格为正的预算，且单步计算终止，则每轮 \(U_K\) 与 \(U_S\) 在有限步内结束。未决任务可以跨轮保留；单轮终止不等于知识或技能最终收敛。

### 当前版本不作出的保证

当前版本不声称整体非凸系统达到全局最优，不声称任务性能逐任务单调提升，不保证有限预算内发现全部真实机制，也不保证发现 \(V_0\) 之外的新 Capability。FOMAML 的全局收敛、开放世界概念发现以及统一标量目标的最优性均不属于第一篇工作的理论保证。

---

# 7. Knowledge Evolution（知识演化）

Knowledge Evolution 表示环境知识不断修正与完善。

Knowledge Evolution 的目标包括：

- 修正错误知识
- 提高知识置信度
- 发现新的 Capability Dependency
- 提高知识可迁移性

## 7.1 初始化与固定节点集合

当前版本由 LLM 根据对 Agent 公开的环境描述提出初始 Capability Schema 与候选因果图：

\[
Environment\ Description\xrightarrow{LLM}(V_0,M_0).
\]

完成节点准入审核、语义归一化与参数约束检查后，冻结节点集合：

\[
V_t=V_0.
\]

其中，\(V_0\) 是初始化审核完成时的 Capability Schema 集合，\(V_t\) 是在线运行时刻 \(t\) 的集合。固定的是 Schema 的种类与语义，不是运行时状态、实体绑定、数量参数或关系。例如，已有 `inventory_at_least(agent, item, quantity)` 时，将某条关系中的木材阈值从 2 修正为 1，不产生新节点。

所有初始关系通过 `AddCandidate` 以 `candidate` 状态进入 Knowledge Bank，并进入待验证列表。关系保留来源、适用范围、参数约束和状态历史；公开规则与 LLM 推测不得混为同一来源。初始化结束后，在线算法不依赖 LLM。

在线阶段只在已有节点之间发现、检验、修正、合并、替代或退役关系。需要现有 Schema 之外的谓词才能表达的解释记录为 `knowledge_gap`，不在线创建新 Capability。搜索困难或证据不足本身不等于 `knowledge_gap`。

## 7.2 操作与状态

Knowledge Bank 的最小操作集合为：

\[
\mathcal O_K=\{AddCandidate,RecordEvidence,Refine,Merge,Retire,Retrieve\}.
\]

关系的认知状态与生命周期状态独立管理：

```text
认知状态：candidate -> testing -> confirmed / rejected
          confirmed -> testing  # 系统性反证
生命周期：current / superseded / retired
```

`rejected` 表示证据不支持该关系，不删除记录；被更简单或更准确关系替代时使用 `superseded`。保留 `derived_from`、`merged_from`、`replaced_by` 和 `status_history`。待验证任务的原因、调度与重试条件属于算法运行状态，不增加关系的认知状态类别，也不增加环境结构知识类型。

## 7.3 经验接收与待验证列表

Knowledge Evolution 持续接收经验，按事件触发候选生成与修正，在有限预算内分批验证，未决任务跨轮保留。一轮是一次知识更新调度，不必等于一个环境动作或一个 episode。经验接收和验证不要求严格交替：重复经验可以只补充证据，没有新经验时也可以继续处理已有任务。

将合法公开观测映射为已有 Schema 的运行时事实，通过 `RecordEvidence` 记录状态转移、来源、适用范围与上下文。无法观测的值不得当作 `false`；观测缺失不改变 Capability 在环境状态上的布尔定义。

实际成功可以提供相应上下文中的可达证据，但共现不自动证明因果关系或条件必要性。当前 Policy 的普通执行失败不能直接构成结构反证，实际执行耗时不直接成为最低参考时间。资源前置数量与实际消耗量也必须区分。

待验证列表有三种入口：

1. 初始化后尚未验证的候选关系。
2. 新经验提示遗漏的关系：现有机制无法解释的成功转移等现象，能够用已有 Schema 表达时提出候选，否则记录 `knowledge_gap`。
3. 已有关系受到新证据质疑，包括对 confirmed 关系的结构反证以及公开规则发生冲突或变化。

生成候选时沿用稀疏父集与局部剪枝原则：从相关条件、单源和低阶组合出发，排除重复、约束矛盾与无关组合，只有存在歧义时才扩展相关组合，不全局枚举所有节点子集。允许在已有 Schema 内修正数量参数、父集、启动/保持角色、适用范围，或提出替代机制与成本修饰候选。

重复问题补充已有条目的证据，不重复创建任务。证据按其来源与适用范围使用，不将不同环境版本或不兼容范围的结果直接累计为同一结论。Knowledge 的语义独立于 Policy，但知识发现的进度和覆盖仍可能受到交互经验分布影响。

## 7.4 验证调度与未决任务

在可执行任务中按以下原则调度：

1. 优先复核受到有效结构反证质疑的 confirmed 关系。
2. 其次验证当前目标涉及的不确定关系。
3. 再处理其他候选，并照顾等待较久的问题，避免长期得不到验证。

优先级只决定先查什么，不改变确认标准。已有 confirmed 关系达到系统性反证标准时，应立即退回 `testing`，不能等待排队复核结束才调整状态。

每轮验证使用有限计算预算；预算耗尽或没有可执行任务时结束本轮。未决问题保留原因与再次执行所需条件：

| 未解决原因 | 后续处理 |
|---|---|
| 搜索预算耗尽 | 获得更多预算或能够继续原搜索时重试 |
| 当前基础世界不能合法施加干预 | 选择或等待其他符合适用范围的基础世界 |
| 干预存在混杂 | 寻找副作用更少的干预方式或基础世界 |
| 结果不能区分竞争解释 | 安排能够区分解释的补充验证 |

暂时没有执行条件的问题继续保留，先调度其他可执行任务。条件没有变化时，不反复执行同样的验证。调度不保证在有限资源内发现全部机制，也不要求所有关系都被确认后才能继续环境交互。

## 7.5 对照、消融与时间角色

完整、公开且无冲突的规则可以先用实际转移验证；隐藏、不完整、冲突或时间角色不明的部分进入局部干预。每个验证任务明确要检验的命题、关系、适用范围、基础世界、干预对象与时间窗口。

对需要干预的候选，先取得对照条件下达到目标的可达证据。优先复用适用的已有证据，不要求每次重建世界或重复搜索。用于检验新达成目标的基础状态应满足候选的启动条件，且目标尚未成立；保持条件按机制要求检查。

| 对照结果 | 处理 |
|---|---|
| `FOUND` 或适用的实际成功证据 | 可以继续必要性消融比较；缺少参考时间保证时只补充相关验证 |
| `PROVEN_UNREACHABLE` | 在该范围内调查遗漏条件、适用范围或候选问题，暂不进行必要性比较 |
| `UNKNOWN` | 保留问题，不判定候选错误 |

先逐个消融父条件，再针对未解决的歧义补充相关组合验证。例如，对启动父集 `{A, B}`，从同一基础世界分别消融 A 和 B，两次独立进行，不能在消融 A 后的世界上继续消融 B。

必要性消融 `do_[0,H](A=0)` 要求整个验证窗口内 A 不成立，不能允许执行路径违反赋值。未被消融的父条件仍遵循原有时间角色：启动条件只要求在机制启动时成立，不额外强制全程保持；保持条件在执行期间必须成立。角色不明时使用分阶段验证。机制启动时刻不等于 episode 起点。

验证流程为：

```text
同一基础世界的配对条件
-> Environment Adapter 验证干预合法性与最小修改
-> 检查副作用和混杂
-> 状态/动作层 Reference Planner 搜索
-> FOUND / PROVEN_UNREACHABLE / UNKNOWN 与参考时间、保证和范围
```

若消融 A 必然连带改变 B，不能将结果解释为 A 的独立作用。无效赋值排除，混杂证据仅作辅助支持，不能单独用于确认关系。有限窗口内不可达不能直接外推为无限时间不可达；全程消融也不能单独辨别条件只需在启动时满足还是必须保持。

Reference Planner 使用完整状态和规则，但学习端只接收受控的有效性、混杂、可达性、参考时间及其保证、适用范围等验证结果。隐藏规则、完整状态和 Oracle 搜索路径不暴露给学习端。当前 Module 性能不参与结构判断。

## 7.6 结果判读、替代机制与成本证据复用

以下判读只适用于合法、可比较且无影响判断的混杂的证据：

| 配对结果 | 允许的判读 |
|---|---|
| 对照可达，消融后被证明不可达 | 支持该条件在所检验范围、约束与窗口内具有必要性 |
| 对照与消融均可达 | 该条件不是该范围内目标可达的全局必要条件，继续区分父条件冗余和 OR 替代机制 |
| 两侧可达，缺少该条件时参考时间可靠增加 | 支持对应干预条件和适用范围下的成本修饰作用 |
| 任一关键结果为 `UNKNOWN` | 本次不足以判定，不作为反证 |

对候选 `A AND B -> Y`，消融 A 后仍可达，可能是 `B -> Y` 已足够，也可能是当前世界中的 C 支持另一条机制 `B AND C -> Y`。因此先记录证据，不直接删除 A；优先检验更简单的候选，并在公开规则或可观测经验提供线索时提出相关替代机制。不得读取 Oracle 路径来获得隐藏的 C。暂时无法区分时继续 `testing`。

若更简单关系在相应范围内得到确认，再执行替代；若确认的是另一条机制，则按 OR 关系分别保留。某条机制需要 A，与所有到达目标的路径都需要 A，是不同命题。替代路径的存在本身不能否定原机制，亦不能自动完成成本归因。

必要性验证的数据同时用于可达性与参考时间判断。硬机制与成本修饰分开表示，不要求证据分开采集。只有数据不足，或需要回答不同干预问题时，才追加验证。

两种时间比较必须保留区别：

- 全程禁止 A 的比较，回答“无法使用 A 会增加多少参考成本”。
- 仅初始提供 A 的可用性干预 `do_t=0(A=1)`，与初始未拥有 A、后续允许获得 A 的对照比较，回答“提前拥有 A 能节省多少参考时间”。

前一种比较的差值不能直接解释成后一种收益。参考时间按可比较的上下文和约束记录；近似搜索保留误差或界限，仅两次搜索返回不同路径长度不足以确认成本作用。同样，一次未观察到时间差也不足以断言所有范围内都没有成本作用。

## 7.7 证据驱动的状态更新与关系维护

每次有效新证据到来，先通过 `RecordEvidence` 追加记录，再立即评估相关关系，不等待整批任务结束。一次验证任务完成不代表整条关系验证完成；单个父条件的必要性证据不自动证明其他父条件、时间角色或整个适用范围。

按照已确认的结构知识确认原则，检查证据有效性、适用范围覆盖、条件与角色辨别程度以及未解决冲突。统计更新、预算和确认规则见下述“统计更新与确认规则”；具体数值仍为运行或实验参数。

### 统计更新与确认规则

Knowledge Evolution 不为整条机制维护单一总置信度，而为可达性、必要性、时间角色和成本修饰分别维护原子命题。一次有效验证结果映射为：

\[
y\in\{1,0,\bot\},
\]

其中 \(1\) 表示支持，\(0\) 表示在声明范围内的有效反证，\(\bot\) 表示 UNKNOWN、无效干预或无法排除混杂。只有 \(1\) 和 \(0\) 进入统计分母。

对原子命题 \(c\)，维护支持数和反证数：

\[
S_{c,t}=\sum_{\tau\le t}\mathbf 1[y_{c,\tau}=1],\qquad
C_{c,t}=\sum_{\tau\le t}\mathbf 1[y_{c,\tau}=0].
\]

使用简单 Beta 先验和后验更新：

\[
p_c\sim Beta(\alpha_0,\beta_0),\qquad
p_c\mid E_t\sim Beta(\alpha_0+S_{c,t},\beta_0+C_{c,t}).
\]

第一版默认使用 \(\alpha_0=\beta_0=1\)。后验下界 \(L_{c,t}\) 和上界 \(U_{c,t}\) 用于决策，置信水平和分位点方法为实验设置。令 \(n_{c,t}=S_{c,t}+C_{c,t}\)，原子命题按以下最小规则更新：

- `confirmed`：(n_{c,t}\ge n_{min}) 且 (L_{c,t}\ge \tau_{confirm})；
- `rejected`：(n_{c,t}\ge n_{min}) 且 (U_{c,t}\le \tau_{reject})；
- `testing`：其余情况，包括证据不足或竞争解释无法区分。

完整机制只有在目标可达、声明的父条件及时间角色均获得相应证据时才确认。成本命题单独更新，不改变硬可达性结论。confirmed 关系只有在新的合法结构反证达到拒绝规则时才退回 `testing`；一次普通执行失败或 UNKNOWN 不触发回退。

第 \(t\) 轮使用一个总验证预算 \(B_t\)。任务 \(q\) 的成本为 \(cost(q)\)，执行集合 \(Q_t\) 满足：

\[
\sum_{q\in Q_t}cost(q)\le B_t.
\]

任务沿用既有优先级：先处理受到有效结构反证质疑的 confirmed 关系，再处理当前目标相关的不确定关系，最后处理其他候选并照顾等待较久的任务。预算耗尽时保留已获得证据，未完成任务跨轮保留，不判为失败。预算数值、\(n_{min}\)、先验、置信水平、阈值、任务成本和成本实际意义阈值 \(\Delta\) 均未固定。

### Refine 与替代

修改父集、参数要求、时间角色、定义或适用范围时，通过 `Refine` 创建派生关系，以 `candidate` 开始并保留 `derived_from`。旧证据中适用于新命题的部分可以引用，但不重复计数，也不直接继承旧关系的 confirmed 状态；可以依据适用证据重新评估。

仅提出新候选，不足以废弃旧关系。旧关系是否退回 testing 由其自身反证决定。新关系得到确认，且能够在旧关系的适用范围内提供更简单或准确的解释后，才将旧关系标为 `superseded` 并记录 `replaced_by`。新关系只在更小范围成立时，不能替代整个旧关系，必要时分别建立对应范围的关系。

### Merge 与去重

父条件、时间角色、目标、参数约束和适用范围规范化后完全相同的候选，直接向已有关系补充来源与证据，不创建第二条关系。

已经分别存在的关系，只有确认语义等价后才通过 `Merge` 合并。合并记录保留 `merged_from`；双方证据按来源去重，同一次观察或验证只计入一次。根据合并后的证据重新评估认知状态，不简单继承较高状态，也不隐藏证据冲突；旧关系标为 `superseded` 并指向合并结果。

目标相同但父条件不同的 OR 机制，或时间角色、适用范围不等价的关系，不能直接合并。

### Retire 与维护范围

`Retire` 表示关系明确退出当前维护与使用范围，不代替真假判定。足以否定关系使用 `rejected`；被替代或合并使用 `superseded`；明确不再维护其适用范围时才使用 `retired`。

例如，明确停止支持某旧环境版本时，可以退役仅适用于该版本的关系；仅进入新版本不能自动退役旧知识。低使用频率、与当前目标无关、当前 Agent 不会执行、验证昂贵或长期 UNKNOWN，都不构成退役理由。

退役保留原证据、认知状态、原因和 `status_history`，允许 `confirmed + retired` 表示在原范围内已确认、但已退出当前维护范围的知识。

## 7.8 Retrieve 与正向使用边界

| 检索目的 | 关系范围 |
|---|---|
| 作为已确认知识参与正向规划或推理 | 适用范围匹配的 `current + confirmed` |
| 候选生成、复核、去重与历史追溯 | 可以检索全部状态，保留状态与范围标记 |

confirmed 关系退回 testing 后，暂时退出已确认知识的正向使用集合，但保留以供复核。rejected 关系可用于识别历史上已否定的候选；适用范围改变或出现新有效证据时，不得仅凭历史标签永久禁止重新调查。

没有检索到 confirmed 机制，只能说明当前知识不足，不能据此判定目标在环境中不可达。affordance 仍由规划器动态推导，不持久化。

## 7.9 完整流程与未定参数

以下为已确认流程的伪代码摘要，不是正式实现或新增 API：

```text
初始化：
  根据公开描述提出并审核 V0、初始候选关系
  冻结 Schema 集合；AddCandidate；加入待验证列表

持续运行：
  接收公开经验；RecordEvidence；立即评估受影响关系
  从初始化候选、遗漏关系线索、关系质疑维护去重后的待验证列表
  仅使用已有 Schema 生成或修正候选；表达不足记录 knowledge_gap

  本轮预算内，依优先级选择可执行任务：
    明确命题、适用范围、基础世界、时间角色与干预条件
    复用适用证据；公开规则可由实际转移验证
    需要干预时先取得对照可达证据，再做单条件消融
    Adapter 验证合法性、副作用与混杂；Reference Planner 搜索
    RecordEvidence；同时评估可达性与参考时间
    立即更新相关关系认知状态
    有歧义则安排相关组合、替代解释或分阶段验证
    满足条件时执行 Refine、Merge、替代或 Retire，保留历史
    未决任务记录原因和重试条件，条件不变时避免重复执行

  预算耗尽或无可执行任务时结束本轮，保留未决任务
  正向 Retrieve 仅返回范围匹配的 current + confirmed 关系
```

每轮产出更新后的 Knowledge Bank、关系与证据历史、未决验证任务及 `knowledge_gap`。不要求清空待验证列表，也不承诺发现全部机制或获得全局最优参考时间。

完整流程和统计更新规则已确认；预算数值、调度的具体实现、样本量、置信水平、阈值和后验计算方法仍未固定。Knowledge Bank 与 Skill Library 的最小调用协议见第 8 节，Module、SPT 与 SPI 的正式表示见第 4.3、4.4 和 4.6 节。

讨论理由与已确认的 Crafter 木镐示例见 [Knowledge Evolution 完整算法决策](docs/decisions/2026-09-06-knowledge-evolution-algorithm.md)。该示例是流程说明，不是实验结果。

---

# 8. Knowledge Bank 与 Skill Library 接口

Knowledge Bank 和 Skill Library 是严格分离的长期记忆，但以 Capability 状态描述作为共同语言。Knowledge Bank 表达环境中哪些状态条件支持达到目标状态；Skill Library 表达当前 Agent 能够如何实现所需状态转移，以及如何通过 SPT 持续改善未来实现。两库都不取代对方的职责。

运行时由规划层承担协调职责。规划层不是第三个长期知识库，可以与任务规划器或执行调度器合并实现；它读取当前状态和目标，选择状态转移机制，检查资源接口和顺序，查询或请求技能实现，并根据实际结果继续规划。规划层不持久化环境机制、Module、SPI 或 SPT。

## 8.1 职责边界

| 部分 | 职责 | 不负责 |
|---|---|---|
| Knowledge Bank | 返回已确认的环境状态转移机制、适用范围和参考成本 | 动作、Policy、Module、SPI、SPT 或当前 Agent 是否会执行 |
| Skill Library | 维护 Module、SPI、Skill Family/SPT；选择或生成状态转移的实现 | 判断环境机制是否真实成立，或更新 Knowledge Bank 的结构关系 |
| 规划层 | 将当前目标分解为 Capability 转移，协调机制与实现，处理资源冲突和执行顺序 | 将技能性能写成环境知识，或保存两库的长期内容 |

同一个 Capability 可同时出现于环境机制与技能实现契约，但含义不同：环境机制中的条件描述环境允许该转移的结构条件；实现契约中的条件描述某个 Module 或 SPI 需要、消耗、释放或产生的状态。Module 的额外执行限制不能自动写入 Knowledge Bank。

例如，环境机制要求 `nearby(agent, ball)`，而某个 `PickupBall` Module 只会从球体左侧稳定执行。前者可以作为环境机制条件；后者只是当前技能实现的适用条件。只有独立环境验证才能将后者提升为 Knowledge Bank 的关系。

## 8.2 规划层查询 Knowledge Bank

规划层以目标 Capability、当前运行时 Capability 事实和环境范围查询 Knowledge Bank：

```text
RetrieveMechanisms(
    target_capability,
    current_capability_facts,
    environment_scope
) -> Mechanism[]
```

`current_capability_facts` 是本轮运行时输入，不写入 Knowledge Bank。`environment_scope` 用于匹配环境、版本和其他已声明的适用范围。

Knowledge Bank 默认只返回范围匹配的 `current + confirmed` 机制；每条结果至少包含：

```text
id
P_start
P_hold
Y
reference_time
applicability_scope
```

当前未满足某个启动条件，不等于机制不适用；仍须允许检索这类机制，供规划层寻找其缺失条件的实现路径。`reference_time` 保留第 7 节的上下文、干预条件和精度保证，未知时不能当作零成本。

Knowledge Bank 不返回动作、Policy、Module、SPI、SPT、技能成功率或规划器内部状态。多个指向同一目标的机制按 OR 分别返回，由规划层结合当前状态、资源接口和可用实现选择或安排验证；Knowledge Bank 不按当前 Agent 的技能能力替规划层作选择。没有返回 confirmed 机制只说明当前知识不足，不代表环境不可达。

affordance 仍由规划层根据当前状态、机制和环境约束动态推导，不持久化到 Knowledge Bank。

## 8.3 规划层请求 Skill Library 实现

规划层根据选定机制和当前状态发出状态转移请求：

```text
RequestImplementation(transition_request) -> ImplementationResponse

TransitionRequest:
  input_capabilities
  target_capability
  resource_requirements
  environment_scope
  execution_context
```

`input_capabilities` 是计划在该转移启动时满足的状态条件，不代表当前已经满足；`target_capability` 是需要实现的结果；`resource_requirements` 表达规划层已知的资源要求或冲突；`execution_context` 包含实际运行时事实、对象绑定、局部上下文和所选机制的启动/保持角色，不能在转交实现时丢失 `P_hold`。它们都不是新的永久 Capability Schema。

Skill Library 按以下顺序响应：

1. 查找输入、输出和资源契约匹配，且当前可用的 Module。
2. 没有合适 Module 时，查找对应 Skill Family 的 SPT，实例化面向当前目标的 SPI，完成 Policy 学习或适应，并在资格检查通过后封装为新 Module。
3. 没有匹配 Module 且没有可用 SPT，或此次 SPI 学习/适应未通过资格检查时，返回不可用。

响应状态为：

```text
reused_module
created_module_from_spi
unavailable
```

这里描述逻辑调用完成后的结果。`created_module_from_spi` 表示本次 SPI 学习或适应已经通过资格检查并封装为 Module；刚实例化、尚未学会或未通过资格检查的 SPI 不能作为可执行技能。学习调度和等待的具体实现不在最小协议中展开；`unavailable` 仅说明本次未获得实现，不表示环境结构不可达。

`ImplementationResponse` 至少包含：

```text
status
module_id                 # 可为空
spi_id                    # 可为空
spt_id                    # 可为空
implementation_contract:
  start_capabilities
  hold_capabilities
  declared_output_capabilities
  resource_consumption
  resource_release
  implementation_constraints
  applicability_scope
```

该对象返回第 4.6 节定义的实现契约 \(\Gamma\)。启动输入、保持输入、声明输出、资源消耗和资源释放是所有技能实现声明的最低语义字段。`implementation_constraints` 是 Module/SPI 的实现适用条件，不改变环境机制语义。规划层只需要知道该实现是否可用、是否需要学习及其契约，不需要读取 SPT 的内部参数、Policy 参数或训练轨迹。

SPT 的完整路径必须保留：

```text
已有 Module
    或
SPT -> 实例化 SPI -> Policy 学习或适应 -> 资格检查 -> Module
```

因此 Skill Library 不是静态的 Module 查询目录。SPT 继续从多个 SPI 的经验中更新，并改善后续 SPI 的实例化和学习；SPT、SPI 与 Module 的正式表示见第 4.3、4.4 和 4.6 节，Skill Evolution、资格与版本规则见第 9 节。最小接口本身不重复定义这些内部算法。

## 8.4 执行结果与两条反馈路径

执行接口及其结果为：

```text
ExecuteImplementation(implementation, current_state) -> TransitionResult

TransitionResult:
  target_achieved          # true / false；无法判定时明确记为未知
  observed_state_changes
  consumed_resources
  released_resources
  produced_capabilities
  execution_status
```

`current_state` 仅指学习端合法可见的状态或观测，不是 Oracle 完整状态。`execution_status` 记录执行是否完成、失败或无效等情况，不与实现来源混用；复用既有 Module 还是由 SPI 新建 Module 由 `ImplementationResponse.status` 记录。目标真假根据实际可观测结果判定，观测不足不等于 false。

规划层使用实际环境状态，而不是技能声明中的预期输出，检查目标、资源变化和后续转移是否仍可行，然后继续规划、选择替代机制或发起新的状态转移请求。

同一次执行生成两条职责不同的反馈路径：

```text
同一次学习或执行
  公开状态变化、合法性、环境范围和干预元数据
    -> Knowledge Evolution 的 RecordEvidence 入口
  SPI、实例参数、Policy 学习过程和执行上下文
    -> Skill Library 内部的 Skill Evolution / SPT 更新
```

第一条路径仅提供第 7 节规定的环境证据，仍须经过候选、验证与状态更新流程；它不因一次执行成功或失败自动修改 Knowledge Bank。第二条路径用于改善 SPI 和 SPT，不进入 Knowledge Bank。完整训练经验不需要在 Knowledge Bank 与 Skill Library 的接口上传输。

公开证据摘要包括 `before_state`、`after_state`、`public_observation`、`environment_scope`、`intervention_metadata` 和 `evidence_validity`；前后状态仍限于公开可见部分，普通执行不能伪装为干预。Policy 成功率、训练步数、Module 性能、SPT 参数和梯度不作为跨库结构证据。SPT 内部反馈需保留解释经验所需的实例上下文，但不要求永久保存全部原始轨迹，具体摘要或更新形式由 Skill Evolution 算法确定。

例如，`DropKey` 一次失败时，规划层可以重新规划或将该实现视为当前不可用；Skill Library 可以将带有 SPI 上下文的经验用于后续学习；Knowledge Bank 不会因此推断新的环境前置条件，也不会否定空手状态的环境机制。

## 8.5 资源冲突与技能组合示例

规划层必须使用实现契约检查连续转移的资源占用、消耗与释放。该职责使多个单独可成功的 Module 能够安全组合。

设一个单物品槽的门、钥匙和球体场景，目标是取得门后的球体。这是资源组合示例，不是 Crafter 的库存或配方规则，也不是主实验环境选择。以下只展开与手部占用相关的条件，其余开门、丢弃位置合法性等环境条件仍须满足：

```text
hand_occupancy(agent) = key -> door_state(door) = open
hand_occupancy(agent) = empty AND nearby(agent, ball)
    -> hand_occupancy(agent) = ball
```

Skill Library 可提供：

```text
OpenDoor:
  input: hand_occupancy(agent) = key
  output: door_state(door) = open
  resource_consumption: none
  resource_release: none

DropKey:
  input: hand_occupancy(agent) = key
  output: hand_occupancy(agent) = empty
  resource_consumption: none
  resource_release: hand slot

PickupBall:
  input: hand_occupancy(agent) = empty, nearby(agent, ball)
  output: hand_occupancy(agent) = ball
  resource_consumption: none
  resource_release: none
```

丢弃钥匙解除其持有状态并释放手部槽位，不代表钥匙在环境中被消耗或销毁；拾球占用手部槽位，通过输出的 `hand_occupancy` 表达，也不把“空手谓词”当作被销毁的实体资源。真实物料消耗例如制作木镐的 `wood: 1` 才记录为数量消耗。状态失效、产生和占用变化通过输入、输出及实际状态变化表达。

若开门后实际状态仍为 `hand_occupancy(agent) = key`，规划层发现 `PickupBall` 的空手输入不满足，于是安排：

```text
FindKey -> OpenDoor -> DropKey -> FindBall -> PickupBall
```

规划层决定必须丢弃钥匙及其时机；Policy 或执行器完成底层丢弃动作。`DropKey` 可以是 Module，也可以是环境允许的原子动作对应的执行实现。技能接口的资源字段服务于这种时序与冲突检查，不能反向变成 Knowledge Bank 关于环境必要条件的结论。

## 8.6 最小调用协议与边界

完整运行时调用顺序为：

```text
规划层读取当前状态和任务目标
-> RetrieveMechanisms 查询 Knowledge Bank
-> 选择机制并分解为 Capability 转移
-> RequestImplementation 请求 Skill Library 实现
-> Skill Library 返回既有或新建 Module，或 unavailable；可附带 SPI/SPT 来源标识
-> ExecuteImplementation 执行实现并返回 TransitionResult
-> 规划层用实际状态继续规划
-> 公开环境证据进入 Knowledge Evolution；SPI 经验在 Skill Library 内回流 SPT
```

本协议不预先规定分布式调用、并行技能、跨进程版本协商、复杂置信度传输、Policy 后端或完整统计字段。它只固定两库之间的最小职责边界和运行时信息流；SPT 的内部更新与版本规则由第 9 节独立规定，不改变本接口的信息边界。

接口选择理由与讨论示例见 [Knowledge Bank 与 Skill Library 接口决策](docs/decisions/2026-09-08-knowledge-bank-skill-library-interface.md)。

---

# 9. Skill Evolution（技能演化）

Skill Evolution 通过多个 SPI 的学习反馈更新同一 Skill Family 的 SPT，使未来 SPI 在有限预算内更快、更稳定地形成合格 Module。它不保存越来越多的 Policy，也不改变固定的 Skill Family 语义：

\[
T_f^t=\langle T_f^{static},\omega_f^t\rangle
\longrightarrow
T_f^{t+1}=\langle T_f^{static},\omega_f^{t+1}\rangle.
\]

## 9.1 SPI 上下文与适应

第一篇工作只在接口和 Policy 类兼容的 SPI 之间共享初始化与元参数；异构接口需要显式适配后才能混合。对 SPI \(I_j\)，构造不含隐藏环境信息的上下文：

\[
z_j=Encode_f(b_j,q_j,\Gamma_j,P_j,\Omega_j),\qquad
\zeta_j^0=G_{\omega_f^t}(z_j).
\]

SPI 在有限预算内使用 support 数据学习或适应 Policy，并使用独立 query 数据评估学习效果：

\[
\theta_j^{k+1}=U_{A_j}(\theta_j^k,D_j^{support,k}),\qquad
\ell_j^k=\mathcal L_j(\theta_j^k;D_j^{query}).
\]

训练/更新数据与独立评价数据必须分离。跨 SPI 反馈可以在 Skill Library 内保留并回放；缓冲区容量、采样方式和配额属于实现设置。

## 9.2 FOMAML 更新

第一篇工作采用上下文条件化 FOMAML 作为 Skill Evolution 的唯一主方法。对一批 SPI，使用 support 内循环得到 \(\theta_j^{K_j}\)，使用 query 损失形成一阶元梯度：

\[
g_j^{FO}=\left(\frac{\partial G_{\omega_f}(z_j)}{\partial\omega_f}\right)^\top
\nabla_{\theta_j^{K_j}}\mathcal L_j^{query}(\theta_j^{K_j}),
\]

\[
\widetilde\omega_f=\omega_f^t-\beta\frac{1}{|\mathcal J|}\sum_{j\in\mathcal J}g_j^{FO}.
\]

更新只产生候选 SPT \(\widetilde T_f=\langle T_f^{static},\widetilde\omega_f\rangle\)，不直接覆盖当前版本。Reptile 和完整二阶 MAML 作为备选或未来扩展，不属于第一篇工作的主算法。

## 9.3 SPI 到 Module 的最小资格规则

SPI 学习得到候选 Policy \(\pi_j^*\) 后，Skill Library 使用独立的 held-out qualification episodes 进行资格检查，并在评价期间冻结 Policy。检查至少包括启动/保持条件、目标可检测性、契约与资源约束、适用范围以及有限执行预算。

若候选 Policy 在有效资格试验中的目标成功率达到预设门槛，并满足契约和资源约束，则形成 Module；若证据充分且明显低于门槛，判定为不合格；若证据不足，则继续评估或保留为未决。不可判定或无效结果不自动当作失败，明确的硬契约或安全违反不能由成功率抵消。

资格判断只决定是否可执行。多个合格 Module 之间的性能评分和选择属于 Skill Library 内部软比较，不能替代硬资格，也不能写入 Knowledge Bank。具体样本数、置信方法、阈值和范围分组方式为实验设置。

## 9.4 SPT 候选接受与版本恢复

候选 SPT 使用跨 SPI 反馈训练，并在独立评价数据上与当前 SPT 比较。比较时保持 SPI、适应预算、算法和评价规则一致。候选只有在平均未来 SPI 学习效率达到预设改善要求，且既有重要 SPI 的性能不低于容忍范围、契约与安全约束满足时才接受；否则继续使用当前 SPT。

候选接受后切换为新的活动版本，并保留上一稳定版本；证据不足或门槛失败时保持当前版本。若新版本出现明确的安全/契约问题或可比范围内的系统性学习效率退化，可以恢复上一稳定版本；普通单次失败不足以触发恢复。是否设置试运行阶段属于实现选择，不在核心规范中规定完整状态机。

SPT 更新只影响未来 SPI 的初始化和学习，不原地覆盖既有 Module。旧 Module 只有在性能、接口或资格有效性受到质疑时才重新评估，并形成新版本；既有合格 Module 不因 SPT 恢复自动删除。

## 9.5 最小流程与未定参数

~~~text
对每个 Skill Family：
  用当前 SPT 实例化 SPI
  在有限 support 预算内学习 Policy
  用独立 query 反馈更新候选 SPT
  用 held-out qualification 判断是否形成 Module
  用独立评价比较候选 SPT 与当前 SPT
  接受则切换 SPT 并保留旧版本；否则保持当前版本
~~~

内外循环预算、反馈缓冲区容量、资格样本数与阈值、最小改善量、退化容忍度、恢复检测窗口和 Policy 后端均为运行或实验参数，未在本规范中固定。

选择理由见 [Skill Evolution 算法决策](docs/decisions/2026-09-21-skill-evolution-algorithm.md)，复杂度收敛说明见 [理论设计原则与复杂度收敛决策](docs/decisions/2026-09-21-theory-design-principles-and-simplification.md)。

---

# 10. Module Library（技能库）

Module Library 是持续学习的技能记忆。

其中保存：

每一个已经学习完成并通过资格检查的技能 Module，以及 SPT 演化所需的 SPI 上下文和技能经验。

每个 Module 绑定一个 SPI 并保存自己的 Policy；一个 SPI 可以形成多个不同实现版本的 Module。SPT 属于 Skill Family 级共享对象，不复制进每个 Module。

随着 Module 数量增加：

SPT 将不断获得新的经验。

Module Library 不断扩展。

SPT 不断演化。

二者共同提升持续学习能力。

---

# 11. Current Assumptions（当前假设）

为了聚焦核心问题，当前版本采用以下假设：

1. 初始 Capability Schema 由 LLM 根据对 Agent 公开的环境描述提出，并经过准入审核；第一篇工作不研究在线新节点发现。

2. 初始 Skill Program Template 可以人工定义。

3. 每个技能属于某一个 Skill Family。

4. 每个 Module 保存自己的 Skill Program Instance。

5. 多个 SPI 可以共同推动对应 Global SPT 的持续演化。

6. Knowledge Bank 表示环境结构，严格独立于当前 Agent 是否已经拥有实现某个状态转移的 Module。

7. Capability 必须可检测，但允许只能通过合法动作或规划被间接干预。

上述假设将在未来工作中逐步放宽。

---

# 12. Future Extensions（未来扩展）

当前项目重点研究：

Knowledge Evolution

Skill Evolution

未来可进一步扩展：

- 自动发现新的 Skill Program Template
- 自动划分 Skill Family
- 更复杂环境下的开放世界技能演化
- 多智能体知识共享
- 在线 Concept Formation 与新 Capability 节点发现
- 世界模型辅助知识验证

---

# Research Scope（研究边界）

本项目聚焦于持续强化学习中知识与技能的持续演化问题，重点研究智能体如何通过环境交互不断修正世界知识，并利用不断演化的技能程序模板提升未来任务的学习效率。

为了保证研究目标清晰，当前版本仅关注以下研究内容：

## 本文研究的问题

### 1. Knowledge Evolution（知识演化）

研究世界知识如何随着强化学习经验不断更新，包括：

- Capability（能力）之间依赖关系的持续修正；
- 知识置信度的持续更新；
- 新知识的持续积累；
- 知识对后续任务的迁移作用。

### 2. Skill Evolution（技能演化）

研究技能程序模板（Skill Program Template, SPT）如何利用多个技能程序实例（Skill Program Instance, SPI）的经验不断演化，提高未来技能学习效率。

重点关注：

- SPT 如何实例化为 SPI；
- SPI 如何适应具体技能目标；
- 多个 SPI 如何共同推动 SPT 演化；
- 演化后的 SPT 如何提升新技能学习效率。

### 3. Knowledge-guided Continual Learning（知识引导的持续学习）

研究 Knowledge Bank 与 Module Library 如何共同作用，实现：

- 技能发现；
- 技能迁移；
- 技能复用；
- 持续学习。

目标是在连续任务中不断提升学习效率，而不仅仅保持历史任务性能。


## 当前版本不研究的问题

为了突出核心科学问题，以下内容不作为当前工作的研究重点。

### 1. 自动发现 Skill Program Template（技能程序模板）

当前版本假设 Skill Family（技能族）可以预先定义。

SPT 的自动发现、自动划分与自动合并属于未来研究方向。

### 2. 世界模型（World Model）学习

本文不研究环境动力学建模，也不预测未来状态。

Knowledge 表示的是 Capability（能力）之间的因果依赖关系，而不是状态转移模型。

### 3. 大语言模型作为在线决策器

LLM 并不是系统主体。

如果使用 LLM，其作用仅限于提供初始世界知识先验，而非参与在线规划或动作决策。

整个系统应能够脱离任何大语言模型独立运行。

### 4. 通用机器人控制

本文关注的是持续学习框架，而不是某一种机器人平台。

不同环境可以拥有不同动作空间和不同 Policy 实现，但共享相同的知识演化思想。

### 5. 开放世界无限技能发现

当前版本假设任务空间有限且可逐步扩展。

开放世界中无限技能族的自动发现与组织属于未来工作。

## Research Boundary（研究边界总结）

本文关注的是一个新的持续强化学习范式：

Knowledge Evolution（知识演化）负责回答：

> 智能体应该学习什么（What to Learn）。

Skill Evolution（技能演化）负责回答：

> 智能体应该如何学习（How to Learn）。

二者共同驱动持续学习能力不断提升。

除此之外，诸如世界模型学习、大语言模型规划、自动技能发现等问题均不属于本文的核心研究内容，而将作为未来工作的进一步扩展。

---

# Design Principles（设计原则）

整个项目始终遵循以下原则：

1. Knowledge 与 Skill 分离。

2. Knowledge 决定学习什么。

3. Skill 决定如何学习。

4. Policy 不是长期知识。

5. Module 是技能复用的基本单位。

6. 持续学习能力来自 Knowledge Evolution 与 Skill Evolution，而不是保存越来越多的 Policy。

7. 所有算法、代码、实验和论文必须遵循本文件定义。 

8. **最小充分性**：只保留回答科学问题、保证接口语义或使主算法可实现所必需的定义与规则。

9. **科学贡献优先**：优先表达 Knowledge/Skill 分离、Capability 契约和两类演化的核心机制，不把通用工程能力包装成理论贡献。

10. **核心与防护分离**：核心理论、实现防护和未来扩展分开记录；防护可以存在，但不必全部进入数学主线。

11. **每个问题一个主方法**：第一篇工作为每个主要问题选择一个可实现的主算法，替代方案只保留选择理由和适用边界。

12. **超参数不等于理论**：样本数、置信水平、阈值、缓冲区容量和更新频率保留为实验设置，除非它们直接决定语义正确性。

13. **封闭假设优先**：第一篇工作在受控、有限、可验证的假设下完成，不提前解决开放世界和所有异常情况。

14. **数学服务于理解**：公式用于明确对象、输入、输出和更新关系，不为每个程序字段建立同等复杂的数学结构。

15. **按必要性增加复杂度**：只有在正确性、可实现性或核心评价确实需要时，才增加状态、统计量、约束或流程。

