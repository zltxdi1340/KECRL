# Module、SPT 与 SPI 形式化决策

> 日期：2026-09-19  
> 状态：已确认；Skill Evolution 的具体算法仍待确定  
> 适用范围：KECRL 第一篇工作，数学建模阶段

## 1. 背景

此前的 PROJECT_BIBLE 已从概念层面规定：SPT 是 Skill Family 的共享模板，SPI 是具体技能实例，Module 是 Skill Library 保存和复用技能的基本单位。Knowledge Bank 与 Skill Library 的最小接口进一步要求技能实现声明输入、输出、资源消耗、资源释放与实现适用条件，但尚未确定三者的正式数学和程序表示。

本次决策不把最初关于 SPI 应保存哪些字段的假设作为不可修改前提，而从已经确认的 Capability 接口、两库职责边界、规划层职责以及 Skill Evolution 目标反推三者的分工。正式规范写入 PROJECT_BIBLE 第 4.3、4.4 和 4.6 节；本文件记录候选选择、理由、影响与仍未决定的问题。

## 2. 已确认约束

1. SPT 是长期演化对象，当前版本的 Skill Family 预先定义。
2. Capability 是 Knowledge Bank、规划层和技能实现契约之间共享的状态语言。
3. Knowledge Bank 判断环境机制，Skill Library 管理当前 Agent 如何实现状态转移。
4. 规划层负责跨机制、跨技能的运行时组合，不是第三个长期知识库。
5. 刚实例化、尚未学会的 SPI 不能当作可执行技能返回。
6. 完整训练轨迹、Policy 参数、梯度和 Module 性能不进入 Knowledge Bank。
7. 技能实现的额外限制和普通执行失败不自动成为环境结构知识。

## 3. 候选方案与选择理由

### 3.1 SPT 作为纯符号程序模板

该方案具有较强可解释性，也容易检查输入、输出和资源，但难以统一表达连续控制、Policy 初始化和跨实例学习效率提升。SPT 演化容易退化为规则编辑，不能充分承载“提高未来技能生成和学习能力”的研究目标，因此未单独采用。

### 3.2 SPT 作为纯 Policy 元学习器

该方案可以直接生成 Policy 初始化参数并使用元学习更新，但会使 SPI 退化为一次参数生成结果，弱化 Capability 契约、成功与终止判定以及技能程序结构，也容易把 SPT 变成共享 Policy 参数库，因此未单独采用。

### 3.3 结构化程序规格与可学习生成器结合

最终选择该方案。SPT 同时保存稳定的 Skill Family 语义结构与可演化的生成状态；SPI 是面向具体目标的技能程序与学习规格；Module 是 SPI 经学习或适应并通过资格检查后形成的可执行实现。

该划分使三层职责互不重复：

```text
SPT：跨实例共享的生成与学习能力
SPI：具体技能应该如何构造和学习的规格
Module：已经学成、可以复用和执行的实现
```

## 4. SPT 正式定义

对 Skill Family \(f\)，时刻 \(t\) 的 SPT 为：

\[
T_f^t=\left\langle T_f^{static},\omega_f^t\right\rangle.
\]

\(T_f^{static}\) 至少规定：

- Skill Family 标识；
- 参数绑定 Schema；
- 实现契约构造器；
- 程序结构 Schema；
- SPI 实例化接口；
- 反馈与更新接口。

\(\omega_f^t\) 是可演化状态，可以承载共享表征、Policy 初始化先验、条件化生成参数、探索先验或适应超参数。第一篇工作固定 Skill Family 语义：

\[
T_f^{static,t+1}=T_f^{static,t},
\]

主要演化 \(\omega_f^t\)。该定义不提前规定 \(\omega_f^t\) 必须是神经网络、显式参数表还是其他表示。

## 5. SPI 正式定义

给定 SPT 版本、参数绑定和状态转移请求：

\[
I_j=Instantiate(T_f^t,b_j,q_j),
\]

并定义：

\[
I_j=\left\langle
id_j,f,v_f^t,b_j,q_j,\Gamma_j,P_j,\zeta_j^0,A_j
\right\rangle.
\]

其中 \(\Gamma_j\) 是具体实现契约，\(P_j\) 是实例化后的程序规格，\(\zeta_j^0\) 是 SPT 生成的初始化状态，\(A_j\) 是学习或适应规格。程序规格可以声明观测或特征接口、允许的 Policy 类、训练目标以及成功、失败和终止判定，但不固定为某一种程序语言或网络结构。

SPI 不保存最终 Policy、长期 Module 统计、Module 生命周期或 Knowledge Bank 的机制置信度，也不要求永久保存完整训练轨迹。学习失败的 SPI 可以提供带上下文的技能反馈，但不会形成可执行 Module。

## 6. Module 正式定义

Module 定义为：

\[
m_i=\left\langle
id_i,spi_i,\pi_i,\Gamma_i,E_i,\Sigma_i,\mu_i
\right\rangle.
\]

其中 \(spi_i\) 是绑定的 SPI，\(\pi_i\) 是已经形成的 Policy，\(\Gamma_i\) 是实现契约，\(E_i\) 是技能经验，\(\Sigma_i\) 是内部统计摘要，\(\mu_i\) 保存来源、版本和生命周期等元数据。

实现契约为：

\[
\Gamma_i=\left\langle
I_i^{start},I_i^{hold},O_i,C_i,R_i,X_i,\Omega_i
\right\rangle.
\]

它分别表达启动输入、保持输入、声明输出、资源消耗、资源释放、实现特有约束和适用范围。声明输出不是实际环境事实；实际结果只能由 `ExecuteImplementation` 返回的 `TransitionResult` 判定。

一个 SPI 可以形成多个 Module，例如不同 Policy 版本或不同接口范围的实现；每个 Module 只绑定一个 SPI。Module 必须通过 Skill Library 内部资格检查后才能进入可执行候选集。硬资格判断与候选之间的软性能评分必须分开，具体阈值和评分公式由 Skill Evolution 决定。

## 7. 完整形成路径与职责边界

```text
SPT
  -> Instantiate
SPI
  -> Policy Learning / Adaptation
  -> Qualification
Module
  -> ExecuteImplementation
TransitionResult
```

SPT/SPI 负责一个明确 Capability 转移的技能生成与学习规格；规划层负责多个机制、技能和资源的全局组合。SPI 不自行查询 Knowledge Bank，也不保存跨机制任务计划。Module 的实现契约与环境机制 \(M=(P_{start},P_{hold},Y)\) 是不同命题，即使两者使用相同 Capability，也不能相互替代。

## 8. 与最小接口的关系

`RequestImplementation` 返回既有 Module，或者由 SPI 完成学习/适应、通过资格检查后新建的 Module，并附带其 \(\Gamma\) 契约。接口可以返回 `spi_id` 和 `spt_id` 追踪来源，但不执行裸 SPI。`ExecuteImplementation` 执行 Module 并返回 `TransitionResult`；`TransitionResult` 是返回对象，不是与 `ExecuteImplementation` 并列的第四个调用。

规划层根据实际观测结果继续规划，不能把 \(O_i\) 的声明输出当作已经发生的状态变化。技能经验在 Skill Library 内用于 Module 更新和 SPT 演化；只有符合 Knowledge Evolution 信息边界的公开状态证据才能进入 Knowledge Bank。

## 9. 尚未决定且不得提前固化的内容

- SPT 可演化状态的具体参数化形式；
- SPI 反馈摘要和 feedback encoder；
- SPT 更新公式、训练目标和优化算法；
- Policy 类、Policy 学习与适应算法；
- Module Experience 保存原始轨迹、缓冲区还是充分统计量；
- Module 资格阈值、性能评分公式与统计更新；
- Module 生命周期状态及既有 Module 的重评估规则；
- 并行技能、递归技能调用和分布式执行协议。

## 10. 影响范围与后续

- PROJECT_BIBLE 第 4.3、4.4 和 4.6 节成为 Module、SPT 与 SPI 的正式规范。
- PROJECT_BIBLE 第 8 节的 `ImplementationResponse` 直接引用 Module 实现契约，并明确 `ExecuteImplementation -> TransitionResult`。
- ROADMAP 中“定义 Module”和“定义 SPT 与 SPI”标记为完成。
- 下一讨论项是 Skill Evolution 的具体算法，不进行代码实现或实验设计。
