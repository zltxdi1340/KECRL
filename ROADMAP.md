# ROADMAP.md

# KECRL Research Roadmap（项目路线图）

> Version: 1.7
>
> Status: Pre-implementation Design Preparation
>
> 本文档用于管理整个项目的研发进度，只记录研究阶段、目标、当前状态和下一步计划，不记录理论推导。

---

# Overall Goal（总体目标）

构建一种基于 Knowledge Evolution（知识演化）与 Skill Evolution（技能演化）的模块化持续强化学习框架（Knowledge-Evolving Continual Reinforcement Learning, KECRL），实现知识、技能与策略的持续积累和演化，提高跨任务学习效率与持续学习能力。

---

# Stage 1：理论框架设计（Theory Design）

**状态：✅ 已完成**

## 目标

建立整个项目的理论框架，统一核心概念，明确研究边界。

## 当前任务

- [x] 明确 Scientific Question（科学问题）
- [x] 明确 Core Philosophy（核心思想）
- [x] 定义 Knowledge（知识）
- [x] 定义 Capability（能力）
- [x] 定义 Skill Program Template（技能程序模板，SPT）
- [x] 定义 Skill Program Instance（技能程序实例，SPI）
- [x] 定义 Module（技能模块）
- [x] 定义 Knowledge Bank（知识库）
- [x] 定义 Skill Library（技能库；历史文档中的 Module Library 为同一对象）
- [x] 完成整体学习流程设计
- [x] 完成项目文档初始化

## 当前阶段成果

目前已经形成统一理论框架，后续所有工作均以 PROJECT_BIBLE 为唯一理论依据。

---

# Stage 2：数学建模（Mathematical Modeling）

**状态：✅ 核心数学建模已完成；方法稿与配套文档已完成本轮内部一致性评审，进入实现设计准备**

## 目标

将理论框架形式化，建立完整数学描述。

## 计划内容

- [x] 定义基础环境决策过程与公开观测接口
- [x] 定义 Capability 空间（概念、边界、粒度与准入标准已确认）
- [x] 定义 Knowledge Graph（机制结构、操作、生命周期与验证原则已确认）
- [x] 定义 Module（可执行单位、实现契约、资格与性能边界已确认）
- [x] 定义 SPT 与 SPI（固定语义结构、可演化状态、实例化规格及与 Module 的边界已确认）
- [x] 定义 Knowledge Evolution 完整流程（含验证调度、证据复用、状态更新与关系维护）
- [x] 确定 Knowledge Evolution 的具体统计更新公式与确认规则（Beta-Binomial 原子命题更新、单轮预算与三态确认；数值为运行参数）
- [x] 定义 Knowledge Bank 与 Skill Library 最小接口（规划层协调、机制查询、实现请求、执行结果与反馈分流）
- [x] 定义 Skill Evolution（最小充分的 FOMAML、跨 SPI 反馈、Module 资格与 SPT 接受规则已确认）
- [x] 定义 Continual Learning Pipeline（运行时任务闭环、双反馈分流与任务边界更新已确认）
- [x] 定义 Knowledge Evolution 与 Skill Evolution 联合目标（两个局部演化目标、任务层耦合评价与独立更新边界已确认）
- [x] 完成整体 KECRL 分层数学形式化与有限可证明性质

## 阶段产出

- `paper/method.md` 方法稿、四组最小伪代码与框架图规范（已形成，本轮内部一致性评审已记录）
- [接口与数据契约](docs/interface-and-data-contract.md)、[参数清单](docs/runtime-and-experiment-parameters.md)（已形成，本轮内部一致性评审已记录；具体技术选型与参数仍未冻结）
- [内部评审记录](docs/meetings/2026-09-29-method-internal-review.md)（不代表外部学术评审或实验验证）

---

# Stage 3：代码重构（Framework Refactoring）

**状态：✅ 最小 GPU-capable 实现、受控回归和归档已完成；目标环境适配待开始**

## 目标

将本科 HRC 项目升级为新的 KECRL 框架。

2026-10-04：核心接口、Knowledge/Skill 对象、Pipeline、CUDA policy backend、受控离散回归和结果追踪已实现。受控离散环境已封存为接口回归基准，不作为方法有效性主环境。详见 [受控环境归档说明](docs/controlled-discrete-environment-archive.md)。

## 计划内容

- [x] 重构整体目录
- [x] 重构 Environment Interface
- [x] 建立 Knowledge Bank
- [x] 建立 Skill Library
- [x] 建立 SPT Framework
- [x] 建立 SPI Framework
- [x] 重构 Continual Learning Pipeline

---

# Stage 4：算法实现（Algorithm Development）

**状态：🟡 参考实现和受控回归已完成；目标环境验证待开始**

## 目标

完成 Knowledge Evolution 与 Skill Evolution 的算法实现。

## 计划内容

- [x] Knowledge 更新机制
- [x] SPI 实例化机制
- [x] SPT 演化机制
- [x] Module 管理机制
- [ ] Counterfactual Verification
- [x] Continual Learning Pipeline

---

# Stage 5：实验设计（Experiments）

**状态：🟡 受控环境回归已完成；Crafter 目标环境适配待开始**

当前实验方面的准确进展是：理论层已经明确需要验证的对象、模块边界和主要反馈路径，但尚未完成环境部署、实验代码、正式训练、对照实验或结果统计。当前不把候选环境、基线和指标写成已确定方案。

## 目标

验证理论有效性。

## 当前实验准备

- 已明确实验需要分别检验 Knowledge Evolution、Skill Evolution、Module 复用和持续任务闭环。
- 已明确实验应区分环境结构证据、技能学习反馈和任务层表现，不能用 Module 性能直接替代 Knowledge 证据。
- 已明确需要验证知识更新、SPT 更新、Module 资格与版本恢复等核心规则是否按理论工作。
- 已完成受控离散环境的 GPU smoke、20-run 候选比较、holdout 回归和 400-episode 预算敏感性实验；全部保留为 `formal_result=false`。
- 受控环境由于任务简单且 query success 饱和，不能区分方法有效性，已封存而非删除。
- 下一阶段先完成 Crafter 依赖、观测/动作适配和单环境 smoke，再冻结 Crafter 任务划分、baseline、指标和预算。

## MiniGrid

- 不作为后续主实验环境；最终环境组合在实验设计阶段确认。

## Crafter

- [x] 依赖与版本锁定（Crafter 1.8.3；正式依赖仍需最终冻结）
- [x] 环境迁移（原生 reset/step 薄适配）
- [x] 观测/动作契约适配（RGB 64x64x3、inventory allowlist、17 动作）
- [x] 单环境 GPU smoke（仅连通性验证，`formal_result=false`）
- [x] 单步 Pipeline smoke（fixture Module；验证反馈分流，`formal_result=false`）
- [ ] Knowledge Evolution
- [ ] Skill Evolution
- [ ] Module Reuse

## Baselines

- [ ] Continual RL
- [ ] Modular RL
- [ ] LLM Agent

---

# Stage 6：论文撰写（Paper Writing）

**状态：🟡 Method、实验候选记录和局限性草稿已形成；Crafter 正式结果尚未开始**

## Method

- [x] Method 初稿与本轮内部一致性审查（评审记录已形成，非投稿定稿）
- [x] Framework Figure 的文本规范（正式投稿图待制作）
- [x] Algorithm 最小伪代码（文档表达，尚无实现）

## Experiments

- [ ] Main Results
- [ ] Ablation
- [ ] Visualization

## Discussion

- [ ] Analysis
- [ ] Limitation
- [ ] Future Work

---

# Milestones（里程碑）

## Milestone 1

完成理论框架设计。

**状态：✅ 已完成**

---

## Milestone 2

完成数学建模。

**状态：✅ 核心数学建模已完成；方法稿与配套文档本轮内部一致性评审已记录，与 Stage 2 一致**

---

## Milestone 3

完成代码框架重构。

**状态：⬜**

---

## Milestone 4

完成核心算法。

**状态：⬜**

---

## Milestone 5

完成全部实验。

**状态：⬜**

---

## Milestone 6

完成论文投稿版本。

**状态：⬜**

---

# Current Focus（当前重点）

当前重点：

> 七项核心理论设计与数学形式化均已确认，方法稿与配套文档已有内部评审记录。最小 Python 类型、校验、内存适配器、任务版本视图和受控运行闭环已创建，已验证确认机制过滤、复用 Module、四类任务结果映射、反馈分流、任务视图字段、知识证据边界及运行时证据保持 `unknown`；这不是算法或实验结果。Knowledge Evolution、FOMAML、环境部署和正式实验尚未开始。

2026-10-01 补充：首轮受控适配器已增加实现契约兼容性检查，覆盖适用范围、启动能力、可比较资源和实现约束；不兼容请求返回 `unavailable`。这仍属于接口边界验证，不是资格算法、Knowledge Evolution、FOMAML 或实验结果。

## 讨论顺序与状态

1. [x] Module 的正式定义。
2. [x] SPT 与 SPI 的数学和程序表示。
3. [x] Skill Evolution 的具体算法。
4. [x] Knowledge Evolution 的统计更新公式、预算和确认规则。
5. [x] Continual Learning Pipeline。
6. [x] Knowledge Evolution 与 Skill Evolution 的联合目标。
7. [x] 整体 KECRL 的数学形式化和可证明性质。

Module、SPT 与 SPI 的正式规范见 PROJECT_BIBLE 第 4 节，Pipeline、联合目标与整体形式化见第 6 节，Knowledge Evolution 见第 7 节，Skill Evolution 见第 9 节。各项选择理由记录于 docs/decisions；两库边界与演化规则继续有效。

当前处于实现前设计阶段，后续实现必须先遵守 `docs/meetings/2026-09-29-method-internal-review.md` 的准入条件；实验环境、基线和指标仍未冻结。

现有目录与依赖盘点及接口契约第 4—6 节设计已完成。首批骨架位于 `src/knowledge/contracts.py`、`src/skills/contracts.py`、`src/continual_learning/contracts.py`，另有两个包初始化文件；已完成构造时校验、受控内存适配器、任务版本视图、最小运行闭环和知识证据禁止字段校验，尚未实现持久化、资格判定、真实版本切换或演化更新。

首轮实现建议与验收范围见 [首轮实现准备计划](docs/implementation-first-pass-plan.md)。本轮仅使用 Python 标准库创建类型骨架，未安装依赖；其他后端与实验选择仍未冻结。
