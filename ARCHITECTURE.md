# ARCHITECTURE.md

# KECRL 项目架构说明（Architecture）

> 本文档定义 KECRL 项目的整体组织架构、目录职责、研发流程以及 AI 协作规范。
>
> PROJECT_BIBLE.md 定义"研究什么"，ARCHITECTURE.md 定义"如何组织整个项目"。

---

# 1. 架构设计原则（Architecture Principles）

整个项目遵循以下原则：

## 1. Scientific First（科研优先）

整个仓库按照科研流程组织，而不是按照软件工程组织。

所有代码、实验和论文都服务于 Scientific Question。

---

## 2. Single Source of Truth（单一事实来源）

每一种信息只能存在一个官方位置。

例如：

- 理论定义 → PROJECT_BIBLE.md
- 项目路线 → ROADMAP.md
- 科研日志 → docs/research_logs/
- 论文正文 → paper/
- 正式代码 → src/
- 临时验证 → workspace/

任何内容不得出现多个版本。

---

## 3. Separation of Responsibilities（职责单一）

每个目录都有唯一职责。

任何文件都必须有唯一归属。

禁止职责重叠。

---

## 4. Minimal Sufficient Design（最小充分设计）

理论和实现只保留回答科学问题、保证接口语义或支持主算法所必需的复杂度。核心理论、实现防护和未来扩展必须分开；未经确认的状态、统计保证和替代算法不得写成正式结论。

## 5. Theory Drives Everything（理论驱动开发）

所有开发必须遵循以下顺序：

Scientific Question

↓

Theory

↓

Algorithm

↓

Code

↓

Experiment

↓

Paper

不得直接从代码开始开发。

---

# 2. 项目整体结构（Project Structure）

整个项目分为六个层次。

```
Theory
│
├── PROJECT_BIBLE.md
├── ROADMAP.md
└── ARCHITECTURE.md

↓

Research

docs/

↓

Implementation

src/

↓

Experiment

experiments/

↓

Publication

paper/

↓

Archive

legacy/
```

整个仓库围绕这一科研流程组织。

---

# 3. 根目录职责（Root Directory）

## README.md

项目入口。

包括：

- 项目简介
- 环境安装
- 使用说明
- 快速开始

不包含科研内容。

---

## PROJECT_BIBLE.md

整个项目最高设计文档。

定义：

- Scientific Question（科学问题）
- Core Philosophy（核心思想）
- Core Concepts（核心概念）
- Learning Framework（学习框架）
- Research Scope（研究边界）
- Design Principles（设计原则）

整个项目所有理论必须保持一致。

---

## ROADMAP.md

项目开发路线。

记录：

- 当前阶段
- 已完成工作
- 下一阶段计划
- 长期路线图

不记录理论。

---

## ARCHITECTURE.md

当前文档。

定义：

- 项目组织方式
- 目录职责
- 文件管理规范
- AI 协作规范

---

# 4. docs/（科研工作区）

docs 用于记录科研过程。

不保存正式论文内容。

---

## literature/

文献管理。

包括：

- 阅读笔记
- Related Work 分类
- 文献总结
- 调研结果

---

## meetings/

会议记录。

包括：

- 导师讨论
- 周报
- 汇报内容
- Meeting Minutes

---

## ideas/

科研想法。

所有未经验证的新想法统一记录。

只有经过讨论确认后才能写入 PROJECT_BIBLE。

---

## research_logs/

科研日志。

建议每天一篇。

包括：

- 今日目标
- 今日完成内容
- 新发现
- 遗留问题
- 下一步计划

不记录 Debug。

---

## decisions/

设计决策。

每一个重要决定单独建立一个 Markdown 文件。

统一格式：

背景

↓

备选方案

↓

最终决定

↓

原因分析

↓

影响范围

用于未来论文和 Reviewer 回复。

---

## figures/

文档图片。

仅供文档说明使用。

---

# 5. paper/（论文工作区）

保存论文正文。

包括：

- Outline
- Abstract
- Introduction
- Related Work
- Preliminaries
- Method
- Experiments
- Discussion
- Conclusion
- References

禁止保存：

- 阅读笔记
- 实验日志
- 临时代码

---

## paper/figures/

论文正式图片。

要求：

- 可直接投稿
- 高清矢量
- 与正文一致

---

# 6. src/（正式算法实现）

src 只保存正式算法。

整体建议组织如下：

```
src/

knowledge/
    Knowledge Bank
    Knowledge Evolution

skills/
    Skill Program Template
    Skill Program Instance
    Module Library

continual_learning/
    Continual Learning Pipeline

counterfactual/
    Counterfactual Verification

environments/
    Environment Interface

utils/

visualization/
```

这里的代码应与 PROJECT_BIBLE 完全对应。

禁止：

- Demo
- Debug
- Prototype
- 临时代码

---

# 7. experiments/（实验）

保存所有实验。

建议结构：

```
experiments/

configs/

logs/

results/

analysis/

notebooks/
```

这里只保存实验。

不保存算法。

---

# 8. scripts/

工具脚本。

例如：

- 数据处理
- 自动绘图
- 表格生成
- 指标统计
- 自动运行实验

禁止放核心算法。

---

# 9. assets/

公共资源。

包括：

- Framework 图
- 流程图
- PPT 图片
- 图标
- 海报

供整个项目共享。

---

# 10. configs/

统一配置。

包括：

环境参数

训练参数

模型参数

实验参数

---

# 11. datasets/

数据集。

包括：

数据下载

数据说明

预处理结果

不保存代码。

---

# 12. checkpoints/

模型权重。

保存训练结果。

按照：

环境

↓

算法

↓

日期

进行组织。

---

# 13. workspace/

科研实验区。

允许：

Prototype

Debug

快速验证

Notebook

实验代码

当验证成功后：

迁移到 src/

否则删除。

---

# 14. legacy/

历史项目。

例如：

HRC_Bachelor

该目录默认只读。

任何新开发不得修改。

---

# 15. 文件流转规范（Workflow）

整个项目遵循统一流程：

```
Idea

↓

docs/ideas

↓

讨论

↓

PROJECT_BIBLE

↓

数学建模

↓

src/

↓

Experiment

↓

paper/

↓

Publication
```

任何内容都应按照该流程推进。

---

# 16. 各目录之间的关系（Relationship）

PROJECT_BIBLE

↓

定义理论

↓

ROADMAP

↓

安排研究计划

↓

src

↓

实现算法

↓

experiments

↓

验证算法

↓

paper

↓

描述算法

整个项目形成闭环。

---

# 17. AI 协作规范（Instructions for AI Contributors）

所有 AI（ChatGPT、Codex、Claude、Gemini 等）必须遵守以下规范。

## 必须首先阅读

开始任何任务之前，应首先阅读：

- PROJECT_BIBLE.md
- ROADMAP.md
- ARCHITECTURE.md

PROJECT_BIBLE 为最高优先级。

---

## 严格遵守当前研究阶段

不得提前实现未来阶段内容。

不得跳跃开发。

---

## 理论优先

如果代码与 PROJECT_BIBLE 冲突：

修改代码。

不得修改理论。

理论修改必须经过讨论。

---

## 不允许擅自修改目录结构

任何新增目录必须经过确认。

不得创建：

- misc
- temp
- backup
- final
- test
- new
- others

---

## 正式代码与实验代码分离

正式算法：

src/

实验验证：

workspace/

实验结果：

experiments/

不得混放。

---

## 保持统一命名

所有代码、文档、论文必须使用 PROJECT_BIBLE 中定义的术语。

例如：

Knowledge Bank（知识库）

Capability（能力）

Skill Program Template（技能程序模板，SPT）

Skill Program Instance（技能程序实例，SPI）

Module（技能模块）

Policy（策略）

禁止出现多个名称表示同一概念。

---

## 不允许直接修改历史项目

legacy/ 默认只读。

任何修改必须得到明确许可。

---

## 文档优先于代码

如果发现理论发生变化：

先修改 PROJECT_BIBLE。

再修改代码。

最后修改论文。

禁止直接修改代码而不同步文档。

---

## 理论复杂度审查

在新增理论前，必须说明它属于核心规范、可选实现细节还是未来扩展；默认采用最简单的充分方案。只有正确性、可实现性或核心评价确实需要时，才增加额外字段、状态机、统计保证或算法。

---

## 所有工作必须服务于 Scientific Question

任何新增算法、代码、实验和论文内容，都必须回答 PROJECT_BIBLE 中定义的 Scientific Question。

若无法建立直接联系，应重新评估该工作的必要性。
