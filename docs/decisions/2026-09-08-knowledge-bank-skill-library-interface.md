# Knowledge Bank 与 Skill Library 接口决策

> 日期：2026-09-08  
> 状态：已确认  
> 适用范围：KECRL 第一篇工作，讨论顺序第 8 项

## 1. 背景

Knowledge Bank 保存可迁移的环境结构知识；Skill Library 保存 Module、SPI 和 SPT 组成的技能生成与复用能力。两者不能直接相互承担职责：将 Module 性能写入 Knowledge Bank 会破坏其 Agent 无关性，而让 Skill Library 判断环境因果关系会混淆实现能力与环境可达性。

此前已确认 Capability 是两者共同使用的环境状态语言，技能接口至少表达输入、输出、资源消耗和资源释放。本次讨论进一步确认最小调用协议、规划层职责和 SPT 经验回流边界。正式规范位于 [PROJECT_BIBLE 第 8 节](../../PROJECT_BIBLE.md#8-knowledge-bank-与-skill-library-接口)。

## 2. 备选方案

### 2.1 两库直接相互调用，或由规划层协调

选择由规划层承担运行时协调职责。它不是第三个长期记忆，也可以与既有任务规划器或执行调度器合并实现。

Knowledge Bank 只回答环境中哪些 Capability 条件支持目标状态；Skill Library 只回答当前哪些 Module 可实现转移，以及能否由 SPT 产生或学习 SPI。规划层读取当前状态和目标，安排机制、技能、资源与顺序。

这样可以解释“开门后丢弃钥匙再拾取球体”的组合问题：环境知识表达空手是拾球机制的条件，技能实现声明资源占用与释放，规划层据此安排 `DropKey`，而底层 Policy 负责实际执行。

### 2.2 跨库传递完整技能经验，或在 Skill Library 内部回流

选择只把完成环境结构判断所需的公开状态结果送入 Knowledge Evolution；SPI、实例参数、训练过程和执行上下文保留在 Skill Library 内，回流到 SPT。

理由是 SPT 更新必须区分经验对应的 SPI、对象、目标和执行条件，但这并不要求 Knowledge Bank 接收完整训练轨迹、梯度、Policy 参数或 Module 性能。一次 Policy 失败可以是 SPT 更新的有效反馈，却不能自动构成环境机制反证。

### 2.3 静态 Module 目录，或保留 SPT 实例化路径

选择保留如下完整路径：

```text
已有 Module
    或
SPT -> 实例化 SPI -> Policy 学习或适应 -> Module
```

Skill Library 不只是为 Capability 提供现成技能的版块。SPT 是持续演化对象，多个 SPI 的经验应在库内更新 SPT，以改善后续实例化和学习。接口只暴露实现可用性与 Capability/资源契约，不固定 SPT 的内部表示或更新公式。

## 3. 最终决定

接口以三个最小对象组织：

| 调用 | 目的 |
|---|---|
| `RetrieveMechanisms` | 规划层向 Knowledge Bank 查询范围匹配的已确认机制 |
| `RequestImplementation` | 规划层向 Skill Library 请求实现 Capability 状态转移 |
| `TransitionResult` | Skill Library 将实际状态、资源变化和执行状态返回规划层 |

Knowledge Bank 默认返回 `current + confirmed` 关系，不返回动作、Policy、Module、SPI 或 SPT。Skill Library 返回 Module 或 SPI/SPT 标识、输入、输出、资源消耗、资源释放和实现适用条件。它的额外实现限制不自动成为环境机制条件。

执行后的公开状态变化通过 Knowledge Evolution 既有的证据入口处理；SPT 经验回流不经过 Knowledge Bank。规划层只以真实状态结果继续规划，避免把技能声明的预期结果误当作环境事实。

## 4. 影响范围

- [PROJECT_BIBLE](../../PROJECT_BIBLE.md) 第 8 节作为接口的唯一正式规范。
- 第 7 节的 `Retrieve` 规则继续生效；接口不改变 Knowledge Evolution 的验证标准或信息边界。
- SPT/SPI 的最终数学和程序表示、SPT 更新规则、Policy 训练细节仍待后续 Skill Evolution 建模确定。
- 本协议不进入实验设计，不实现代码，不确定并行或分布式系统细节。

## 5. 后续

第 8 项接口已完成。下一项按既定顺序讨论实验环境、基线与指标；在其讨论开始前，不再扩展未确认的 SPT 数学或实现协议。
