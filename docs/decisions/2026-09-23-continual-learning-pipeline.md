# Continual Learning Pipeline 决策

> 日期：2026-09-23  
> 状态：已确认  
> 适用范围：KECRL 第一篇工作，数学建模阶段

## 1. 背景

Knowledge Evolution、Skill Evolution、Knowledge Bank 与 Skill Library 的职责和接口已经分别确定，仍需明确它们如何连接为持续学习运行闭环。本决策只规定必要的运行时顺序、反馈分流和更新边界，不增加新的知识库或学习算法。

## 2. 已确认约束

- Knowledge Bank 与 Skill Library 严格分离。
- 规划层负责运行时协调，但不是第三个长期知识库。
- 运行时通过 RetrieveMechanisms、RequestImplementation、ExecuteImplementation 完成机制查询、实现获取和执行。
- 执行结果以实际观测和 TransitionResult 判定，Module 的声明输出不等同于环境事实。
- 公开环境证据进入 Knowledge Evolution；SPI 上下文和技能学习经验留在 Skill Library。
- Knowledge Evolution 和 Skill Evolution 分别遵循 PROJECT_BIBLE 第 7、9 节，不因 Pipeline 而改变。

## 3. 方案选择

候选方案包括每次执行后同步完成两类演化、完全异步演化，以及在运行时记录反馈并在任务或明确更新边界执行演化。第一篇工作采用第三种：运行时任务闭环保持直接；演化在边界运行。这样避免验证或元训练阻塞当前任务，也避免引入任务中途切换复杂版本状态机。

## 4. 正式流程

~~~text
读取公开观测与目标
-> RetrieveMechanisms
-> 选择当前使用的机制
-> RequestImplementation
-> ExecuteImplementation
-> 检查实际状态与 TransitionResult
-> 目标达成则结束；否则继续规划或选择替代机制
-> 分别记录环境证据与技能经验
-> 在任务或明确更新边界运行 Knowledge/Skill Evolution
-> 新知识与新接受的 SPT 版本默认供后续任务使用
~~~

任务结果保留 completed、continued、unavailable、unknown 四类。unavailable 表示当前未获得合格实现，不代表环境不可达；unknown 不当作失败或结构反证。

任务内固定开始时使用的 Knowledge Bank 与 SPT 版本视图。演化结果默认供后续任务使用，且 SPT 更新不原地改写既有 Module。

## 5. 尚未决定

以下内容不是 Pipeline 核心定义，不在本决策中固化：任务自动分段、并行任务调度、更新边界的具体触发频率、任务中途读取新 Knowledge、自动重试和规划搜索算法。

## 6. 影响范围

- PROJECT_BIBLE 第 6 节定义 Pipeline；接口细节继续由第 8 节定义。
- ROADMAP 将 Continual Learning Pipeline 标记完成，下一项为 Knowledge Evolution 与 Skill Evolution 的联合目标。
- 不改变两库职责、信息边界或各自演化算法。
- 不涉及代码实现、实验环境或实验指标。
