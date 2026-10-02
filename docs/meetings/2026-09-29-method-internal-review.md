# KECRL 方法稿与实现前文档内部评审记录

> 日期：2026-09-29  
> 状态：内部评审完成；实现准入条件已记录

## 1. 评审范围

本次评审核对了：

- `paper/method.md`
- `PROJECT_BIBLE.md` 第 4、6、7、8、9 节
- 2026-09-06 至 2026-09-26 的相关 decision 文档
- `docs/interface-and-data-contract.md`
- `docs/runtime-and-experiment-parameters.md`

## 2. 已通过项

1. Knowledge Bank 与 Skill Library 的长期职责和信息边界一致；Planner 只负责运行时协调，不形成第三个长期知识库。
2. Capability、机制 `M=(P_start,P_hold,Y)`、SPT、SPI、Module 和实现契约的符号及语义一致。
3. Knowledge Evolution 使用原子命题的 Beta-Binomial 更新；`UNKNOWN`、混杂和普通 Policy 失败不会直接形成结构反证。
4. 确认、拒绝、`testing`、`Refine`、`Merge`、`Retire`、替代和历史保留规则与正式规范一致。
5. Skill Evolution 只采用上下文条件化 FOMAML 主方法；support/query 分离，候选 SPT 独立评估，既有 Module 不被原地改写。
6. Module 资格判断、SPT 接受/保持/恢复、Continual Learning 闭环和两条反馈路径均有最小伪代码表达。
7. 框架图规范明确区分两个持久化存储，以及环境证据和技能反馈的分流方向。
8. 接口契约覆盖 `RetrieveMechanisms`、`RequestImplementation`、`ExecuteImplementation`、`TransitionResult` 和 Knowledge Evidence 的最小字段。

## 3. 实现准入条件

进入接口实现前必须保持以下约束：

- 不在线创建 `V_0` 之外的新 Capability Schema。
- 不把 Policy 参数、梯度、完整训练轨迹或 Module 性能写入 Knowledge Bank。
- 不把声明输出当作实际环境事实；目标必须由实际可检测结果确认。
- 不把 `unavailable` 或 `unknown` 解释为环境不可达。
- 任务内固定起始 Knowledge Bank 与 SPT 版本视图；更新供后续任务使用。
- 所有参数配置显式区分理论固定项、运行参数和实验参数。

## 4. 仍未确定

以下项目继续保持未冻结：`n_min`、确认/拒绝阈值、置信水平、验证预算、调度细节、FOMAML 学习率与适应步数、Module 资格样本数和阈值、SPT 最小改善量、回滚检测窗口、Policy 后端、环境、基线和指标。

首版 `Beta(1,1)` 仅是已确认的弱先验默认，不代表所有实验必须使用该先验。Crafter 木镐案例仍只是流程示例，不是实验结果。

## 5. 结论与下一阶段

方法稿和实现前文档达到进入接口实现设计的条件。下一阶段只设计与现有契约对应的最小接口骨架和数据结构，不扩展理论对象，不冻结实验环境，不运行训练。

