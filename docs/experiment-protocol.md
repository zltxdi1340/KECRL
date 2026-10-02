# KECRL 最小实验协议

> 状态：实现准备版。CPU smoke test 只验证代码和数据流，不是正式实验结果。

## 研究问题

1. Knowledge Evolution 是否能按 Beta-Binomial 规则更新、确认和拒绝原子命题？
2. 合格 Module 是否能在契约、范围和资源兼容时复用，并在不兼容时返回 `unavailable`？
3. 上下文条件化 FOMAML 是否改善后续 SPI 的学习效率？
4. Knowledge/Skill 双反馈分流和持续任务闭环是否可追溯运行？

## 环境与任务

CPU smoke 使用受控模拟环境，仅验证 `RetrieveMechanisms -> RequestImplementation -> ExecuteImplementation` 数据流。正式环境、任务划分和真实 Policy 后端待定。

## 基线与消融

正式实验至少包含一个合理基线和一个关键消融。候选消融为关闭 Module 复用、关闭 Knowledge Evolution 或关闭 Skill Evolution；最终选择和依据待定。

## 指标

接口验证记录任务结果、`unavailable` 率、反馈字段完整性和版本可追溯性。正式实验记录 SPI 学习效率、独立 query 损失、Module 资格率、任务成功率及资源/时间成本；具体定义和统计检验待定。

## 预算、种子与结果

每次运行保存完整配置、随机种子、Git commit、Python/框架/CUDA/GPU 信息、结构化日志和 JSON/CSV 结果。正式服务器运行另外保存 checkpoint 和失败日志。Knowledge 验证预算、FOMAML 内外循环预算、训练步数、worker 数和混合精度必须从配置读取。

## 未决项目

正式环境、基线实现、任务划分、`n_min`、后验分位点、Module 资格阈值、SPT 接受/回滚阈值、FOMAML 学习率、训练预算和多随机种子集合均未冻结。不得把这些项目写成理论结论或实验结果。
