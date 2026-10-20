# 历史实验封存区

本目录保存 2026-10-20 之前完成的受控环境和 Crafter 诊断文档。它们是研究审计与负结果证据，不再作为当前主线的实验协议。

## 目录

- `crafter/`：Crafter RGB、PPO、生存、空间读出、wood gather、SPT 迁移和 Module pipeline 诊断。
- `controlled/`：受控离散环境的 formal freeze、统计分析、预算、holdout 和结果审查材料。

源代码、测试、冻结配置、实验运行器和小型结果摘要仍保留在原位置，以便复现历史结论。模型 checkpoint、重复轨迹、原始运行目录和生成日志不纳入 GitHub；本次整理删除了本地生成的 `.log`、`.out` 和 `.err` 文件，未删除结果摘要。

当前研究路线见 [`docs/research-roadmap-2026-10-20.md`](../research-roadmap-2026-10-20.md)：

1. LIBERO：SPT/SPI 同族实例迁移与元学习效果；
2. Continual World：持续任务流、前向迁移和遗忘；
3. Crafter：长期记忆、探索和 Knowledge Evolution；
4. RoboCasa365：Knowledge–Skill 联合验证。

封存材料不构成正式结果。每个文件仍保留原始日期、预算和协议范围，引用时应注明其为诊断、pilot 或历史 freeze。
