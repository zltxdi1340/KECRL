# KECRL 实现准备盘点与边界

> 日期：2026-09-29。最新状态：最小 Python 类型与接口骨架已创建，语法及导入检查通过；算法、部署和训练尚未开始。第 1 节保留创建骨架前的盘点事实，不代表最新文件清单。

## 1. 当前文件事实

本次递归检查了项目内文件（含隐藏文件），未执行项目代码或安装依赖。结论仅适用于当前 `D:\Project\KECRL`，不推断其他路径中的历史项目。

| 范围 | 盘点结果 | 对下一步的影响 |
|---|---|---|
| `src/` | `knowledge`、`continual_learning`、`counterfactual`、`environments`、`utils`、`visualization` 均只有 `.gitkeep` | 当前无正式实现可审查或复用 |
| `legacy/HRC_Bachelor/` | 只有 `.gitkeep` | 当前没有可迁移 HRC 源码；保持只读 |
| `configs/`、`scripts/`、`workspace/` | 只有占位文件 | 无现成配置、运行入口或原型 |
| `experiments/`、`datasets/`、`checkpoints/` | 只有占位文件 | 无训练产物或正式实验结果 |
| 根目录及依赖文件 | 未发现依赖清单、锁文件、构建配置或 `.git` 元数据；`README.md` 为短占位文件 | 不能宣称已有可运行环境或已知依赖版本 |
| 理论文档与 `paper/method.md` | 已有理论、方法稿、伪代码、框架图文本规范、接口契约及参数清单 | 可复用的是已确认设计与文档，不是算法代码 |

`ARCHITECTURE.md` 建议的 `src/skills/` 尚不存在。本轮不创建目录；目录落位应在实现设计时单独明确。当前不需要迁移、删除或重构任何历史源码。若后续提供其他路径中的 HRC 项目，需另做只读适配审查，不能预先认定兼容。

## 2. 最小实现范围建议

以 [接口与数据契约](interface-and-data-contract.md) 为唯一字段映射依据，以 [PROJECT_BIBLE](../PROJECT_BIBLE.md) 为理论依据。下一步先形成逻辑对象与调用责任的设计，不重复扩展协议。

| 既有部分 | 首轮需要设计的内容 | 边界 |
|---|---|---|
| Knowledge Bank | Capability Schema、机制、证据、原子命题统计和既有历史字段的对应关系；`RetrieveMechanisms` | 只返回范围匹配的 `current + confirmed`；无机制不等于不可达 |
| Skill Library | SPT/SPI/Module 的引用和版本关系；`RequestImplementation` | 先复用合格 Module，否则实例化、学习并独立资格检查；SPI 不可直接执行 |
| 执行与环境接口 | `ExecuteImplementation`、`TransitionResult` 与公开可检测结果的映射 | 声明输出不替代事实；保留未知结果和启动/保持角色 |
| Planner / Pipeline | 三项调用的顺序、资源检查、实际结果检查、双反馈分流和任务版本视图 | 只保存任务内上下文，不成为第三个长期知识库 |
| 两类演化 | 环境证据和技能反馈分别进入既有更新流程的接入位置 | 接口骨架不等于 Beta-Binomial 验证或 FOMAML 已实现 |

实现设计应同时覆盖“已有 Module”“SPI 学习后取得合格 Module”“未取得实现”“结果未知”这些已有路径，不新增状态。未实现的学习后端不能通过伪造资格或返回成功来代替。

## 3. 后续功能检查的验收要点

以下是未来实现的检查要求，本轮未编写或运行测试，也不构成实验协议：

- 查询不泄漏未确认或不适用的机制；同一目标的不同机制仍分别保留。
- 请求中的计划条件与实际观测明确区分；启动、保持及资源约束不在调用中丢失。
- 未通过资格检查的 Policy/SPI 不成为可执行 Module。
- 目标只由实际检测确认；`unknown`、`unavailable` 和普通 Policy 失败不成为结构反证。
- Knowledge Evidence 不携带 Policy 参数、梯度、完整轨迹或 Module 性能。
- 任务使用起始 Knowledge/SPT 版本视图；接受或恢复 SPT 不原地修改既有 Module。

可在后续功能验证中使用受控输入检查这些契约；此类输入不代表选定实验环境，也不能证明算法有效性。

## 4. 未决项与推进顺序

语言、依赖版本、序列化、存储形式、Policy 后端、环境适配器、Reference Planner 及上下文生成器的具体参数化尚未确定。所有数值继续遵循 [参数清单](runtime-and-experiment-parameters.md)，不在本文件补设默认值。

建议按以下顺序推进：

1. 最小数据结构与调用时序设计已形成，见 [接口契约第 4—6 节](interface-and-data-contract.md)。首批类型与接口骨架已创建；后续补齐契约约束并验证边界行为，不能把导入成功当作功能验收通过。
2. 在进入实现阶段后，建立最小接口闭环并检查上述契约；不把功能通过当作研究结果。
3. 按既有规则逐步实现 Knowledge Evolution、独立 Module 资格检查、FOMAML 与 SPT 版本接受/恢复。
4. 单独确认实验协议、环境、基线、指标、数据划分与预算，再开展小规模验证和正式实验。

本次盘点不改变已确认研究结论；已有内部评审记录也不等同于外部学术评审或实验验证。
