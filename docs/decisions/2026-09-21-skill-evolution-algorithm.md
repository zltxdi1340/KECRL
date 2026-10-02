# Skill Evolution 算法决策

> 日期：2026-09-21  
> 状态：算法结构与状态规则已确认；具体数值和 Policy 后端未固定  
> 适用范围：KECRL 第一篇工作，数学建模阶段

## 1. 背景

Module、SPT 与 SPI 的形式化已经确认：SPT 保存固定 Skill Family 语义与可演化状态，SPI 是具体技能的程序和学习规格，Module 是 SPI 学习后通过资格检查形成的可执行实现。本次决策确定多个 SPI 如何更新 SPT、SPI 如何取得 Module 资格，以及 SPT 候选版本如何接受、拒绝、试运行和回滚。

正式算法统一维护在 PROJECT_BIBLE 第 9 节。本文件保存候选方案的比较、选择理由、状态语义和未定边界，不作为第二份独立演化的算法规范。

## 2. 元更新方案选择

### 2.1 经验统计聚合

只累计成功率、训练步数和超参数，容易实现和解释，但难以迁移复杂表征，也没有直接优化未来 SPI 的适应过程，因此不作为主算法。

### 2.2 相似 Module 的 Policy 迁移

复制或微调相似 Module 的 Policy 具有工程价值，但长期演化对象会重新偏向具体 Policy，相似度也可能导致负迁移。它可以是 SPI 初始化的辅助来源，不能替代 SPT Evolution。

### 2.3 Reptile

Reptile 将共享初始化朝各 SPI 适应后的参数移动，实现简单、不经过内循环反向传播，也可适配部分不可微内循环。但其元目标是隐式的，难以直接表达学习曲线、适应成本、契约风险和上下文条件生成；当内循环只有一步时还容易接近普通联合训练。因此将其保留为简化或回退后端，不作为主算法。

### 2.4 FOMAML

最终选择上下文条件化 FOMAML。它保留 support 内循环与独立 query 外循环，直接优化“从 SPT 初始化后，SPI 在有限预算内学得多快、多稳”，但省略完整 MAML 的二阶项。它能够自然纳入 Capability 契约、SPI 上下文、学习曲线、适应成本、稳定性和风险约束，与 KECRL 的 Skill Evolution 目标最一致。

FOMAML 本身不是项目创新点。KECRL 的贡献来自它在 Knowledge/Skill 分离、Capability 契约、SPT-SPI-Module 分层、跨 SPI 持续回放、资格判定和版本安全边界中的组合。

## 3. 主算法决定

第一篇工作采用：

```text
同族兼容接口
-> 上下文条件化 SPT 初始化
-> SPI 有限预算内循环适应
-> support/query 职责分离
-> 跨 SPI Meta-Buffer 回放
-> FOMAML 一阶元更新
-> 独立候选版本验证
-> provisional 试运行与可回滚版本
-> 已有 Module 不可变、按需刷新
```

不采用完整二阶 MAML。第一版允许从共享初始化和简单上下文编码开始，不要求大型超网络；异构观测、动作或 Policy 类不直接混合更新。

## 4. SPI 资格规则

资格检查与 Policy 训练数据分离，评估期间冻结 Policy。单次试验分类为 `success`、`failure`、`unknown`、`invalid` 或 `critical_violation`。其中 unknown 不等于失败，invalid 不进入成败分母，critical violation 是不能被成功率抵消的硬门槛。

候选实现必须先通过形式完整性检查，再按声明适用范围分层评估。成功率使用单侧 Wilson 下界，普通契约违反率使用单侧 Wilson 上界；资格还要求最低有效样本量、unknown/invalid 比例限制、范围覆盖和零 critical violation。具体数值不在本次固定。

资格决策采用三态：

- `qualified`：创建正式 Module；
- `not_qualified`：证据充分但明确不达标，或发生 critical violation；
- `inconclusive`：证据、覆盖或可判定性不足，保留待评估。

硬资格与合格 Module 之间的软性能评分严格分开。技能失败、资格失败或证据不足均不自动构成 Knowledge Bank 的环境机制反证。

## 5. SPT 候选版本规则

候选 SPT 不直接覆盖 stable 版本。更新数据和验证数据在同一次版本决策中分离；新旧版本在相同 SPI、预算、算法、划分规则及配对随机条件下比较。主指标是包含多个适应检查点、适应成本和契约风险的学习效率分数。

候选版本只有同时满足以下条件才进入 provisional：

1. 总体配对改善超过最小实际意义门槛；
2. 固定预算内的 SPI 资格通过率非劣；
3. 重要历史上下文组非劣；
4. 契约和安全风险达标；
5. 没有 critical violation；
6. 数据有效、独立且覆盖充分。

结果分为 `provisional`、`rejected` 和 `inconclusive`。provisional 仅表示离线验证通过，仍需试运行证据才能晋升 stable；rejected 表示证据充分且门槛失败；inconclusive 表示暂时无法判断，不得伪装成拒绝。

## 6. 回滚规则

SPT 版本以不可变对象保存，通过活动指针切换。严重安全或契约违反、模型或接口完整性问题、验证数据泄漏可立即回滚；一般性能退化必须在可比范围内达到系统性证据标准。回滚保留全部来源、证据和关联 SPI。

SPT 回滚不自动删除由该版本生成、但已独立通过资格检查的 Module。若只发生总体生成能力退化，Module 可以继续使用；若回滚原因使其资格、安全或契约有效性受疑，则相关 Module 进入 `quarantined`，重新资格检查后才能恢复。

SPT 更新不原地修改既有 Module。新版本主要服务后续 SPI；旧 Module 只在性能、接口或显式升级需求触发时惰性刷新，并形成新版本。

## 7. 数据使用与信息边界

`support`、`query`、`qualification` 和 `SPT validation` 分别承担 Policy 适应、元目标、Module 资格和 SPT 版本决策。每条经验必须记录其职责和版本使用历史，防止同一次决策中的训练—验证泄漏。

Meta-Buffer 属于 Skill Library。SPI 上下文、学习曲线、Policy 参数、梯度、Module 资格和 SPT 版本性能不进入 Knowledge Bank；只有第 7、8 节允许的公开环境状态证据走 `RecordEvidence` 路径。

## 8. 尚未固定的内容

- 内外循环预算与更新频率；
- Meta-Buffer 容量、配额和采样权重；
- FOMAML 的具体 Policy 优化器与损失实现；
- Module 最低样本量、成功率、违反率、unknown/invalid 阈值；
- SPT 置信水平、最小改善量、历史非劣界和资格率非劣界；
- provisional 所需证据量和回滚检测窗口；
- 完整 Module 生命周期；
- 异构接口、递归技能、并行技能和分布式训练。

## 9. 影响范围与后续

- PROJECT_BIBLE 第 9 节成为 Skill Evolution 的唯一正式算法规范。
- ROADMAP 将“定义 Skill Evolution”标为完成。
- Module 形式化决策中先前列为未定的资格、SPT 更新和必要回滚边界由本决策补充；其他未定项继续有效。
- 下一讨论项是 Knowledge Evolution 的统计更新公式、预算和确认规则，不进行代码实现或实验设计。
