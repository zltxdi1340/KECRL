# Crafter 可行性诊断状态（2026-10-13）

后续 ready-only 蒸馏对照已完成；最新进度见
[2026-10-14 诊断状态](crafter-diagnostic-status-2026-10-14.md)。下文保留本日诊断记录。

当前正把 wood 的 SPI executor 从诊断规则转换成自主 RGB 动作链。SPT/SPI、
Knowledge/Skill、Module 和 Pipeline 的理论及接口已实现；当前阻塞点仍是可执行
基础 Policy，没有满足完整资格契约的真实 wood Module。

## 当前完成与验证

1. 稳定环境 wrapper 与 CUDA 确定性路径已经通过跨进程回放；正式比较以交互步数
   记录预算。已有 100k CNN-only / CNN+GRU wood 诊断均值为 0.533/0.500，
   没有达到候选 0.8 的稳定 Module。GRU、固定死亡惩罚及已有辅助损失尚无稳定收益。
2. 受控局部场景、自然迁移、匹配状态覆盖、动作后果、ready-do、turn 和 approach
   增量干预逐项完成。共同 ready/turn 门控之外增加 approach，三个 seed 的两种
   head pooled 成功率为 93.3%/92.2%，说明正确局部动作衔接可以显著改善 wood。
   这些高成功率含 oracle，不能算作真实 RGB executor 资格。
3. 最新[动作链审计](crafter-wood3-rgb-chain-audit-2026-10-13.md)复核 360 集、21833
   事件。旧 head 原采样的 approach/turn/ready 动作仍不稳定，约三分之二的
   oracle approach 状态粗几何组合未见于实际训练集；覆盖不足与读出泛化都需定位。
4. 最新[自主学习 pilot](crafter-wood3-rgb-chain-learning-diagnostic-2026-10-13.md)
   冻结 seed 0 的原 CNN，等类别状态数和优化更新数比较 native-state 与 chain-state
   线性 head，oracle 只用于监督采集和标签。全新自然 episode 上关闭动作替换后：
   原 actor 12/30、native-state 3/30、chain-state 4/30。局部保留集部分读出改善没有
   转化为完整任务收益；两个副本都出现 ready/do 与自然动作链退化。
5. 学习 pilot 的全量回归 `380 passed`；追加同状态交叉读出后的全量回归 `384 passed`。
   pilot 在独立进程完整重复，数据、特征、head 和轨迹
   digest 一致；独立分析器重建监督数据、特征、离线指标及自然执行 RGB/动作回放。
   策略/PPO optimizer、源 checkpoint 与 Crafter 安装包封存哈希保持不变。
6. 对 pilot 的 90 集自然轨迹做了同状态交叉读出：三种行为策略各自的 RGB 状态
   都由三个 frozen head 评分。native 轨迹上原 head 的 ready/do 有效概率质量为
   66.2%，native 拟合 head 只有 2.4%；chain 轨迹的转向状态上三个 head 贪心正确率
   都低于 9.3%。因此能力遗忘与新状态转向读出困难同时存在，不能只归因于死亡。

## 当前判断与下一项

当前有明确的局部执行瓶颈与泛化退化证据。单 seed pilot 没有支持“只换成 oracle
采集链覆盖就能解决”；训练到保留、离线到自然执行的落差，也不足以证明 RGB 本身
信息不足或 head 容量不足。奖励与信用分配尚未在控制其余因素时独立识别。

交叉读出已完成。下一项用新 episode seed 做最小的保留能力对照：在相同 RGB
监督数据上比较普通动作集合拟合与加入原 actor ready/do 输出蒸馏的 head，使用新的
validation/heldout/fresh seed；fresh 执行只看 RGB，不加动作规则。这检验能否学习
approach/turn 而不抹掉原有 do 能力。若同状态上的转向仍对所有 head 都困难，再做
RGB 表示诊断；局部自主链改善而剩余失败主要由死亡阻断时，转入跨任务共享生存
能力研究，不把总成功率 0.8 当作启动生存研究的硬条件。

当前环境不存在摔落或溺水机制；全部原生失败死亡和外部截断仍计入总成功率。
生存、探索和局部采集的指标分别解释，不能通过删除失败获得资格结果。

当前仍未注册 Module 或更新 Knowledge/SPT。wood 修通以后还需要验证挖矿、
工具前置条件、组合调用及持续任务链；本轮没有 stone/coal/pickaxe 结论。

GitHub 归档源码、配置、测试、中文结果与紧凑验证摘要；checkpoint、监督 head、
RGB/特征及逐步轨迹留在实验机器。因此 GitHub 新检出包含复现协议与入口，不能
直接恢复所有本机数据。`.vscode/` 不纳入提交。
