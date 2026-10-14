# Crafter 可行性诊断状态（2026-10-14）

当前正在验证 wood 的 SPI executor 能否自主完成 RGB 动作链。理论、SPT/SPI、
Knowledge/Skill 分离及 Module/Pipeline 接口已实现；真实自主 Policy 仍是阻塞点，
没有通过完整契约资格检查的 wood Module。

## 已完成与已验证

稳定 wrapper、真实 CUDA 和跨进程确定性已验证。100k CNN-only/CNN+GRU 的先前
三 seed 诊断均值为 0.533/0.500。固定死亡惩罚、已有辅助损失和增大 head 容量
尚无稳定收益。局部 oracle 衔接 approach/turn/do 后曾获得高成功率，说明动作链
值得研究；含 oracle 的结果不能用作自主 executor 或 Module 资格。

前一轮 RGB 学习 pilot 及同状态交叉读出表明，原 CNN 上单独拟合局部动作 head
可能破坏已有 do 行为，并在新策略自行访问的 turn/approach 状态上泛化失败。

最新[ready-only 蒸馏诊断](crafter-wood3-rgb-chain-ready-distillation-2026-10-14.md)
已完成。冻结 seed 0 的原 100k CNN，以同一批 216 行 native RGB 和相同初始化、
优化预算比较普通集合拟合与在 ready 行上蒸馏原 actor 分布；使用全新的
validation、heldout 和配对 fresh episode seed。fresh 只看 RGB，无动作替换。

| 策略 | 新 fresh 成功率 | 死亡 | 外部截断 |
| --- | ---: | ---: | ---: |
| 原 actor | 13/30（43.3%） | 17 | 0 |
| 普通集合拟合 | 5/30（16.7%） | 23 | 2 |
| ready-only 蒸馏 | 10/30（33.3%） | 20 | 0 |

蒸馏降低了训练 ready 的 teacher 分布偏移，但没有可靠保留新状态上的 do。
在蒸馏轨迹的相同 ready 帧上，原 actor 的 do 概率质量为 66.2%，蒸馏 head
为 11.4%；按 episode 平衡为 65.5% 对 23.1%。同一批 turn 帧上三种 head
贪心有效率均不超过 10.3%。这支持继续定位局部泛化缺口；不能把死亡作为唯一
解释，也不能宣布蒸馏普遍无效或 encoder 缺少相关信息。

跨进程完整重复通过。独立分析器重建新旧监督行、特征、离线指标、验证选择，
逐步复核 32 集新监督轨迹和 90 集 fresh 轨迹；实际动作及概率完全一致。
9177 个局部状态由三个 head 同时评分，共 27531 条记录。策略、PPO optimizer、
checkpoint 和安装包哈希未变。全量回归 `391 passed`；GPU 空闲。

## 当前限制与下一项

本轮是一个来源策略 seed 的诊断 pilot。10/30 高于 5/30，仍低于原 actor；
没有跨来源策略 seed 的稳定收益证据。新 episode seed 与训练/验证/保留 RGB
仍有少量精确重合，帧数和交叉评分行数不能充当独立 episode 样本数。
完整任务统计保留所有死亡与外部截断，不删除失败来提高资格分数。

下一项检查**自然失败状态的分项空间信息可读性**：相同 RGB/标签上比较冻结
CNN 最终表示与固定玩家中心局部 RGB，分别检查树方向、玩家朝向和可执行接近
方向，按几何与昼夜/资源条件分组。训练/验证/保留按 episode 隔离，使用新保留
seed，容量差异单独报告。该离线 probe 不接管 Policy；据结果决定修表示、
补覆盖或修改 head 学习目标，避免直接重复扩大 CNN/GRU。

局部自主链改善后，继续研究跨任务共享生存能力；总成功率达到候选 0.8 不是
启动生存研究的硬条件。奖励和信用分配的独立贡献仍待控制对照。当前环境没有
摔落/溺水机制，死亡统计保留实际原生原因。

当前没有正式 PPO 训练、Module 注册或 Knowledge/SPT 更新；0.8 仍是待验证
候选门槛，不能代替完整 Module 契约。wood 修通后还需验证挖矿、工具前置条件、
组合调用和持续任务链；本轮没有 stone/coal/pickaxe 结论。

GitHub 归档代码、协议、测试、中文备注和紧凑分析。checkpoint、RGB、特征、
head 和逐步轨迹留在本机；新检出不包含完整原始实验数据。`.vscode/` 不纳入提交。
