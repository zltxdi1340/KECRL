# Crafter 可行性诊断状态（2026-10-15）

## 当前结论

当前仍在正式实验前。最新完成的是 gather 族 E1 wood→coal 的非正式 SPT→SPI 集成 pilot：两个 replica、A/B/C 三臂，B/C 各 8,192 source 步，三臂各 4,096 target 步；canonical run7 与独立审计结果见[集成记录](crafter-gather-spt-transfer-pilot-2026-10-15.md)。此前 seed-0 的 episode 隔离状态覆盖 pilot 仍保留：原始 CNN-only 成功 25/40，旧/新各 216 行动作头均为 8/40，合并 432 行动作头为 16/40。新增代码的定向测试为 `9 passed`；提交 `f1f49e4` 后需以最终代码版本重新执行全量回归。

gather pilot 的步数账本、SPT/SPI 版本绑定、CNN 生成器、候选隔离、全新 target optimizer、coal 冷列和 RGB 可见性审计均通过；B/C 的共享 template 与 generator 确实发生更新。canonical run7 的 coal 评价终点为 A `15/80`、B `7/80`、C `8/80`，失败全部是 external truncation，未观察到死亡。两 seed 方差大，B 没有稳定超过 A/C，所以该 pilot 只证明链路可运行，不支持 SPT 优势或劣势的正式结论。独立结果审计返回 `valid=true`。

合并数据有改善方向，但同时改变样本量与覆盖，仍弱于原策略。旧/新 216 行两臂都选 epoch 0，不能把相同成败标签当作已排除覆盖问题。这些实验仍是 wood Policy 后端诊断，不是跨 SPI 的 SPT 效果验证。

前一轮空间 head 的同批对照为 source 24/40、final embedding 13/40、conv2 22/40；这组结果使用不同 fresh seed，不能与最新结果拼成因果改善曲线。其同状态 cross-readout 显示，final head 在自己的访问状态上的 turn 有效概率质量约 6.0%，而 source/conv2 在同一状态池约 13.9%/12.7%。状态覆盖与条件泛化仍未解决，不能归于单一表示或生存因素。

本轮的两个 head 只学习六个非-do 动作；原 CNN、actor/critic 和优化器冻结，`p(do)` 在每帧复用 source 策略概率，最大实测差 `1.19e-7`。局部几何 action-set 标签用于 216 条 train 状态拟合，也用于 115 条 validation 状态选择；fresh 执行没有 oracle，没有动作替换。死亡和截断全部保留，未执行生存干预或死亡原因分层。配置中的 `teacher_scope` 字段没有明确包含 validation 选择，详细报告已更正此范围说明。

## 已完成的同状态检查

已将三臂 120 集 fresh 轨迹逐步恢复为 15,550 个 RGB 状态，并让三个冻结策略在同一状态池上交叉输出 46,650 条动作概率。恢复使用封存动作，不采样或执行新动作；主运行、跨进程重复、独立 analyzer 和输入哈希检查全部通过。详细结果见[同状态 cross-readout](crafter-wood3-spatial-cross-readout-2026-10-15.md)。

阶段/死亡前窗口的 episode 宏平均显示，final head 的 27 次死亡里 18 次在第一根 wood 前，source/conv2 分别为 4/8 次。冻结 source critic 在 source path 上的 value 随 wood 阶段降低，并在死亡前 approach 状态转负；与 `gamma=0.99` Monte Carlo return 的绝对误差仍约 0.23–0.61，且死亡前 turn 样本少。因此 value 有粗粒度阶段信号，但不足以排除 reward/credit 问题，也不能把 survival 归因为唯一瓶颈。

## 研究主线调整与下一项

上述按 episode 隔离的覆盖实验已完成。等样本量覆盖扩展只有未完成草案，尚未运行，暂不作为自动下一项。后续优先形成直接检验 Skill Evolution 的实验：学习采集实例后更新 SPT，是否提高新采集 SPI 的学习效率，并与普通预训练比较。

新的[SPT/SPI 迁移协议草案](crafter-spt-spi-transfer-protocol-2026-10-15.md)与设计清单已形成并完成一次 E1 集成 pilot：同一 gather 族含 wood/stone/coal 三个绑定；E1 检验 wood->coal，E2 轮流留出一个绑定；A 为更新前模板，B 为 FOMAML 候选，C 为等容量、等来源交互预算的普通预训练。统一工具、多材料受控场景和 CNN 生成器链路已接通；正式 E1/E2 仍需在冻结 manifest、独立进程复核和成本检查后启动。

关键缺口已从“接口未连接”转为“正式实验前的可比性和学习信号审计”：pilot runner 使用真实 RGB rollout、CNN actor 残差生成器、FOMAML query 和普通 PPO 对照，但当前预算下迁移收益不稳定。协议中的预算与统计设置仍未冻结；正式迁移实验尚未启动。

无需 wood 达到候选 0.8 才能研究 SPT；SPI 学习失败也可提供反馈。受控组件实验不等于自然 Crafter Module 资格或完整 KECRL 验证。当前没有新的活动 SPT 切换、Module 注册或 Knowledge 更新；自然 stone、coal、pickaxe、工具获取、生存及持续任务链仍需独立验证。

详细协议与配对结果见[空间动作头 fresh 诊断](crafter-wood3-spatial-actor-fresh-2026-10-15.md)。
