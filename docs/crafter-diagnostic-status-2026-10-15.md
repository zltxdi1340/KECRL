# Crafter 可行性诊断状态（2026-10-15）

## 当前结论

当前仍在正式实验前。最新完成 gather 族 E1 wood→coal 的动作链审计、学习率修正、来源场景池对齐 pilot、独立进程复现、E2 三 fold runner smoke 和 E2 candidate 成本/provenance 预检：E1 使用两个 replica、A/B/C 三臂，B/C 各 8,192 source 步，三臂各 4,096 target 步；E2 smoke 使用 holdout_coal/stone/wood、每 fold 一个 seed。主结果为 `results/crafter_gather_spt_transfer_pilot_v3_run1/`（代码 `e43d678`），复现为 `results/crafter_gather_spt_transfer_pilot_v3_run2/`（代码 `155b204`），E2 smoke 为 `results/crafter_gather_spt_transfer_e2_smoke_v1/`。本轮定向测试 `20 passed`，全量回归 `438 passed`。

v1 曾将 B 的配置外循环 lr=0.001 覆盖为实际 0.0003，已修复并记录 optimizer 实际值。v2 独立进程复现的数值记录一致，但严格 source scene pool 审计失败：C 只使用了 B support 池的实际前缀并增加其他场景，B query 池未进入 C。v3 声明并配对 support/query 候选池，provenance 和严格场景池审计均通过。v3 终点为 A `15/80`、B `7/80`、C `19/80`，共 41 次成功、199 次 external truncation、0 死亡。v2 的 B `25/80` 未复现，当前没有 SPT 优势证据。

动作链机械正对照覆盖 32 个布局、三种材料：最短原生路线为 1--6 步，8/16/32 步均 96/96 成功。六个冻结 coal Policy 在相同 83 个 RGB 状态上读取 498 次动作分布，各策略 probe 前后参数哈希不变。修复后的 B 在 approach/turn/do 状态都偏向 do，阶段区分仍弱。这定位了执行表现，尚不能区分表示、head、状态覆盖、优化和奖励/信用分配的责任；脚本正对照不是学习策略成功率，stone 也未进行学习策略评价。

此前 seed-0 的 episode 隔离状态覆盖 pilot 仍保留：原始 CNN-only 成功 25/40，旧/新各 216 行动作头均为 8/40，合并 432 行动作头为 16/40。

合并数据有改善方向，但同时改变样本量与覆盖，仍弱于原策略。旧/新 216 行两臂都选 epoch 0，不能把相同成败标签当作已排除覆盖问题。这些实验仍是 wood Policy 后端诊断，不是跨 SPI 的 SPT 效果验证。

前一轮空间 head 的同批对照为 source 24/40、final embedding 13/40、conv2 22/40；这组结果使用不同 fresh seed，不能与最新结果拼成因果改善曲线。其同状态 cross-readout 显示，final head 在自己的访问状态上的 turn 有效概率质量约 6.0%，而 source/conv2 在同一状态池约 13.9%/12.7%。状态覆盖与条件泛化仍未解决，不能归于单一表示或生存因素。

本轮的两个 head 只学习六个非-do 动作；原 CNN、actor/critic 和优化器冻结，`p(do)` 在每帧复用 source 策略概率，最大实测差 `1.19e-7`。局部几何 action-set 标签用于 216 条 train 状态拟合，也用于 115 条 validation 状态选择；fresh 执行没有 oracle，没有动作替换。死亡和截断全部保留，未执行生存干预或死亡原因分层。配置中的 `teacher_scope` 字段没有明确包含 validation 选择，详细报告已更正此范围说明。

## 已完成的同状态检查

已将三臂 120 集 fresh 轨迹逐步恢复为 15,550 个 RGB 状态，并让三个冻结策略在同一状态池上交叉输出 46,650 条动作概率。恢复使用封存动作，不采样或执行新动作；主运行、跨进程重复、独立 analyzer 和输入哈希检查全部通过。详细结果见[同状态 cross-readout](crafter-wood3-spatial-cross-readout-2026-10-15.md)。

阶段/死亡前窗口的 episode 宏平均显示，final head 的 27 次死亡里 18 次在第一根 wood 前，source/conv2 分别为 4/8 次。冻结 source critic 在 source path 上的 value 随 wood 阶段降低，并在死亡前 approach 状态转负；与 `gamma=0.99` Monte Carlo return 的绝对误差仍约 0.23–0.61，且死亡前 turn 样本少。因此 value 有粗粒度阶段信号，但不足以排除 reward/credit 问题，也不能把 survival 归因为唯一瓶颈。

## 研究主线调整与下一项

上述按 episode 隔离的覆盖实验已完成。等样本量覆盖扩展只有未完成草案，尚未运行，暂不作为自动下一项。后续优先形成直接检验 Skill Evolution 的实验：学习采集实例后更新 SPT，是否提高新采集 SPI 的学习效率，并与普通预训练比较。

新的[SPT/SPI 迁移协议草案](crafter-spt-spi-transfer-protocol-2026-10-15.md)与设计清单已形成并完成一次 E1 集成 pilot：同一 gather 族含 wood/stone/coal 三个绑定；E1 检验 wood->coal，E2 轮流留出一个绑定；A 为更新前模板，B 为 FOMAML 候选，C 为等容量、等来源交互预算的普通预训练。统一工具、多材料受控场景和 CNN 生成器链路已接通；正式 E1/E2 仍需在冻结 manifest、独立进程复核和成本检查后启动。

E2 runner 已实现并完成 bounded smoke：三个 leave-one-binding-out fold 的 resolved config、source material 绑定、target holdout、B/C paired candidate pool 和 RGB provenance 审计均通过。candidate manifest 的成本/provenance 预检也通过，E2/E1/合计交互上限为 `6,846,720 / 2,282,240 / 9,128,960`，跨 replica 角色 seed 区间无碰撞；但 `formal_freeze_ready=false`，正式多绑定实验的最终 hash 封存、统计冻结和启动决定仍未完成。正式迁移实验尚未启动。

无需 wood 达到候选 0.8 才能研究 SPT；SPI 学习失败也可提供反馈。受控组件实验不等于自然 Crafter Module 资格或完整 KECRL 验证。当前没有新的活动 SPT 切换、Module 注册或 Knowledge 更新；自然 stone、coal、pickaxe、工具获取、生存及持续任务链仍需独立验证。

详细协议与配对结果见[空间动作头 fresh 诊断](crafter-wood3-spatial-actor-fresh-2026-10-15.md)。

## 2026-10-16 E2 主分析实现验证

E2 分析器现除逐折 RGB/provenance 与来源场景池审计外，也按候选协议实现主分析：从每个 checkpoint 的逐 episode 成功标签重算成功率和归一化梯形 AUC；每个 replica 内先等权平均三个 holdout fold，再计算唯一主对比 B−C；95% percentile 区间按 replica seed 成组重采样 10,000 次，随机种子 `20261015`，bootstrap 分位数采用 linear 插值。输出保留完整 seed/fold 曲线、逐 fold 均值与样本标准差、配对差，并将未达到 `0.8` 的检查点标为右删失。分析同时核验各臂在相同 fold、seed、checkpoint 使用相同 scene/action seed，记录摘要与 episode 标签一致，并在 source/target/evaluation 角色之间检查 canonical scene record 与初始 RGB digest 碰撞。

三折 E2 smoke 的 provenance、RGB 重建、B/C 场景池和材料留出审计均通过；它使用一个 seed、`0/8` 支持步和每点两个 episode，因此分析器明确返回 `not_estimable`，没有计算主效应。缩小预算的合成三折测试验证了 seed 内折平均和成组 bootstrap。当前定向测试 `24 passed`，全量回归 `442 passed`，使用 `kecrl` Conda 环境（Python 3.10.22、PyTorch 2.14.1+cu130）。

这只封住了候选分析实现和 smoke 边界，不代表统计协议、输入 manifest 或正式实验已经冻结；当前协议仍标记为 candidate，正式训练未启动。正式启动前仍需完成一次提交后的全源文件/依赖哈希封存、最终 manifest 核对与独立复核，并明确启动决定。

## 来源 seed 池与动作 RNG 核验补充

检查 runner 后发现，64-scene 来源候选池小于单块 256 个交互步的最坏 episode 数；每集至少消耗 1 步，因此最坏可达 256 集，旧池存在耗尽风险。正式候选现扩大为每 role 256 个 scene seed，并采用 query offset 300、slot stride 600；preflight 和 runner 都验证容量和不重叠。

旧训练收集器以 block 为单位设置 action RNG，却没有把该 block seed 写进每集 manifest。正式候选改用 `block_action_seed + episode_index` 的逐集 action seed，并在结果审计中校验，旧 pilot 仍按其原 block-stream 规则作为校准数据，不与正式结果合并。更新后的三折 bounded smoke 已在独立进程重复两次：逐 episode seed、材料绑定、RGB 重建、B/C 来源池相等审计全部通过；去掉路径和 elapsed time 后 aggregates 完全一致，9 个 Policy checkpoint SHA-256 一致。由于只有一个 seed 和 0/8 支持步，不计算主效应。全量回归 `442 passed`，正式运行仍未启动。

候选 preflight 现在附带哈希绑定的 seed-schedule manifest 草案，明确列出 3 个 fold、30 个 replica-fold、3,840 个 source slot、1,920 个 target-support block 和 15,000 个配对 evaluation episode。候选 scene/action seed 角色区间、256 scene pool 容量和预算均通过；manifest 仍标记为未封存，当前工作区有 tracked 修改和未跟踪路径，不能当作正式冻结快照。

## 2026-10-18 动作 RNG 与 E2 候选回归核验

本轮核验当前 E2 候选改动，不启动正式预算训练。运行环境为 `kecrl` Conda 环境（Python 3.10.22、PyTorch 2.14.1+cu130、2 张 RTX 4090）；定向测试 `41 passed`，全量回归 `442 passed`，`git diff --check` 通过。

逐集 action seed 改动通过两个全新进程的三折 bounded smoke。两次 smoke 都使用 seed 11、source support/query 各 4 步、target checkpoint `0/8` 和每个 checkpoint 2 个评估 episode；输出位于 `/tmp/crafter_gather_spt_transfer_e2_smoke_20261018_codex/` 与 `/tmp/crafter_gather_spt_transfer_e2_smoke_20261018_codex_run2/`。独立分析器对两次运行的全部 fold 均确认 comparability、provenance 和严格来源场景池相等有效；逐折训练记录一致，9 个 target Policy checkpoint 的 SHA-256 一致。

分析器按预期返回 `not_estimable`：smoke 的 seed、checkpoint 和 episode 数不满足正式候选预算，因此没有计算主效应。候选 preflight 输出位于 `/tmp/crafter_gather_spt_transfer_e2_candidate_preflight_20261018.json`，结构、成本和 seed-schedule 检查通过；候选统计配置仍为 `ready_for_formal_training=false`。此外，分析器目前将 `formal_freeze_ready` 固定输出为 `false`，所以它是 candidate 一致性/成本预检，而不是能自动放行正式实验的门控。当前 tracked 文件有修改且存在未跟踪路径，hash 快照也未封存。正式前还需落实并冻结统计/样本方案、核对最终 manifest 与依赖版本、形成可复现提交快照，并单独明确是否启动正式训练。

## 2026-10-18 候选功效与成本审查

按 E2 seed 级三折平均 B−C AUC 做正态敏感性模拟：候选 10 seed 在 `Δ/σ=0.67` 时 95% percentile-bootstrap 区间排除 0 的比例约 0.61；若 `Δ=0.10, σ=0.15`，约 18 seed 才接近 0.8，若 `σ=0.20` 则约 30 seed。两个 E1 pilot seed 不足以估计 E2 方差，因此没有更改候选 seed 数。按两次 pilot 每秒约 677–680 个实际环境步线性外推，当前 E1+E2 最大预算约需 3.7–3.8 小时单卡串行；不是保证上限。计算假设、敏感性表与限制见[功效和成本审查](crafter-spt-spi-power-cost-audit-2026-10-18.md)。

为辅助选样，E1+E2 全预算的线性换算为 10/18/30 个 seed 对应约 3.7/6.7/11.2 小时单卡串行；该换算按候选每 seed 固定成本缩放，未增加新的训练数据或正式实验结果。

用户确认启动后，已冻结 formal v1 配置并启动 30-seed E1/E2 正式运行：E1 在 GPU 0，E2 在 GPU 1。冻结哈希、输出位置和运行状态见[正式运行记录](crafter-spt-spi-formal-run-2026-10-18.md)。当前无完整结果，正式效果必须等待训练完成和独立审计后再解释。

首分钟状态检查已看到 E1 seed 101 的 A 臂及 E2 `holdout_coal` seed 101 的 A 臂写出结果账本与 checkpoint；两进程继续运行。该部分臂记录未作效果分析，剩余 B/C 臂、其他 seed/fold 与独立完整结果审计仍待完成。
