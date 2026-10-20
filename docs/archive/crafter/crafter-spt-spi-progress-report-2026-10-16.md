# Crafter SPT/SPI 研究进度汇报

日期：2026-10-16

阶段：正式实验前的可行性诊断与协议准备。下述成功率来自诊断或 pilot，不是正式实验结果。

## 研究目标

当前主问题是：**学习一个采集 SPI 后，依据该经验更新 SPT，是否能提升另一个采集 SPI 在有限适应预算内的学习效率？这种收益是否超过普通策略预训练？**

本问题直接检验 Skill Evolution 的一个受控组成部分。当前不是要证明完整 KECRL，也不是要先把自然环境中的 wood 策略调到某个成功率门槛。理论边界仍为：Knowledge 保存有证据支持的环境机制，SPT 生成技能族实例，SPI 描述具体任务学习，Policy 执行动作，Module 则须通过独立资格契约后才能注册。

## 已完成工作

### 1. 框架与环境接口

- 已完成 Knowledge/Skill 分离、SPT/SPI/Module 定义及更新边界的理论设计；Knowledge Evolution、context-conditioned FOMAML、Module/Pipeline 接口和主要调用路径已实现并做过诊断。
- 已打通 Crafter RGB 输入、真实 CUDA 训练路径、策略/优化器/RNG 检查点与诊断记录。早期全量回归为 `140 passed`；当前迁移实验相关改动后的最近一次全量回归为 `442 passed`。
- 当前没有合格的自然 Crafter wood Module；没有执行活动 SPT 切换、Module 注册或 Knowledge 更新。

### 2. 自然 wood 基础策略诊断

早期 wood>=3 诊断显示基础策略尚不稳定。五 seed 的一次候选资格测量中，原网络 80 episode 平均成功率为 `0.29`，延长到 160 episode 为 `0.26`，空间 CNN+GRU、160 episode 为 `0.34`；三种设置均为 `0/5` seed 达到候选 `0.8`。后续不同预算与配置的诊断并非同一实验，不能拼接成一条学习曲线。

随后稳定 wrapper 下的另一组 100k 交互步 CNN-only / CNN+GRU 诊断均值为 `0.533 / 0.500`，仍未稳定达到候选资格线。由于任务采样和协议不同，这些数字只作为各自诊断的结果，不能与上面的 episode 数实验直接比较。

PPO 更新单位校准在 3 seed、每臂 8,192 交互步下比较 episode 更新与固定 1,024 步更新。Qualification 均值分别为 `0.250` 和 `0.217`，未显示固定步数 batch 有稳定收益；两臂 KL 很小、clip fraction 均为 0。失败以死亡为主（episode 更新 109 次，固定步数 130 次），说明该校准没有解决自然环境执行问题。

后续 wood 诊断逐步拆查了局部动作链、RGB 状态读出、动作头、状态覆盖和死亡阶段。脚本在审计布局上可通过原生动作完成采集，部分规则辅助的局部动作衔接也有高成功率；但这些结果使用 oracle/监督标签，只验证动作链可执行或提供诊断信号，**不代表学习策略成功，更不构成 Module 资格**。普通 RGB 策略在 approach、turn、do 等阶段仍缺少稳定的条件化动作选择。当前证据不能把低成功率单独归因于视觉表示、head、状态覆盖、奖励/信用分配或生存中的任一项。

### 3. 实例化 SPT→SPI 迁移链路与 pilot

已在受控 `quiet-material-gather-v1` 场景中实例化 `gather` 技能族，包含 tree→wood、stone→stone、coal→coal 三个目标绑定。训练和评价仍使用原生 Crafter RGB、移动、转向、`do`、工具检查和库存变化；策略不接收坐标、路径或几何有效动作集合。为隔离采集技能，本协议统一提供木镐、充足初始生存资源，并移除敌人、水和熔岩，因此不测试工具制作或自然生存。

已跑通实际 `SPT → Instantiate → SPI → support adaptation → independent evaluation` 路径，并比较：

| 臂 | 初始化来源 | 作用 |
|---|---|---|
| A | 更新前 SPT | 无新增来源经验的基线 |
| B | 来源 SPI 经验经 context-conditioned FOMAML 更新后的 SPT | 检验 SPT 更新后的迁移 |
| C | 同等来源任务和交互预算的普通 PPO 预训练 | 判断收益是否只是普通预训练 |

E1 wood→coal v3 是两 seed 校准 pilot：来源 B/C 各 8,192 步，目标适应各 4,096 步，每个检查点每 seed 40 个评价 episode。终点成功数为 A `15/80`、B `7/80`、C `19/80`；240 个终点评价 episode 中有 41 次成功、199 次外部截断、0 次死亡。**该 pilot 没有显示 B 优于普通预训练 C，也不能据两 seed 断言 SPT 更新无效。**

### 4. 可比性、复现与分析器

- 检查发现并修复过来源外循环学习率被普通 PPO 学习率覆盖的问题；之后将实际优化器学习率纳入账本和独立审计。
- v2 复现暴露 B/C 来源场景池不一致；v3 改为声明配对的 support/query 候选池，并通过严格场景池审计。相同环境 seed 不承诺不同策略产生相同轨迹，审计重点是候选场景池、角色隔离与 episode provenance。
- E2 三折 runner 已覆盖 holdout coal、stone、wood。候选统计分析器实现了逐 seed/fold 成功率曲线、归一化 AUC、seed 内三折等权平均、B−C 配对主对比及按 seed 成组 bootstrap。bounded smoke 仅使用一个 seed，在目标支持预算 0 步和 8 步各评估两个 episode，只验证运行边界；分析器正确返回 `not_estimable`，没有估计方法效应。
- 两次独立进程的 E2 bounded smoke 在去除路径和耗时字段后汇总一致，9 个策略 checkpoint 的 SHA-256 一致；RGB 重建、目标材料绑定、B/C 候选场景池和动作 seed 审计通过。
- E2 候选成本/provenance 预检通过，候选 E1+E2 交互步数上限约 `9,128,960`。候选 seed schedule 覆盖 3 折、30 个 replica-fold、3,840 个来源槽和 15,000 个配对评价 episode。本机 `/tmp/crafter_gather_spt_transfer_e2_candidate_manifest_plan_2026-10-16_final.json` 的 SHA-256 为 `0cbabf7802be948d36597e8eb8318952d533db39537fd815d4f146a4d397ae68`；它仍是未封存草案。

## 已验证结论与解释边界

1. 自然 wood 基础策略仍不足以通过候选资格门槛；早期增加 episode 数或引入 GRU没有显示稳定改善。候选 `0.8` 不是当前 SPT 迁移实验的启动门槛。
2. 当前组件已能端到端运行真实 SPT 更新、SPI 实例化、策略生成、目标适应和独立评价；这证明实验路径已实例化，不等于证明 SPT 更新有迁移收益。
3. E1 校准结果中 B 低于 A、C，且评价成功率整体偏低。它只提示需要按预定预算继续检验；没有正式方法优劣结论。
4. v3 修复并验证了已发现的来源学习率和候选场景池边界问题。不同策略访问到的 episode 仍会不同，这是策略行为造成的轨迹差异，不能声称轨迹完全匹配。
5. 受控场景没有死亡、工具制作、敌人或危险地形，因此当前迁移实验不能回答自然 Crafter 的生存、挖矿工具链或持续任务能力。
6. stone、coal 目前进入的是受控绑定/留出设计；尚无自然环境下 stone、coal、pickaxe 或完整挖矿链的学习策略资格结论。

## 当前难点

- **学习信号弱且有限预算 pilot 方差明显。** E1 两个 seed 的成功率低，B/C 的小样本排序不稳定，尚不足以判断元更新是否带来可重复收益。
- **自然执行瓶颈尚未完全归因。** 局部动作、表示读出和覆盖诊断提供了线索，但离线读出或 oracle 正对照不能替代自主轨迹；存活、导航、阶段识别及稀疏奖励信用分配仍可能共同作用。
- **公平比较需要严格记录边界。** 过去已出现学习率覆盖和场景池不一致；这些缺口已在候选实现中增加审计，但正式 manifest、代码/依赖哈希和统计版本尚未封存。
- **组件结果与自然任务存在范围差异。** 统一提供木镐可隔离 gather 的目标绑定迁移，却绕过了工具制作、材料前置关系和危险环境；后续必须另外验证自然技能链。
- **成本与正式准入仍待决定。** 当前预算为候选值，正式实验尚未启动；单次 smoke 不能替代正式 seed 数和评价预算。

## 下一步计划

1. **冻结协议候选。** 审阅并锁定 E1/E2 的任务边界、seed 数、来源/目标步数、checkpoint、评价 episode 数、唯一主指标及统计代码版本。确认代码、配置、依赖和 manifest 哈希，并由独立进程复核；候选值在封存前仍允许修订，修订必须另存版本。
2. **完成正式前准入检查。** 确认三臂初始化与模型容量匹配、B/C 来源场景候选池及动作 seed 可审计、目标 optimizer 重置、评价 RNG 隔离、折间独立和成本可承受。当前 `formal_freeze_ready=false`，因此尚不进入正式训练。
3. **冻结后运行 E1 与 E2。** E1 回答 wood 经验到 coal SPI 的直接迁移；E2 在 wood/stone/coal 三种绑定上轮流留出，是主要的跨绑定检验。按固定预算完成全部 seed，不根据最终评价挑 checkpoint 或追加有利 seed；按预注册 AUC 和 B−C 主对比报告，并同时给出 B−A、逐折曲线、死亡/截断、目标外采集与成本。
4. **组件实验后再做自然系统验证。** 若要把结果扩展为可执行 SPI/Module，需要在自然 Crafter 中独立验证木材采集、工具制作、stone/coal 开采、前置条件组合、风险生存、持续任务链和 Module 契约；再按项目理论边界评估 SPT 接受、Knowledge 更新和 Pipeline 调用。当前受控迁移结果不能替代这些验证。

## 参考记录

- [最新诊断状态与 E2 实现验证](crafter-diagnostic-status-2026-10-15.md)
- [SPT/SPI 迁移协议候选](crafter-spt-spi-transfer-protocol-2026-10-15.md)
- [E1 集成 pilot 与复现结果](crafter-gather-spt-transfer-pilot-2026-10-15.md)
- [采集动作链及学习率、场景池审计](crafter-gather-execution-chain-2026-10-15.md)
- [PPO 失败归因与更新单位校准](crafter-ppo-failure-attribution-calibration-2026-10-08.md)
- [wood 状态覆盖 pilot](crafter-wood3-spatial-coverage-2026-10-15.md)
