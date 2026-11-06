# LIBERO task 4/7 机制证据汇总与下一假设选择（2026-11-01）

已有证据支持“离线专家动作拟合与自由运行控制之间仍有缺口”，没有把低成功率唯一归因于视觉、夹爪、状态漂移或预算。局部 2× gripper 候选已按规则停止，保持 CNN-only uniform-loss、500-update baseline。

**选择时的下一项为共享参数上的平移—夹爪优化干扰**：用六个现有 baseline 的固定权重做 source-support 梯度读出，区分梯度方向冲突与全局裁剪缩放。它与“夹爪权重不够”是不同的机制假设，不通过搜索附近权重继续已关闭的候选。本文保留选择时的支持/反证与设计；后续梯度诊断现已完成，更新见文末及[执行报告](libero-task4-task7-gradient-interference-diagnostic-2026-11-01.md)。

## 本轮做了什么与证据范围

检查工作区、现行报告、Policy/局部损失 runner，并重新核对下列 11 个既有结果文件的 SHA-256；从局部损失原结果的 6,000 个逐 update 梯度范数重算裁剪次数。未改变历史 artifact、checkpoint、代码、训练配置或数据。新增 optimizer updates、环境 reset 和 action steps 均为 **0**。

当前较强的配对证据来自 task 4/7、训练 seeds `20261022/20261023/20261024`。历史 task 3/7 单 seed、phase-balanced、runtime-native prefix 等结果单列作为线索，不与当前完整 causal CNN-only baseline 合并统计。source `meta_support` demos `0..29` 与 `meta_query` demos `30..49` 是不同示教角色；后者不是 official evaluation-state 编号，也不是未访问过的最终测试集。candidate states `0..19` 已反复使用，不称 unseen。

## 支持证据、反证与未解决问题

下表“反证”指限制某个解释的实际观察，不表示排除了该机制的所有版本。证据强度取决于具体对照、数据角色和覆盖范围。

| 机制解释 | 支持它的已有证据 | 反证或限制 | 仍未解决／当前决定 |
|---|---|---|---|
| 动作—观察时序错误是瓶颈 | [causal 对照](libero-bc-causal-alignment-diagnostic-2026-10-22.md)已确认 stored `obs[t]` 在 `action[t]` 后记录，不能当动作前输入。 | 当前 baseline 已用真实首帧及 `obs[t-1] → action[t]`，仍只有 `7/120` 闭环成功。 | 时序修正必要，但不充分；不再把已修正的 same-index 错配当现行训练原因。 |
| 历史数据与当前 runtime 不一致 | [语义审计](libero-replay-semantics-audit-2026-10-25.md)、[重复 replay](libero-current-runtime-replay-audit-2026-10-25.md)中，四条 task 3/7 示教均在首动作后与 HDF5 分叉，三次当前运行却各自确定；包括 replay 成功控制。 | 已检查 controller、频率和动作边界一致；不能指定某个版本为原因。全量 source-support replay 的 `17/150` 是完整动作前缀覆盖数，另 `133/150` 因当前成功边界提前结束，不能说其余示教全部 replay 失败。 | 双协议边界仍未解决；[完整前缀覆盖](libero-source-support-replay-feasibility-audit-2026-10-25.md)不足，不能直接用 runtime-native 替换完整训练输入。 |
| source 数据缺少关系、阶段或 close 类别 | [条件覆盖](libero-conditional-coverage-audit-2026-10-24.md)发现阶段比例和动作分布存在条件差异；当前专家目标误差在 open 切换窗口集中。 | 五个 source 任务两角色具有相同的关系×几何类别且均含三个命令阶段。早期 [task 7 gripper 分解](libero-runtime-native-prefix-task7-gripper-output-diagnostic-2026-10-27.md)显示 support 大量包含 close，策略也能拟合，却在部分 candidate 上不 close。 | 简单的类别缺失不足以解释；细粒度视觉几何、动作历史与闭环访问状态覆盖仍未确定。命令阶段不等于物理抓取阶段。 |
| 只需要更多 updates | [预算曲线](libero-task4-task7-public-state-jitter-budget-sweep-diagnostic-2026-10-31.md)中 500→2000 持续降低 support MSE；task 4 query 均值也继续降低。 | task 7 baseline 的 query 在 1000/2000 均未优于 500；没有更长预算的闭环对照。 | 不能说 500 已收敛或最优，也不能说加预算会修复控制。继续固定 500 以保持后续机制可比。 |
| action head 单独学习已足够／视觉路径无关 | [冻结层级复核](libero-runtime-native-prefix-task4-task7-module-freeze-seed-replication-diagnostic-2026-10-29.md)的 support 排序 `12/12` 稳定，[query 排序](libero-source-meta-query-generalization-diagnostic-2026-10-30.md) `6/6` 稳定，Full Policy 更好。 | 冻结同时改变可训练参数量及优化自由度；Full Policy 的离线优势尚未形成可靠闭环。 | 表示与后续映射参与拟合，但未分离具体层的因果贡献，也未说明共享表示是否使输出目标相互干扰。 |
| eye-in-hand 敏感性过高是统一根因 | [修复后的七对窗口](libero-task4-task7-repaired-evidence-decomposition-diagnostic-2026-11-01.md)逐相机读出中，eye-in-hand 局部动作敏感性最高。 | [跨任务平移增强](libero-runtime-native-prefix-cross-task-eye-hand-augmentation-diagnostic-2026-10-28.md)降低敏感性，却使四任务离线 MSE 变差，选定 candidate 成功数由 `6/18` 变 `3/18`；设置不同于当前完整 support 训练，不能混合统计。 | 相机参与动作生成不等于相机有错，降低敏感性也不等于控制更好；不重启该增强。 |
| 公开状态漂移是单一根因 | [固定 Policy 替换](libero-task4-task7-full-policy-observation-state-replacement-sensitivity-diagnostic-2026-10-31.md)会改变动作，任务内部 drift 与输出敏感性共同变化。 | pooled 相关与任务内相关方向相反；[闭环轨迹](libero-task4-task7-full-policy-closed-loop-attribution-diagnostic-2026-10-31.md)中 task 4/7 的成功—失败 joint offset 关系不同。offset 相对 reset，不是专家误差。 | 漂移可能参与反馈放大，尚未证明它是起点；不能由 offset 大认定动作错误或失败。 |
| public-state jitter 能修复闭环 | [同专家观察误差](libero-task4-task7-expert-action-error-diagnostic-2026-11-01.md)显示两任务 query 平移 MSE 都在三 seed 下降，task 4 夹爪也改善。 | task 7 夹爪两 seed 变差；[配对闭环](libero-task4-task7-public-state-jitter-500-closed-loop-diagnostic-2026-10-31.md) baseline `7/120`、jitter `6/120`，有 4 个 baseline-only、3 个 jitter-only、3 个共同成功。 | 不将 jitter 升为默认；这否定了当前设置作为稳定修复的依据，没有排除所有状态鲁棒性方法。 |
| 缺少 recurrent memory 是统一原因 | [容量匹配 CNN/GRU](libero-memory-representation-diagnostic-2026-10-25.md)中 GRU 在部分 transport gripper 指标更好。 | task 3/7、seed `20261021`、phase-balanced、500 updates 下，CNN-only 整体 action/translation 更好，release 未获稳定改善。 | 该历史单 seed 不能排除一切历史信息收益，但不支持直接恢复 GRU；当前先不增加记忆复杂度。 |
| open 切换误差只需提高局部权重 | [专家目标诊断](libero-task4-task7-expert-action-error-diagnostic-2026-11-01.md)的 `24/24` 个 task×role×seed×arm 条件都在 open 窗口出现最高夹爪误差；局部 2× 后窗口均值下降。 | [配对训练](libero-task4-task7-local-gripper-loss-diagnostic-2026-11-01.md)两个任务都仅 `2/3` seed 局部 MSE 下降；task 4 平移 `+7.03%`，七维 task 4/7 `+4.19%/+6.63%`，整体夹爪错误也存在 open/close 取舍。 | 当前 2×、±5 步、500-update 候选失败，分支停止；误差热点不是修复有效性的证据。该候选没做闭环，不能说其成功率更差。 |
| 首步分叉完全由后续观察漂移造成 | [四格分解](libero-task4-task7-repaired-evidence-decomposition-diagnostic-2026-11-01.md)中固定 Policy 替换观察确有影响。 | 同一首步观察已经产生跨 Policy 动作差；前 20 步同观察换 Policy 的差大于固定 Policy 换完整观察，task 4 为 `0.0452–0.0454` 对 `0.0191–0.0221`，task 7 为 `0.1004–0.1113` 对 `0.0268–0.0373`。 | 已分开模型映射差与局部输入敏感性，但不是正确动作比较；七对按 outcome 选出，不能代表全部状态或归因于 action head。 |
| task 4 补采差异说明训练不可复现 | 初次直接跳到目标 state 的四个窗口不匹配原轨迹，不能用于原 success transition 的解释。 | [恢复 reset 顺序](libero-task4-ordered-prefix-reproduction-diagnostic-2026-11-01.md)后 `56/56` 窗口精确匹配 reset/action/公开状态，四个目标 archive 有效；原生 sampler 配置项累积已取证。 | 复现问题已按原生命周期解决，旧失败窗口仍隔离；这是评估协议边界，不是低成功率本身的修复。 |

历史全局 6× gripper 臂使用 uniform phase、`12N` 分母，本次局部 2× 用 `7N`；[历史报告](libero-phase-gripper-loss-diagnostic-2026-10-25.md)已按代码纠正文案。二者只能各自说明具体设置的取舍，不把它们合并成权重效应量。局部训练原结果中的索引 dataset access 快照与实际 causal 训练视图之区别，沿用[原报告的说明](libero-task4-task7-local-gripper-loss-diagnostic-2026-11-01.md)，不改写原始元数据。

## 仍需要分离的三类机制

| 候选假设 | 为什么仍可能成立 | 什么观察会削弱它 | 顺序 |
|---|---|---|---|
| **H1：平移与夹爪目标在共享参数上存在优化干扰** | 同输入配对训练出现局部与整体误差取舍；代码共享 encoder/fusion/逐帧投影，且旧/新加权均伴随部分运动误差上升。 | 固定权重、同批输入的分量梯度主要同向，或差异只表现为全局裁剪缩放，而非共享层负向内积。 | **下一项**：先测方向与缩放，不改训练。 |
| H2：表示不能充分区分需要 open/close 的细粒度条件 | 支持集类别齐全却有切换误差；相同初态不同 Policy 已分叉。 | 在当前特征上跨示教的 open/close 条件可分，错误更多集中在最终输出映射。 | 备选。需区分表示混叠、样本邻域稀疏与真实观察不足；当前不能直接断言视觉欠观测。 |
| H3：历史专家分布与当前闭环访问分布不匹配，缺少纠偏覆盖 | replay 语义已分叉；专家状态拟合明显强于候选自由运行，jitter 未带来稳定修复。 | 在受控、可比的观察语义和相同访问状态下差异仍由输出映射主导。 | 保留。需要额外的访问状态/正确动作证据；现有自由运行数据没有专家标签，不可伪造。 |

H1 的优先级来自已有配对设置、可用代码路径与低增量数据需求，不表示它最可能解释所有闭环失败。下一项只检验 H1 在现有 500-update 终点附近的局部版本；未见冲突也不能否定更早训练阶段的干扰。H2 的早期近邻诊断仅检查 task 7 单 seed、两个 candidate 的首帧到 30 个 support 首帧的距离；[原报告](libero-runtime-native-prefix-task7-initial-observation-diagnostic-2026-10-27.md)未校准 OOD 阈值，也没有检验全时序跨示教条件可分性。H3 已有机制敏感性证据，缺少的是可比干预，不能再仅重复一次不对齐的 replay 就声称定位根因。

## H1 的新增只读线索：裁剪次数

从局部训练原 JSON 的 `gradient_norm_by_update` 逐条重算 `norm > 1`，与 `gradient_clipped_update_count` 全部一致，每格分母为 500 updates：

| task | seed | baseline | 局部 2× 候选 |
|---|---:|---:|---:|
| 4 | 20261022 | 30 | 69 |
| 4 | 20261023 | 37 | 91 |
| 4 | 20261024 | 33 | 68 |
| 7 | 20261022 | 15 | 28 |
| 7 | 20261023 | 9 | 26 |
| 7 | 20261024 | 13 | 12 |

五组增加，一组减少，因而“加权后裁剪统一增加”不成立。当前只有合并梯度的范数，没有分量梯度的方向；加权后两臂权重路径也逐步不同，不能用这张表分离因果。全局 norm clipping 按同一正标量缩放合并梯度，本身不改变该梯度的方向；它改变 Adam 接收的梯度，不能把裁剪次数直接换算成实际参数更新强度。以上是原日志的只读摘录，**不是已完成的梯度冲突实验**。

## 下一项具体设计与判断边界（未执行）

问题是：**在现有 500-update baseline 的同一参数与同一专家输入下，降低局部 gripper 误差的梯度是否会一阶推高平移误差？与仅改变裁剪尺度能否分开？**

1. 固定 task 4/7 × 三 seed 的六个 baseline 500 checkpoints，核验文件/权重 hash；不重训、不选择中间点、不加载 1000/2000 点。只读 source `meta_support 0..29` 的完整 `t>=1` causal 输入，分别为 4,442/4,188 步。按 demo 顺序、每条 demo 独立的非重叠 10 步窗口组织；保留 padding mask，排除真实 `t=0`，不重建首帧。共 8,630 个独立观察、25,890 个模型—观察配对，尚不是实际运行量。不读 query 或任何 official evaluation state。
2. 在同一前向图上分解 `L_T/L_R/L_G`，分别只包含平移三维、旋转三维和夹爪一维，但均使用原 `7×valid_steps` 分母，使 `L_base=L_T+L_R+L_G`。另提取原 open 切换 ±5 步并集内的夹爪项 `L_open`，只由 support 专家命令生成，不输入 Policy。零有效步或无局部窗口的指标记 null。
3. 用 `autograd.grad` 读取原始梯度，不调用 optimizer、不写 `.grad` 或参数。对共享 visual encoder、fusion、CNN-only 逐帧 projection 及共享参数整体，报告 `g_T` 与 `g_open` 的内积、余弦、范数，同时记录全局 gripper 与非窗口 gripper 的对照。action head 的平移行与夹爪行互不重叠，作为内积应为零的实现检查；不能因它们为零就宣布网络没有共享干扰。
4. 同一权重上核验 `g_base=g_T+g_R+g_G`，计算 `g_base+g_open` 对应已结束局部 2× 设计的代数梯度，只比较原阈值 1 下的裁剪系数和方向。该计算不更新权重，不复活候选，也不扫描新系数。任何实际 Adam 更新结论都不由原始梯度余弦推出；现有 checkpoint 没有完整 optimizer 历史。
5. 按 task×seed 报告有效批次数、局部步数、负内积比例、余弦均值/中位数及层间差异，保留逐窗口读出，不能只挑负内积案例。共享层冲突若在一个任务三 seed 中方向一致，才保留为该任务的局部优化线索；若仅单 seed 或少数窗口出现，记为条件性现象。没有稳定负向结构则不支持 H1；若主要差在裁剪系数，记录缩放替代解释。技术不一致、非有限梯度、输入/权重 hash 变化立即停止，保留失败记录。
6. 先实现纯离线 runner 和梯度代数/不变性测试，再冻结独立版本配置、批处理大小、CPU 数值设置、输出路径/hash 与实际反向计算预算，之后才执行。新诊断 optimizer updates/reset/action steps 均固定为 0；本轮只记录设计，没有运行它。

固定终点梯度只描述当前权重附近的局部结构，不能重建过去 500 步 Adam 轨迹；固定 `t>=1` 也不能回答首个动作是否正确。即使 H1 得到支持，也须另行设计匹配容量/数据/预算的优化边界对照，不能自动改成多头、梯度修正或取消 clipping。若不支持，优先准备 H2 的跨示教条件可分性诊断，不继续搜索局部损失权重。

## 与主实验的关系及当前状态

这些工作用于判断 Policy 学习配置为什么尚未形成可复现的闭环能力，是 LIBERO SPT/SPI 主实验前的可行性准备。它们没有比较原始 SPT、更新后 SPT 与匹配普通预训练，也没有提供独立环境机制证据。qualification `20..39`、reserved `40..49`、validation、held-out 和 final evaluation 保持关闭；不更新 SPT/Knowledge、不注册 Module。Crafter、Continual World、RoboCasa365 的结果与本项分开。

本轮完成了证据收束和下一假设选择；下一步是实现并冻结上述零更新梯度诊断。保留原 baseline，训练预算继续为每模型 500 updates，已关闭的局部候选不追加训练或闭环。

## 只读来源核验与文档验证

下列既有 JSON 均重新计算 SHA-256，并与其报告记录一致；没有覆盖原文件。历史报告负责各自完整的设备、种子、配置、训练/交互成本和来源细节。本轮只是文件核对与日志算术摘录，没有新实验结果。

| `/tmp/` 下的既有结果文件 | SHA-256 |
|---|---|
| `libero-task4-task7-full-policy-closed-loop-20261030-v1.json` | `03439cc0172f3ba29ee4527d3c64d1da16edd93d2eaf63134344e4c8b15f49ce` |
| `libero-task4-task7-public-state-jitter-500-closed-loop-20261031-v1.json` | `8c9c6167f316072efeba909eb5a2dbf8987369698eddbf5d496bdf8320897ed2` |
| `libero-task4-task7-public-state-jitter-budget-sweep-20261031-v3.json` | `3a2c13feffc9c9d4824138101ae0c48916938d08cf70d236da93f675000a7110` |
| `libero-task4-ordered-prefix-reproduction-20261101-v1.json` | `d147a29896dcfd5cde0539add84fed8f3e880eb289fde1f7f0adfe71ec730d65` |
| `libero-task4-task7-repaired-evidence-decomposition-20261101-v1.json` | `6d8e54e34921d0e3b606ea58c57787f260d830656fe0f028928ecb6a609912c3` |
| `libero-task4-task7-expert-action-error-20261101-v1.json` | `11bed14d3624144553d11e9395b66ee4621e69b5284e740c75117a60e7d7273e` |
| `libero-task4-task7-local-gripper-loss-20261101-v1.json` | `c231ced84aa4d35e573e0ad33f0923bc91e7d7bb6ef0bbd615743505efa4082a` |
| `libero-memory-representation-20261025-v2.json` | `edff7d0271abc633f22a5cea38d77e199450ecb022f21e76f8091ee766089cfc` |
| `libero-conditional-coverage-20261024-v2.json` | `adb9b054e35ba140b4c2c56b6905120662e642d348429d3fe5e98065f53cf6da` |
| `libero-runtime-native-prefix-cross-task-eye-hand-augmentation-20261028-v1.json` | `06e309d7f22bb04dd0b18a5759388989fa56591af015fece3ad4de11eb9f1ecc` |
| `libero-runtime-native-prefix-task7-initial-observation-20261027-v1.json` | `116c313f9d749e6c777de07a8279d1d15065f125e08014f8194c2be3cfc3a19c` |

本轮只修改文档；核验记录保存在 `/tmp/libero-task4-task7-mechanism-evidence-review-audit-20261101-v1.json`，包含本轮 Git/文档与代码 hash、上述 11 个来源 hash、六组裁剪计数、实际核对耗时及零训练/交互范围。相对链接检查发现路线图原有一处示教阶段报告文件名错误，已指向实际存在的原报告，没有修改其研究结果。

文档相对链接、上述 hash 与裁剪计数及 `git diff --check` 已检查，不为文档变更重跑训练或全量 pytest。此前 `433 passed` 是局部损失代码那一轮的 CPU 回归记录，不是本轮重新运行的测试数。

## 后续执行更新：H1 的主指标未形成一致冲突

[固定权重梯度诊断](libero-task4-task7-gradient-interference-diagnostic-2026-11-01.md)已按独立配置完成并复核。float32 v1 因分量求和数值门槛停止，保留失败；有界两精度审计后另冻结 float64 v2，保持原门槛与 source-support/500-update 权重边界。完整 2,670 个主窗口的共享整体局部余弦中位数在六模型均为正，task 4 为 `0.198–0.546`、task 7 为 `0.147–0.288`，不满足预设的三 seed 负向主指标。

仍有 30%–43% 的局部窗口负向，task 7 逐帧投影层三 seed 中位数也均负；不能说所有共享层都无干扰，或排除更早训练/原 batch 8 的冲突。同权重虚拟裁剪窗口在六组增加，只支持尺度可能改变，不能单独解释原训练退化。六份原始梯度、全部 source 输入摘要、分层与权重经独立复核，134 项相关测试通过；新增 optimizer/环境交互均为 0。先前只读文档审计记录的是选择时快照，本文此次进度更新不改写其原始文件。

当前优先级按原分支规则转向 H2：先冻结按示教留一的条件可分性诊断设计，再在固定视觉/fusion/逐帧投影表示中检查 open/close 邻域一致性与动作错误；排除同 demo 相邻帧捷径，不把表示邻域混叠直接解释为环境欠观测。H2 尚未运行，H3 runtime 分布边界继续保留。原 baseline、500 updates、局部候选停止与 qualification/held-out 隔离决定不变。

## 后续执行更新（2026-11-02）

上述 H2 分支及后续固定权重检查现已完成，最新状态见[夜间诊断与晨间报告](libero-overnight-diagnostic-progress-2026-11-02.md)。H2 未达到一致 projection 区分下降条件，错误 tanh 强饱和占比不足以触发预设 BCE 候选；不据此排除所有表示或损失问题。专家观察上的 open 命令偏迟线索保留，公开 gripper 状态固定后局部 MSE 虽六模型下降，每个任务仍各有一 seed 的 sign error 增加。输入覆盖差异在首动作之前已存在，但 task 4 换全时序 source 参照后大多不超线，故不能称全局 OOD 或统一闭环根因。

没有新训练或环境交互，保留所有停止决定及失败 artifact；反馈 JSON 类型错误经完整缓存恢复，未重跑模型。三份独立审计及 532 项 CPU 回归通过。下一项先控制 source 稳定化设置、参照数量和首帧时相，继续固定六个 500-update baseline；新的环境采集与预算仍须独立冻结，不用本轮局部读出自动恢复训练或 SPT/Knowledge/Module 分支。本文上面的“尚未运行”记录的是 11 月 1 日选择时状态，不能替代最新执行报告。
