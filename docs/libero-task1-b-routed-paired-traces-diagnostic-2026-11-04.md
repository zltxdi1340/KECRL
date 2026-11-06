# task1 B路由闭环60对轨迹只读分析（2026-11-04）

**全部60对已分析并独立复核，本轮不修改训练。** 60/60对都在共同观察下的第5步开始出现动作差异，主要是平移分量；前20步夹爪符号仍逐步相同。后续close时序有早、有晚、有缺失，三个baseline-only也不共享一种close/release或公开drift模式。继续停止该路由候选，保留500 updates、5步稳定化和现有loss，不据此统一提前close、提高夹爪权重或压低状态位移。

这是[既有120条闭环](libero-task1-b-routed-history-closed-loop-diagnostic-2026-11-04.md)的只读分解，原baseline3/60、候选0/60不变，不是新成功率或正式SPT/SPI结果。读取新RGB、HDF5、初态/private、checkpoint和feature NPZ payload均0；新模型/编码器/读出、训练和环境调用均0。

## 范围、时钟与核对

先冻结[配置](../configs/libero-task1-b-routed-paired-traces-readonly_v1.json)、[协议](libero-task1-b-routed-paired-traces-protocol-2026-11-04.md)、[执行代码](../experiments/analyze_libero_task1_b_routed_paired_traces.py)与[绑定](../configs/libero-task1-b-routed-paired-traces-binding_v1.json)，再读取本轮episode JSON。固定task1、seed20261022/23/24、current_all7 baseline与history_motion_current_gripper候选的500端点，环境seed20261022、稳定化5步、官方已用candidate0..19。

120份公开JSON含35609状态行、35489动作；配对同seed/state，共17489个共同动作帧、1200个前20步帧及960个5..20步帧。57个共同失败与3个baseline-only全部保留，candidate-only和共同成功空组仍输出null统计。父结果、角色、端点、成功数/四格、原文件hash和逐步偏移/history均重新核对。

action[t]使用obs[t-1]，trace row[t]为动作后状态；前4动作和row0..4公开状态相同，前5个pre-action观察digest也相同。第5步是候选首次启用成熟历史的位置。之后没有完整RGB，只比较公开状态，不把proprio相同当成完整观察相同。

close仅指g>0，open指g<0，0单列；release为首次close后的第一个负命令，不是物理释放。事件缺失不填0/300，首段到episode末尾明确右删失。不同长度episode按相同窗口或min(两臂长度)比较，不把候选300步与baseline成功终点直接相减当同钟差异。

## 第5步已发生的分歧

全部60对的七维、平移、旋转、夹爪输出值及动作后joint/gripper状态，首次exact差异均在第5步。动作前joint/gripper距离为0，完整公开观察digest一致；动作后joint距离开始非零。因此，首次动作分歧不能由两臂此前已经不同的闭环状态解释。但两臂对共同观察/历史的利用方式及训练后读出权重均不同，尚不能把差异单独归因于历史内容、某一层或输入缺陷。

下表每行20对，均值及中位数不混作目标动作误差；动作是归一化控制数值，joint距离不是末端目标距离。

| seed | 第5步平移gap均值/中位 | 旋转gap均值 | 夹爪输出绝对gap均值 | 第5步动作后joint距离均值 |
|---|---:|---:|---:|---:|
| 20261022 | 0.222005 / 0.216080 | 0.014317 | 0.000000906 | 0.003624 |
| 20261023 | 0.153753 / 0.149047 | 0.007751 | 0.000161663 | 0.003772 |
| 20261024 | 0.283447 / 0.275784 | 0.016111 | 0.000044394 | 0.006374 |

第5步夹爪符号差为0/60；不仅当步，前20步两臂符号逐帧全部相同。48对后来首次符号不同，时刻56..133、中位63；其余12对在共同可见段未观察到符号不同，不填作“第301步”。公开gripper状态第5步就有很小数值差，不能因其exact非零称为主要夹爪失败；运动、机械耦合与后续观察变化都尚未分离。

首个分歧的定位对57个共同失败同样成立：其第5步平移gap为0.099577..0.352394，三条baseline-only的0.204178/0.224254/0.241394均在该范围内。不能把“第5步有差异”当作成败分类规则，或把首次启用历史自动认定为错误实现。

## 三个baseline-only逐条解释

下表首close及close后open均是命令步号。R表示首close段被该episode终点右截断，列出的只是已观测长度。

| seed / state | baseline成功步 | 首close：baseline→候选 | close后首open：baseline→候选 | 首close段观测长：baseline→候选 | 同步第20步joint距离 |
|---|---:|---:|---|---|---:|
| 20261022 / 2 | 143 | 66→68 | 140→未见 | 74→233R | 0.044804 |
| 20261022 / 10 | 137 | 65→231 | 未见→未见 | 73R→70R | 0.049431 |
| 20261024 / 5 | 109 | 61→60 | 未见→114 | 49R→54 | 0.087649 |

- **seed22/state2：轻微close延后，后续运动和open组织不同。** 第5步平移gap0.204178、夹爪输出gap约1e-6而符号一致；候选close仅晚2步，随后未见负命令。第100步两轨迹joint距离0.248827，共同终点143步为0.264914。baseline在140步发负命令、143步成功；这是一例关联，不能证明给候选增加release loss即可修复。
- **seed22/state10：候选在baseline成功前全程未close。** baseline第65步close并在137步成功；候选直到231步才close。共同137步窗口中，baseline有73个close动作、候选0个；不能拿两臂完整首段长度73R/70R说“差不多”。第100步joint距离0.527763，共同137步为0.736708；明显分化已在close时机不同之前的早期运动中开始，但没有目标几何证明其具体失败阶段。
- **seed24/state5：close几乎对齐、joint距离较小，仍丢失成功。** 候选反而早1步close；到共同109步终点，两臂都没有close后的负命令，候选的114步open位于baseline终止之后，不能说它在baseline成功前过早open。第100步joint距离仅0.055442，共同109步为0.050516；同钟close量49/50步接近，结果仍相反。这反驳统一提前close、统一延后open或以更小joint距离作为充分修复条件。

![三个baseline-only的同钟动作、joint距离和夹爪命令](/tmp/libero-task1-b-routed-paired-traces-baseline-only-20261104-v1.png)

图只画既有共同可见段，候选231步close明确标为窗外；没有延长或补齐成功轨迹。三个案例是完整outcome分层，不是从更多成功中挑出的代表，更不是独立验证样本。

## 全部配对的close与drift反证

按完整episode描述，baseline观测到close47/60，候选39/60；未close为13→21。可是候选39条close轨迹仍全部失败，baseline另44条close轨迹也失败，单纯恢复close不足以解释全部结果。

| seed | 首close中位数baseline（观测n/20） | 候选（观测n/20） | 候选更早/更晚/同时（两侧均观测） | 两臂均未观测close |
|---|---|---|---|---:|
| 20261022 | 75（7） | 75（5） | 1 / 3 / 0 | 12 |
| 20261023 | 64.5（20） | 59（17） | 13 / 1 / 3 | 0 |
| 20261024 | 62.5（20） | 60（17） | 10 / 5 / 2 | 0 |

双侧已观测38对中24对候选更早、9对更晚、5对相同；另9对仅baseline出现close、1对仅候选、12对两者未见。不同观测分母的均值/中位不能相减后写成普遍延后。close后首open观测数baseline31/60、候选24/60；baseline两条成功没有后续open，缺失本身不等于任务未完成。

公开运动的方向也不统一。5..20步同钟窗口候选减baseline的joint路径长度均值，seed22为−0.010857、seed23为+0.005517、seed24为+0.013906；平移动作norm均值差分别−0.024657/−0.004395/+0.004182。不能由某个seed的更少移动统一增加动作阻尼或drift loss。

三条baseline-only的5..20步joint路径反而都更短，差为−0.007330/−0.001000/−0.002367；seed22两例在20步的joint距离还低于该seed18个共同失败配对的范围，seed24一例在100步的距离低于该seed全部19个共同失败配对。这些都是对“距离/移动越小越可靠”的反例，不能拿它们拟合新阈值。

所有60对达到100步；200/300步仅剩57个共同失败可比，baseline-only三例为null。17,489个共同帧和600个含缺失同步记录全部保留；不能将晚期57对均值与成功终点补值混合。

## 训练决定与下一项

本轮**不改训练**，保留路由候选停止决定及500/5步/现有loss。现有证据支持一个时序定位：早期运动映射先在共同观察下分开，后续公开状态和部分夹爪命令继续分化；它没有证明唯一因果链，也没有识别必要记忆或视觉/head/覆盖的单一根因。

不统一提前close、延长close、增加release/gripper权重或抑制joint位移；三个baseline-only和全部57个共同失败不支持这种单变量通用修复。也不把固定专家观察上夹爪符号持平，推广为自由运行时夹爪时序不变。

下一步若继续，建议先另冻**共同第5步的固定候选运动头交叉读出**：用已保存的同一前5帧特征，比较候选头在真实历史/重复当前下的输出，并复用已记录baseline和候选动作；分开“当前参照下训练后权重不同”与“固定候选头的历史内容影响”。拟新增3批/60行读出，不启动CNN、环境或训练；本轮尚未读取这些feature/head payload或执行新前向。重复当前只是代数参照，需保留分布外限制，不称独立物理因果贡献。只有分离该机制后再提出新的、单独冻结的训练变量。

## 预算、验证与证据

主分析仅解码同一120份JSON，实际120 episodes/60 pairs、35609公开状态行、35489既有动作，17489同钟帧、1200前20帧、960后启动帧和600同步记录，与冻结预算精确一致。CPU Python3.10.22/NumPy1.22.4；记录的分析耗时2.2237s（不含最后JSON写出）。没有研究运行失败、补样本、训练/环境调用或新checkpoint/feature/RGB/HDF5/初态读取。

57项专项及相关分析/闭环测试通过，两份新Python文件编译、环境检查和diff检查通过。未重复运行全量pytest；可选Crafter依赖不属于本项验证，上一轮排除Crafter/smoke的1149项回归保留作历史记录，不称本轮重新执行。

独立审计不导入项目实现，仅从原120份JSON重建全部episode命令段/数值特征、60对逐帧动作/状态、共同窗口、事件顺序、空组/删失、全部seed与outcome统计和3个单列案例。558511项检查通过，最大绝对差2.22e-16，预设容差abs1e-10/rel1e-12；没有新增模型、训练、环境、NPZ或其他源输入调用。绘图也只使用已保存的共同序列，没有额外样本或阈值。

| 证据 | SHA-256 |
|---|---|
| 科学配置 | `8f32a921bc2c3bc5724df8cf2828600ef370d2291f5356df577fe7a2a4c2eb23` |
| 执行绑定 | `857091dc94ed04d78155805e32e592280d3016813c2aa1031fb46eb1411d5b83` |
| /tmp/libero-task1-b-routed-paired-traces-20261104-v1.json | `e2da94226a51ffe1f69bcf2d0f47c199ba1ef25309ceb9bc1936fe1d7e22d79c` |
| /tmp/libero-task1-b-routed-paired-traces-independent-audit-20261104-v1.json | `4fb063924e1e0bb633dafef764322447b0cb770decbdef42e5c46232203763f9` |
| 独立审计脚本 | `1e02daddc30109a20645f05ff2141f7d15d0a189c2dfc684b7d1504775c01dc5` |
| 图 | `4a508cf65d2efedbe4096b290a88a48df227c04dc24451e86301f39bd7daa6b8` |

详细结果保留120个episode_features、60个pair_results完整同钟序列、三个baseline_only_cases、seed_results与pooled_results全部分母/均值/中位数。输入parent summary及120份JSON hash留在provenance中。artifact不进仓库，不改历史绑定/结果、AGENTS.md、.vscode或无关旧修改；validation、held-out、qualification、reserved和final evaluation继续封存，不接受SPT、更新Knowledge或注册Module。
