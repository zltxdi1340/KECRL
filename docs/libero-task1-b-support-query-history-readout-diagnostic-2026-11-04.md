# task1 B完整support／query历史读出结果（2026-11-04）

已实际完成三seed×当前/历史两臂，各500 updates，共6个读出、3000次更新。所有support训练和三个CPU数值桥接通过后，才读取source meta-query demos30..49；102个query编码批次、3个query重复和3个support桥接全部完成，CPU耗时26.13秒，无真实运行失败或预算重试。

**query主门槛未通过，不推进本配置的历史候选或闭环。** 三个seed的open当步错误为11→9、7→9、10→12（各23事件），只有1/3改善；整体七维和平移MSE及完整open段错误率虽在三seed都改善，不能覆盖切换主条件失败。此前support内部的局部线索没有形成稳定的source-query切换收益，仍不能认定历史不足是闭环根因。

## 冻结设置与数据角色

执行前冻结[科学配置](../configs/libero-task1-b-support-query-history-readout-diagnostic_v1.json)、[协议](libero-task1-b-support-query-history-readout-protocol-2026-11-04.md)和[绑定](../configs/libero-task1-b-support-query-history-readout-binding_v1.json)。仅保留[父三折诊断](libero-learned-phase-history-readout-diagnostic-2026-11-04.md)通过支持集条件的task1 B_fomaml表示来源，不选择最好seed、最好折或中间checkpoint。

原三个编码器均为500-update固定端点，不更新SPT或原Policy。新读出从全0权重/bias重新训练，Linear(1280,7)+tanh、8967参数；当前臂重复当前256维表示5份，历史臂使用当前及过去4帧，同demo、同目标，均除sqrt5。完整support demos0..29的共同t>=5有3939帧，scaler只取这些support当前表示，应用所有lag；std<=1e-6置1。

两臂每seed共用numpy default_rng(seed)产生的500×64采样表，Adam lr0.0003、betas(0.9,0.999)、eps1e-8、weight_decay0、clip1，均匀七维action MSE，不增加夹爪/phase权重。记录support 0/100/250/500曲线，永远评价500端点；query只做最终500检查，不据其选择训练预算、权重或停止点。

本轮source meta-query为demos30..49，原始t>=1共有2598帧，历史共同目标2518帧；其中23个已知open事件全部在共同范围内，早期排除0。source-query与本读出fit分开，但它们之前已用于诊断，不能称未触碰或正式最终评价。原编码器见过全部support，当前方法也已由既有诊断选择，本项不是正式未见迁移。

闭环默认稳定化5步、原Policy loss与500预算保持。qualification、validation、held-out、reserved和final evaluation没有访问；未创建环境、读取官方初态、demo init_state或private simulator state。整文件SHA读取字节与角色限制下的数值payload解码分开记录。

## source-query固定500结果

以下均比较相同query专家帧。事件是专家命令切换，不是物理阶段真值；每seed23个事件，模型×事件合计69，不能当69条独立episode。

| seed | 事件符号错误：当前→历史 | 完整open段错误：当前→历史 | 整体七维MSE：当前→历史 | 整体平移MSE：当前→历史 |
|---|---:|---:|---:|---:|
| 20261022 | 11/23→9/23 | 48/232→38/232 | 0.047220→0.043869 | 0.049650→0.043325 |
| 20261023 | 7/23→9/23 | 37/232→35/232 | 0.052335→0.047137 | 0.051194→0.044256 |
| 20261024 | 10/23→12/23 | 54/232→51/232 | 0.047769→0.046095 | 0.046475→0.042560 |

三seed平均七维MSE为0.049108→0.045700，平移为0.049107→0.043380；完整open段错误合计139/696→124/696。但事件当步合计28/69→30/69，错误率40.58%→43.48%。原预设要求每seed事件严格改善，不能用均值、完整段或整体MSE补救；其他三项flag为true，事件flag为false，joint_query_cue=false。

另一个必须保留的反例是：事件当步的连续夹爪MSE在三seed都降低，却有两个seed符号错误增加。以下只是冻结指标的分量读出，不加入事后margin阈值或把它认定为loss根因。

| seed | 事件夹爪MSE：当前→历史 | 整体夹爪MSE：当前→历史 | 事件错误率变化（百分点） |
|---|---:|---:|---:|
| 20261022 | 1.293730→1.180566 | 0.173986→0.170263 | -8.696 |
| 20261023 | 1.251870→1.125050 | 0.205718→0.190714 | +8.696 |
| 20261024 | 1.363157→1.216341 | 0.187512→0.187636 | +8.696 |

seed24整体夹爪MSE也略增加，整体七维改善并不意味着每个分量都改善。整体错误率、连续动作MSE、离散命令错号和闭环成功是不同指标，本项没有新的成功率或模块因果结论。

## support拟合与曲线

完整support上事件错误三seed分别25/20/22→22/17/19，合计67/102→58/102；完整open段322/1377→285/1377。support平均七维0.048487→0.045247、平移0.052169→0.045700。支持集拟合方向不能替代query主结果。

父诊断是每折20条拟合、10条评价，本轮是全部30条拟合后检查不同的query示教；不能把父三折61/102与本轮query30/69直接解释成性能改善或退化幅度，也不能由训练帧指标比较“最好折”。

![完整support曲线与query切换结果](/tmp/libero-task1-b-support-query-history-readout-summary-20261104-v1.png)

support七维MSE曲线如下，依次0／100／250／500。query没有中间点曲线，只有预设500端点。

| seed | 模式 | support七维MSE曲线 |
|---|---|---|
| 20261022 | 当前 | 0.233849／0.053460／0.050041／0.047153 |
| 20261022 | 历史 | 0.233849／0.052440／0.047735／0.044208 |
| 20261023 | 当前 | 0.233849／0.058261／0.054482／0.051876 |
| 20261023 | 历史 | 0.233849／0.055398／0.050540／0.047000 |
| 20261024 | 当前 | 0.233849／0.052956／0.049470／0.046432 |
| 20261024 | 历史 | 0.233849／0.052375／0.047817／0.044534 |

## CPU桥接和query访问顺序

启动时GPU0/1均有其他作业，分别7019/6212MiB、利用率均70%。本协议在结果产生前就固定CPU和数值门槛，没有运行后换设备或放宽容差。Python3.10.22、Torch2.4.1+cu124、NumPy1.22.4、h5py3.11.0，CPU与interop均单线程，OMP/OPENBLAS/MKL均1，CUDA_VISIBLE_DEVICES空、确定性开启。

六个读出完成500更新并归档后，才读取原support首8窗口：stored观察0..79对应action1..80，批次摘要精确等于旧记录。三个原Policy各一次CPU前向，与既有GPU缓存核对如下；所有绝对差均小于冻结1e-5。

| seed | 表示最大差 | 原动作最大差 | 新当前读出最大差 | 新历史读出最大差 |
|---|---:|---:|---:|---:|
| 20261022 | 1.490e-07 | 1.788e-07 | 2.096e-07 | 2.126e-07 |
| 20261023 | 1.490e-07 | 4.843e-07 | 7.364e-07 | 5.135e-07 |
| 20261024 | 1.192e-07 | 1.788e-07 | 2.104e-07 | 3.749e-07 |

三个桥接均通过后才创建query dataset并读取窗口。CausalT1接口只读obs[t-1]/action[t]的公开数据，不读取t0动作或init_state。三个encoder使用相同34个query批次摘要，query首次批次重复的动作/表示逐位相同，输入/target、encoder file/tensor hash和gradNone均通过。

旧完整2598帧query原Policy连续误差与新CPU读出的最大差为4.43642e-8，低于冻结1e-5；旧/新gripper sign accuracy差另存作描述。数值桥接表明设备差异在预设容差内，不是环境或记忆机制干预，也不声称所有GPU/CPU内部浮点结果逐位相同。

## 实际预算和provenance

| 项目 | 实际数量 |
|---|---:|
| 读出模型／每模型更新 | 6／500 |
| optimizer updates／backward／训练前向 | 各3000 |
| 训练有效样本行 | 192000 |
| support指标前向／样本行 | 24／94536 |
| query最终读出前向／样本行 | 6／15108 |
| query原编码器主前向／有效行／含padding槽 | 102／7794／8070 |
| query原编码器重复／有效行 | 3／240 |
| support桥接原编码器前向／有效行 | 3／240 |
| 原checkpoint加载 | 3 |
| 新support公开观察每模态／动作行 | 80／80 |
| 新query公开观察每模态／动作行 | 2598／2598 |
| 原Policy优化更新、环境、初态/private/action0读取 | 0 |
| CPU入口elapsed／query张量缓存 | 26.12815秒／264612610 bytes |
| 进程峰值RSS | 1085792KiB |

两臂各1500更新、各16次clip，实际Adam超参数及每步loss/范数保留。108次冻结原Policy前向是表示采集/对齐，不是新Policy训练；六个新读出才执行了3000更新。未选择checkpoint、更新活动SPT/Knowledge、注册Module或追加闭环。

六份support NPZ保存四点权重/预测、scaler、原始采样行、500步loss/norm和final Adam缓冲；另有三份桥接、query目标/ID、三份query表示、六份新预测和事件日志，均在`/tmp/libero-task1-b-support-query-history-readout-artifacts-20261104-v1/`。旧结果、权重和配置未改写，checkpoint未复制进仓库。

## 验证和限制

专项50项、相关回归48项，共98项通过；两份新Python文件编译及环境检查通过。回归命令第一次引用不存在的tests/test_libero_dataset.py，退出4且未收集测试；核对后改用实际dataset_loader/audit文件并通过。该命令修正发生在真实研究更新/数值访问0时，记录另存`/tmp/libero-task1-b-support-query-history-readout-preflight-command-correction-20261104-v1.json`，科学代码/配置未因此改变。未运行依赖可选Crafter包的全量pytest。

独立NumPy审计2371项检查通过：复算94536行support曲线和15108行query最终读出，最大预测差7.08852e-7；重建support-only scaler、精确配对采样表、causal历史、全部分层和三个seed主判断，并核对CPU桥接、query访问顺序、原权重file hash、实际更新日志及预算。新增训练、原Policy前向和HDF5数值访问均0。审计未重放encoder或优化轨迹，原重复门槛的通过以执行记录/断言为证。

未发现需重训的数据/索引/数值失配。support与query分离和参数匹配仍不识别独立信息因果贡献：输入秩、标准化后的几何与新读出优化均随条件变化，query又有历史诊断暴露，结果只属于当前source内部diagnostic。

## 证据

[runner](../experiments/run_libero_task1_b_support_query_history_readout.py)、[专项测试](../tests/test_libero_task1_b_support_query_history_readout.py)和冻结绑定可复现本步。预算已用完，后续复核应读归档，不重复原Policy前向或训练。

| 证据 | SHA-256 |
|---|---|
| 科学配置 | `e0e717f1c8eadf6447f38dec0b1e6a8d69ea5b92a71054e8265387b3f4174b72` |
| 执行绑定 | `84364b78b36a1c775a61e9c33af5596de65b0f8f0953252b09277091a613504d` |
| /tmp/libero-task1-b-support-query-history-readout-20261104-v1.json | `49b6b7484d626d204ef7ba27c0377392ed571b9af2b3bb29980299d97c8db241` |
| /tmp/libero-task1-b-support-query-history-readout-independent-audit-20261104-v1.json | `98affa088ad00e52a58a01e888333c34eb975613be2008e1f706d4952bafaccd` |
| 独立审计脚本 | `51328ac53c27ab18f91b2f7c88991f7a31facaa5c6c958f437ff551b318b77e2` |
| 曲线/指标汇总 | `7785eb97694d57d8e18fd4daa02d70413297e60e92593d2a9dae076cb919f396` |
| 图 | `a7f79b40ae966d95c35884dd0197be5fafc6d901810f6643c3746435ebf6ea05` |

## 停止决定和下一步

按冻结规则停止本配置向闭环推进：不只选seed22、不用整体MSE或全段均值覆盖切换失败，不加updates、搜索历史长度或恢复旧局部损失。失败的是当前候选的预设query切换条件，不证明历史普遍无用，也不证明阶段/历史缺失是唯一根因。

下一步建议只读已有23个query事件、三seed两臂的原夹爪输出与事件前后预测，先冻配对定义，定位连续MSE下降但符号错误增加来自哪些帧和margin；不新读HDF5、不新增模型前向，也不把这批已暴露query变成loss/超参数搜索集。完成该机制分解后再选择不同假设，而非自动进入candidate闭环。

稳定化5步、原Policy500 updates及现有loss不变；validation、held-out、qualification、reserved和final evaluation继续封存。本项没有新闭环成功率、正式迁移、SPT接受、Knowledge Evolution或Module qualification，各环境结果不合并。
