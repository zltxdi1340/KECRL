# 输入敏感性复核与局部夹爪状态受控读出协议（2026-11-03）

**已完成旧证据复核并冻结新研究设计，尚未运行新Policy读出。** 继续500 updates、闭环稳定化5步和现有loss/训练观测。新问题只针对task0/1固定18端点的open后局部读出：保持视觉和joint不变，替换两维公开`gripper_states`为因果过去，测量符号决策和平移输出如何改变。不据此重启已停止的增强或局部loss，不注册Module。

## 旧审计复核：支持证据、反证与解释边界

本轮核对11份历史结果/审计JSON、8份研究文档和1份前置字段检查记录，共20个hash；有效文件状态、主要判断字段与旧审计结论一致。没有重跑旧模型、图像变换或全部数值指标。可机读复核：`/tmp/libero-source-internal-input-sensitivity-review-20261103-v1.json`，SHA `dcac8ebdafdcbe672b3a413cbd018670f0e40e5e413badcbd7b2ab6b82c8f78a`。

| 既有证据 | 本轮核对到什么 | 保留的反证／对新设计的约束 |
|---|---|---|
| [修复后的四格输入分解](libero-task4-task7-repaired-evidence-decomposition-diagnostic-2026-11-01.md) | 源前缀核验、task7五对精确匹配；同观察换Policy的动作差较大，eye-in-hand局部响应最强 | 仅7个outcome-stratified案例、没有专家目标；不能归为action head独立贡献或按响应大小改训练 |
| [source0/5输入分解](libero-task4-task7-source-stabilization-input-decomposition-diagnostic-2026-11-02.md) | task4/7在三seed、两参照中RGB响应大于本体，eye-in-hand大于agentview | 所有替换无夹爪符号变化；稳定化／混合观察没有对应新专家目标，不能拿旧action0算正确性 |
| [专家相机—目标误差](libero-task4-task7-expert-camera-target-error-diagnostic-2026-11-02.md) | 完成7200+48批；两个相机、两任务的平移主flag均false | 大响应不是更大目标误差；full破坏也损失纹理/可能分布外，不能证明几何必要性或恢复增强 |
| [夹爪反馈输入](libero-task4-task7-gripper-feedback-diagnostic-2026-11-02.md) | 使用有效恢复v2结果；同demo t1锚定的open MSE三seed均降，但每任务各1个seed符号错误上升；zero输入六模型MSE均变差 | 2706次是**部分Policy读出链**，不是0模型计算；整体t1/zero不作为新训练默认，视觉仍有手指信息 |
| [相机取舍分层](libero-task7-expert-camera-component-tradeoff-diagnostic-2026-11-03.md) | 平移/夹爪取舍与符号/幅值误差分开，close非切换帧存在幅值偏离 | 全局方向依seed；连续夹爪幅值不等于实际握力，不能挑task7/seed24替换主判据 |
| [4像素眼在手增强](libero-runtime-native-prefix-cross-task-eye-hand-augmentation-diagnostic-2026-10-28.md) | 敏感性下降，未增强专家拟合变差，candidate小样本未改善 | 不重启该增强；不同任务/seed/观察臂和当前0/1结果不合并 |
| [局部2×gripper loss](libero-task4-task7-local-gripper-loss-diagnostic-2026-11-01.md) | 两任务未通过固定停止规则，审计确认`stop_candidate_branch` | 不追加权重搜索或闭环；新输入probe不是该训练分支复活 |

[更早的runtime输入替换](libero-policy-input-replacement-sensitivity-audit-2026-10-25.md)还有GRU隐状态及runtime/HDF5语义混杂，只作为边界，不将其四轨迹结论直接推广到当前CNN-only。以上多为task4/7，不能当作当前task0/1新读出已得到的结果。

本轮前置核对曾把旧`task7_exact_previous_pair_parity`误按布尔值检查，实际字段是匹配对数5，校验因此停止且未写新复核/样本结果。已改为严格检查整数5，旧数据、容差和研究算子未变。失败记录单独保留`/tmp/libero-source-internal-input-review-first-check-20261103-v1.json`，不改写旧审计。

## 为什么选公开夹爪状态

[非切换参照核对](libero-source-internal-nontransition-translation-reference-diagnostic-2026-11-03.md)保留open后错号与平移误差的局部关联，但未确定视觉、本体反馈或读出路径的唯一原因。选择`gripper_states`是为了检验一个窄的反馈响应问题，**不是因为它已被证明最重要**。

与旧的整demo t1保持/全零输入不同，这次只改变open事件后1..5步、专家仍为open的帧，使用同一事件或上一动作时刻的真实过去观察。避免零值/远期t1锚点与当前状态的大跨度差异，但视觉中的手指仍存在，混合观察仍可能不符合真实状态。不能由正/负结果推断去掉全部夹爪反馈、物理释放成败或训练收益。

## 固定样本、条件与因果索引

[配置](../configs/libero-source-internal-gripper-state-local-control-diagnostic_v1.json)固定task0/1、seeds20261022/23/24、A_original/B_fomaml/C_pooled_bc全部18个500端点的文件/tensor hash。只读source-support demos0..29，沿用`obs[t-1] -> action[t]`、t>=1；不读action0、示教初态、official state、query或封存任务payload。离线额外稳定化0，闭环默认仍5步。

样本来自原NPZ，不按模型错误/成功/seed选帧：全部open事件后1..5步且expert-open的task0 **149帧**、task1 **157帧**，共306个唯一专家步。原正确／错close／neutral和三seed共同错号、同帧混合对照的标签在新读出前固定，干预后不重新筛分。

每帧只关联最近已知同类open事件，等距较早，规则与原phase分析相同。样本/锚点/原符号/原窗口分区清单：`/tmp/libero-source-internal-gripper-state-control-samples-20261103-v1.json`，SHA `746d461b9c32ca63d43244a71a9c434582a83de906e9ec3fc43b5b79ad5dc323`。

设目标动作在t、所属open事件在e，2<=e<t：

| 条件 | 只替换的公开夹爪状态 | 角色 |
|---|---|---|
| identity | 原输入：HDF5 `obs[t-1].gripper_states` | 精确复现原预测 |
| hold_event_preaction_gripper | 发出事件e的open命令前已观测状态：`obs[e-1]` | 主干预，局部事件锚点 |
| lag1_gripper | 上一步动作前已观测状态：`obs[t-2]` | 固定次级时间滞后对照，不救回主失败 |

两锚点始终从未修改的原始缓存取值，同demo、严格过去，无未来状态、零值、缩放或强度搜索。**事件锚点是action e的pre-action观察，不是action e-1的pre-action观察。** lag+1时两个锚点相同，两条件预测须精确相同。anchor不能来自前一次替换后的数组，避免滞后逐步串联。

RGB两相机、joint7、原专家目标、权重、合法mask/索引均不变。source全部6954帧中，6648帧不干预，含31/34个at-event sham帧；CNN-only没有跨帧记忆，这些非干预行和padding的预测必须精确等于identity。

## 冻结预算与为什么保留原batch

为逐元素复现原baseline预测并检查未处理帧，沿用全部原canonical batch：task0每模型38批、task1每模型53批，batch8/sequence10/stride10。每模型三个条件，不另插真实帧预跑或parity前向。

| 研究预算 | 固定数量 |
|---|---:|
| 固定checkpoint加载 | 18 |
| 主完整Policy前向批次 | **2457**（819×3） |
| 重复完整Policy前向批次 | **54**（18×3） |
| 总完整前向批次 | **2511** |
| 主有效预测行 | 187758 |
| 重复有效预测行 | 3996 |
| 主／重复计算槽（含padding） | 196020／4320 |
| 主读出中的局部评分行 | 8262（306×9×3，主有效行的子集） |
| 两干预条件未处理有效行精确比较 | 119664（6648×9×2） |
| 新共享source观察步／模态、动作行 | 6954／6954 |
| 新训练／backward／环境交互／初态或private state | 0 |

重复固定取每task首个含目标post帧的**原batch index1**，与模型响应无关：task0有效77行、task1有效71行，均有80个计算槽；各9模型×3条件，故有效重复`(77+71)×27=3996`，不能写成4320个评分行。重复和三条件复用同一专家步，不是新独立样本。非干预帧及sham检查不另加前向。

GPU1必须在模型或新观察读取前通过idle门槛（<=512MiB、util<=5%、无compute进程）；不占GPU0、不自动改CPU。Python3.10.22、Torch2.4.1+cu124、CUDA12.4、NumPy1.22.4、h5py3.11.0，单线程、deterministic开启、TF32/benchmark关闭，沿用原确定性pool。原RGB/目标/批次摘要和原62586行完整预测须精确匹配；禁止用容差放宽替代该门槛。

## 指标、反证与停止

主指标：306个原expert-open post帧上的夹爪符号错误率差、平移MSE差，按task/arm/seed分别报告。主事件锚点只有在同task/arm三个seed都符号错误率严格下降且各自平移MSE不增加时，才记描述性的联合响应线索。零变化不是改善，次级lag1不能覆盖主失败；即使通过也不自动训练/部署。

次级报告旋转/夹爪/七维MSE、响应向量、原正确→错误/原错误→正确计数、全部冻结错误分层与逐demo/等demo统计。记录逐帧真实gripper输入L2变化及未变化帧数，不删除无变化帧。输入没变化或没产生动作响应，不能据此证明模型不依赖夹爪状态。

改后的输入不是新的专家轨迹；目标误差只描述相对**原状态原专家动作**的读出对齐，不是改后混合观察上的有效新专家标签。相机仍有手指外观，也不能把本probe当作全部夹爪反馈移除。

角色/hash/shape/dtype/有限性/动作范围/源数据复现/权重不变/gradNone/预算/anchor/重复/sham任一失败立即停，保留失败JSON、事件及partial预测，不换锚点/seed/checkpoint、不追加预算或放宽门槛。未来输出另存`/tmp`，旧checkpoint、结果和协议保持。4像素增强、局部2×loss、全局t1/zero输入作为训练默认的停止决定不变；无SPT接受、Knowledge更新或Module注册。

## 当前完成范围与下一步

本轮只完成旧证据复核、固定样本与预算、[纯算子/验证入口](../experiments/validate_libero_source_internal_gripper_state_control_protocol.py)及合成测试。**实际新Policy前向、HDF5/RGB/初态/checkpoint payload读取、训练和环境调用均0**；只解码旧targets/predictions和JSON。配置中的`research_run_executed=false`是冻结阶段事实，不是实验阴性结果。

科学配置SHA `bf38389d06d2d43c74991365d4763cc0cbbe94ee8517ef1c9cf51ca4d4d40206`。设计/算子绑定与最终验证另存，不能代替执行runner绑定。下一步实现执行入口，绑定runner、设计/算子、runtime与启动环境后，再按2457+54运行；当前没有新受控读出结论，不修改500/5步/现有loss，不打开任何封存角色。

## 冻结验证与证据位置

[新专项测试](../tests/test_libero_source_internal_gripper_state_control_protocol.py)与原事件分解专项合计 **48 passed**，验证锚点的时间偏移、严格过去/同demo约束、不可串联替换、未处理帧不变、独立于模型错误的选帧、错号改善与幅值误差的区别，以及配置被改时拒绝继续。两文件`py_compile`通过。未改共享训练代码，不扩展全量pytest；本次测试不覆盖未来GPU读出执行。

CPU环境检查通过：Python3.10.22、Torch2.4.1+cu124；该检查显式设置`CUDA_VISIBLE_DEVICES=''`，输出device_count0是本次CPU验证设置，不是GPU训练路径失败。

独立审计未导入新算子或旧phase分析器，直接从缓存目标符号重建全部306帧、65个at-event参照、原窗口及重复批次，核对18模型的原符号标签和八项预算。审计：`/tmp/libero-source-internal-gripper-state-control-freeze-independent-audit-20261103-v1.json`，SHA `634bddafb842e3d7fc7eabf414a410b85103aeb8afedbf155238cb83e946697a`，内含独立脚本路径/hash。19个直接输入pin、20个历史嵌套pin通过。

[设计/算子绑定](../configs/libero-source-internal-gripper-state-local-control-design-binding_v1.json)锁定本协议、验证入口和测试，**不是执行绑定**。最终验证记录保存在`/tmp/libero-source-internal-gripper-state-control-freeze-final-verification-20261103-v1.json`，其中记录绑定hash、检查结果、保护文件和实际零研究调用。没有用合成测试或冻结预算冒充新Policy读出结果。
