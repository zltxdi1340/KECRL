# task1 B同一候选头双输入专家目标误差协议（2026-11-04）

本项承接[同权重双输入闭环](libero-task1-b-fixed-head-input-closed-loop-results-2026-11-04.md)：真实历史0/60、重复当前4/60，仅seed24增加成功；所有60对动作改变，但动作不同不代表更接近专家目标。用户要求冻结同一候选头、合法专家观察上的目标误差对照，本协议于任何本轮缓存数值解码/研究读出前冻结。当前仅冻结与检查入口，尚未执行；始终是重复source诊断。

## 固定权重、数据角色和时钟

固定task1 B_fomaml三seed20261022/23/24、原编码器500端点对应的已有256维特征、完整source-support训练的候选history_motion_current_gripper头500端点及其support scaler。两输入使用同一个候选头；不加载baseline头、不重训、不重拟合scaler、不选择0/100/250端点，不使用三折模型。候选NPZ内的weights数组会整体解码四个已归档快照，只固定取预定500端点，不运行其他快照。

主评价仅用已经暴露的source meta-query demonstrations30..49，共2598个causal pre-action帧（t>=1），统一取有完整历史的2518帧（t>=5）。这些是示教编号，与official initial-state编号属于不同数据源；本项不打开任何官方初态容器。source-support demos0..29仅是已有权重/scaler的训练来源，本轮不读取或拟合support payload，不调用HDF5、RGB、编码器或环境。

缓存身份来自此前完整support/query运行及其独立审计；必须绑定三个features归档、一个targets+demo/step归档、三个候选头及三个已记录真实历史预测。target[row]与同一row的causal专家观察对应，历史严格为同demo[t,t-1,t-2,t-3,t-4]，不得跨demo、使用未来、padding或action0。全部20个demo、三seed保留，不按闭环成败、事件或误差选子集。

| 输入臂 | 运动6维 | 夹爪1维 | 权重/标准化 |
|---|---|---|---|
| candidate_real_history | 专家当前与过去4帧 | 专家当前重复5次 | 同一候选500头、同一support scaler |
| candidate_repeat_current | 专家当前重复5次 | 专家当前重复5次 | 同上 |

每块256维按固定scaler标准化，五块串接除sqrt(5)，转float32输入；使用相同NumPy float64矩阵算术及tanh比较。真实历史的夹爪输出直接复用同头当前读出，所以夹爪误差差应严格为0。这是合同负对照，不是夹爪学习已改善的证据。

本项从t>=5比较成熟输入，不模拟闭环前4步共享启动。闭环稳定化仍固定5个零动作；专家历史长度5与稳定化步数5是不同概念，不给专家帧额外执行零动作或重标target。因此结果不能直接代替自由运行控制检验。

## 预算、算术复现和停止

| 项目 | 冻结值 |
|---|---:|
| task / seed / 固定候选头 | 1 / 3 / 3 |
| target缓存总行 / 每seed共同帧 | 2598 / 2518 |
| seed×专家帧配对 | 7554 |
| 新重复当前读出 | 3批 / 7554行 |
| 历史运动算术重放 | 3批 / 7554行 |
| 总Linear矩阵算术 | 6批 / 15108行 |
| 目标 / feature / head / 已有预测归档解码 | 1 / 3 / 3 / 3 |
| feature数值解码行（含未作为主评价的t1..4） | 7794 |
| 新CNN / optimizer / backward / 环境 | 0 / 0 / 0 / 0 |
| 新HDF5 / RGB / PT参数 / 初态-private读取 | 0 / 0 / 0 / 0 |

固定seed22→23→24，每seed先一次候选current读出，再一次候选history读出；当前夹爪复用，不隐藏第三批Linear。单个float64算术批次对应全部2518帧，与环境步数或训练更新不是同一计量单位。记录attempted/completed及行数；失败也消耗已尝试预算，不重试补样本。

实际执行使用Python3.10.22、NumPy1.22.4，CPU单线程、CUDA_VISIBLE_DEVICES空，OMP_NUM_THREADS/OPENBLAS_NUM_THREADS/MKL_NUM_THREADS均1。不使用GPU或Torch读出。metadata默认预检只核对JSON与文件字节hash，不数值解码任何NPZ，不打开初态或HDF5。

真实历史重放必须匹配此前候选query预测：原1e-5动作绝对容差、夹爪符号严格相同，且row身份完全一致。两臂夹爪连续值必须逐位相同；目标误差代数最大余差<=1e-12。hash/角色/目标时钟/coverage/端点/scaler/数值/预算任一失配保存failed，主结论无效，保留已完成和部分artifact，不放宽容差、换seed或补跑。

## 指标、分层与解释规则

主指标是每seed全部2518帧的平移目标MSE差，方向固定`重复当前 - 真实历史`，负值表示重复当前更准确。全部三seed严格<0才记为平移一致改善的描述性flag；三seed均值改善而个别seed相反须记混合结果，不能选seed或平均后覆盖反证。

同时报告旋转、运动6维、夹爪与七维MSE，逐帧差的均值/中位及改善/恶化/相同计数、动作delta L2，以及各demo指标和demo等权汇总。联合描述性flag还要求三个seed旋转及七维不增加、夹爪差严格0；它不构成训练接受、SPT更新或Module资格。主指标按帧等权，demo等权作为预定次级报告；不事后切换主权重。

对每个分量保存同一精度下e=真实历史输出-target、delta=重复当前输出-真实历史输出，并检查`DeltaMSE = 2*mean(e*delta) + mean(delta^2)`。前项为指定参照下的误差方向项，后项为响应能量；二者不是独立因果贡献，不按动作响应大小判断准确性。输入/输出差与专家目标误差分别报告。

复用既有专家目标命令分层，不从图像臆造approach/transport/release：all2518、known_open_event23、known_open_segment232、initial_open974、expert_close1312（每seed相同）。事件与段/前缀可重叠，不相加当独立样本；不按本轮误差重新定义窗宽或phase。逐demo和分层均为描述性，没有新增显著性检验、阈值搜索或闭环成功预测主张。

## artifact与边界

[配置](../configs/libero-task1-b-fixed-head-expert-target-error-diagnostic_v1.json)、[执行入口](../experiments/run_libero_task1_b_fixed_head_expert_target_error.py)、[测试](../tests/test_libero_task1_b_fixed_head_expert_target_error.py)和本文独立绑定；历史实验及其绑定文件不变。输出留在/tmp独立同名前缀20261104-v1，排他创建。一个共享NPZ保存合法共同帧/target/history索引与masks，三个seed NPZ保存两输入预测/logit、原真实历史记录及逐元素MSE方向/能量/差；不复制原权重入仓库。

主结论只在三seed、全部7554配对与精确预算都通过后给出。重复当前可能改变候选训练时的表示分布，即使离线误差改善，也不证明闭环收益、必要记忆、环境因果机制、正式SPT/SPI或Knowledge Evolution。保持500端点、稳定化5步、现有loss及旧候选停止决定；validation、held-out、qualification、reserved、final evaluation封存，不自动训练或注册Module。冻结验证和后续执行状态记录于[当前报告](libero-task1-b-fixed-head-expert-target-error-diagnostic-2026-11-04.md)。
