# task1 B固定权重交叉历史读出协议（2026-11-04）

本项在用户授权重跑后、任何新增数值读出前冻结。承接[query事件分解](libero-task1-b-query-event-margin-timing-diagnostic-2026-11-04.md)，原完整历史候选的query门槛失败和不推进闭环决定保持；这不是对旧失败门槛的修订。

固定task1 B三seed20261022/23/24、现有500-update编码器和500-update监督读出。只读既有query表示、目标、两头权重和预测，不运行编码器、HDF5、环境、初态或训练。query demos30..49已暴露，只作source诊断；validation、held-out、qualification、reserved和final evaluation继续封存。

新增唯一格为历史头下的重复当前表示H(C)，复用当前头C(C)和历史头H(H)。相同support scaler，当前+过去4帧，共同t>=5每seed2518行。预算为3个新增读出批次/7554行；另以float64 NumPy重放已有两格作为算术审计，6批/15108行明确单列，不能称全部0新前向。三格都采用Linear(1280,7)+tanh。

保存完整三格误差和69个模型×open事件、696个完整open帧的分层，不只选6个原新增错误。动作精确分解：H(H)-C(C)=[H(C)-C(C)]+[H(H)-H(C)]；logit也同样分解。前项是当前参照下的头权重差，后项是在同一历史头下的内容差；两项不称独立模块因果贡献。tanh动作/误差不能由logit符号或幅值直接替代。

既有两格的数值重放最大绝对差<=1e-5，夹爪符号必须一致；全部父分层指标<=1e-10复现；动作/logit加减恒等式<=1e-12。零输出保留，不选near-zero阈值、不删不利seed。历史头的重复当前输入仅是代数参照，未必属于真实历史分布，不能据此认定闭环物理根因。

[配置](../configs/libero-task1-b-crossed-history-readout-diagnostic_v1.json)、[runner](../experiments/run_libero_task1_b_crossed_history_readout.py)、[测试](../tests/test_libero_task1_b_crossed_history_readout.py)与本文由绑定hash锁定。默认只做元数据检查，--execute才新增读出。所有直接输入、父绑定和依赖hash先核对，输出用独占创建，不覆盖历史或失败；失配保存失败计数及已完成seed后停止。

运行固定CPU单线程Python3.10.22/NumPy1.22.4。原500 updates、稳定化5步、现有loss不改。不自动训练、闭环、更新SPT/Knowledge或注册Module；若内容与权重分解支持新机制，必须另外冻结新候选与全部预算后，才进行配对训练。候选问题可以是运动6维读历史、夹爪1维读当前；不是预先保证有效的修复。
