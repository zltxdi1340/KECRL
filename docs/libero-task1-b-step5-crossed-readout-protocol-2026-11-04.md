# task1 B共同第5步固定候选头交叉读出协议（2026-11-04）

用户要求冻结并比较固定候选头的真实历史/重复当前，以分离历史内容与已学权重差。承接[60对轨迹分析](libero-task1-b-routed-paired-traces-diagnostic-2026-11-04.md)：全部第5步已出现运动差，而前20步夹爪符号相同；本协议在新feature/head数值解码及读出前冻结，不启动训练或闭环。

## 固定对象与三格

仅task1 B、seed20261022/23/24、全部candidate0..19配对，固定编码器500端点与六个完整support读出500端点，不使用OOF或中间端点。现有闭环baseline3/60、候选0/60及候选停止决定保持。正式validation、held-out、qualification、reserved和final evaluation继续封存；没有新增初态解码。

每个场景两臂的前4动作、前5个pre-action观察digest相同。选baseline已存NPZ的特征1..5作为共同输入，构造历史[5,4,3,2,1]与当前[5,5,5,5,5]；support mean/std完全沿用、除sqrt5后float32输入，禁止重新拟合scaler、跨episode取历史或使用未来帧。所有120份episode与feature归档的hash先核对，既有独立审计证明首5个特征两臂逐位一致；再用本轮动作重放作数值桥接。

三格为B(C)（baseline头/当前）、H(C)（候选头/重复当前）、H(H,C)（候选头/运动真实历史/夹爪当前）。**仅H(C)为新增反事实格**；其余两格与已有动作核对。H(H,C)最后一维复用H(C)结果，不能误把夹爪也送入历史。六个head权重/scaler以归档hash锁定，只使用最后500，不读取原CNN的.pt checkpoint。

## 预算与数值定义

| 项目 | 冻结预算 |
|---|---:|
| 全部场景/配对 | 60 |
| 新H(C)读出批次/行 | 3/60 |
| 已有B(C)、H(H,C)算术重放批次/行 | 6/120 |
| 实际Linear矩阵运算/行 | 9/180 |
| 旧episode JSON解码/既有动作行 | 120/35489 |
| feature归档hash核对/数值解码 | 120/60 |
| baseline projection/actions/steps各自解码行 | 各17489 |
| 真正用于输入的前缀特征向量 | 300 |
| 读出头NPZ归档 | 6 |
| 新CNN/optimizer/backward/环境/专家目标行 | 0 |
| HDF5/RGB/初态/private/原Policy checkpoint读取 | 0 |

np.load压缩数组需要解码完整baseline episode，再切前5帧；不能将其说成只解码300行。candidate feature payload不解码，全部120份feature checksum仍核对。head归档weights/bias数组含旧0/100/250/500快照，只使用固定500，不对其他端点读出或选择。原始文件不修改，不复制进仓库。

CPU单线程Python3.10.22/NumPy1.22.4，CUDA主动隐藏。Linear+tanh使用float64 NumPy算术，输入与既有runtime同为float32；已有两格动作最大绝对差<=1e-5、夹爪符号一致才继续解释。所有attempted/completed计数单列，失败即停，不重试或放宽门槛。

主分解使用数值一致的三格重放：

`T = H(H,C)-B(C) = W + D`

`W = H(C)-B(C)`为当前参照下两头已学权重差，`D = H(H,C)-H(C)`为固定候选头的历史内容差。动作与logit逐维恒等式容差1e-12；主夹爪D严格0。另保存以原始记录动作为端点的分解，夹爪微小D残差仅归于数值桥接，不当历史效应。

分平移/旋转/运动6维/夹爪/七维报告T/W/D向量与L2、W·D、cosine。范数不相加；同时保存`dot(W,T)/dot(T,T)`和D对应有符号投影，非零T时两者和1，T严格0时为null。投影可能为负或超过1，不是概率或独立因果贡献，不选择epsilon。保留原始lag特征距离和归一化输入差，按三seed、全部60对、3个baseline-only及57共同失败汇总；两个空成功分层不补样本。

## 执行、解释与停止

[配置](../configs/libero-task1-b-step5-crossed-readout-diagnostic_v1.json)、[入口](../experiments/run_libero_task1_b_step5_crossed_readout.py)、[测试](../tests/test_libero_task1_b_step5_crossed_readout.py)和本文由绑定hash锁定。metadata preflight不解码NPZ；--execute才读出，输出/tmp同名前缀20261104-v1及三个seed派生归档，独占创建，保留失败。旧绑定/数据、AGENTS.md和.vscode不改。

重复当前未必是候选训练分布中的自然历史。精确代数分解只回答指定参照下的计算差异，不证明物理必要记忆、正确动作、闭环根因、SPT更新、Knowledge机制或Module资格；没有专家动作可评分。无论结果如何，不自动改loss/500更新/稳定化5步，不恢复候选闭环。先独立审计全部60格、向量和汇总，再决定下一机制问题；新增训练变量须另行冻结。
