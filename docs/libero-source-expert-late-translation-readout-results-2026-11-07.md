# 固定500端点source专家末段平移动作读出结果（2026-11-07）

**已按冻结预算完成378个主前向＋30个重复批次，30个端点、111888行主预测齐全，独立审计通过。末段平移误差集中依赖任务：task0/1/3两臂各3/3 seed高于补集；task4两臂均0/3，task7 A为0/3、B为1/3。当前不支持给全部任务统一增加末段loss权重。** 保持500 updates、稳定化5步及现有loss，没有启动新训练或环境交互。

本次是source meta_support训练拟合diagnostic，不是新query泛化、完整KECRL验证或Module qualification。G未通过的准入与停止决定保持，H未开展，validation、held-out、qualification、reserved、final payload访问0；不更新SPT/Knowledge或注册Module。

## 冻结与实际执行

[协议](libero-source-expert-late-translation-readout-protocol-2026-11-07.md)与[科学配置](../configs/libero-source-expert-late-translation-readout_v1.json)固定task0/1/3/4/7、seed20261106/07/08、A_original/B_fomaml现有30个500端点。A/B沿用G原始/候选初始化适应臂，文件及weight hash逐一固定。CNN-only、float32、deterministic、TF32关闭、无增强，原四输入和均匀七维action MSE保持；没有新的backward、优化器或生成器更新。

只读source-support demos0..29，`obs[t−1] → action[t]`、t≥1、sequence10/stride10/batch32。每demo最后20个合法动作与其补集对照，全部合法帧都保留。五任务每模型末段均600帧，补集分别为2295/3459/2464/3842/3588帧。末20是示教时间分层，不能自动命名为物理放置、接触或释放；action误差不能换算为原生XY厘米门槛。

[执行绑定](../configs/libero-source-expert-late-translation-execution-binding_v1.json)在真实payload之前固定357项代码/输入引用。先前GPU1有其他compute上下文，准备阶段没有启动数值读取；绑定和实际运行前设备已自然释放，首次资源检查即通过，未抢占或放宽门控。实际设备为物理GPU1，UUID GPU-d90bd7b2-16f6-c0fe-fbc1-7ed4060e95ff，逻辑cuda:0、RTX4090；Python3.10.22、Torch2.4.1+cu124、NumPy1.22.4、CUDA12.4，环境与原G一致。

进程总墙钟31.930秒，活动计时27.889秒，独立审计0.907秒；北京时间2026-11-07 16:14:49开始payload、16:15:17写出主结果。时间短因为是固定权重的离线前向，没有反向传播或训练更新，不能当作重新训练500 updates的耗时。

| 资源 | 冻结值 | 实际attempted/completed |
|---|---:|---:|
| checkpoint读取 / Policy对象 | 各30 | 各30/30 |
| 唯一source HDF5 / 显式File打开 | 5 / 10 | 5 / 10/10 |
| source窗口 | 1933 | 1933/1933 |
| 每模态观察行 / action行 | 各18648 | 各18648/18648 |
| 主前向 / 首批重复前向 | 378 / 30 | 378/378、30/30 |
| 主有效预测行 / 主计算槽 | 111888 / 115980 | 各精确匹配 |
| 重复有效预测行 / 计算槽 | 9210 / 9600 | 各精确匹配 |
| 原首批完整槽重复检查 | 30 | 30/30 |
| G500全support parity | 30 | 30/30 |
| 跨模型输入批次比较 | 315 | 315/315 |
| 输入前后摘要不变 | 378 | 378/378 |
| 权重/无梯度后验检查 | 30 | 30/30 |
| 新训练/backward/环境/初态/private/sealed/旧预测读取 | 全部0 | 全部0 |

最大task输入缓存450530020 bytes，低于2GiB。所有30份权重在首个HDF5数值读取前核验；五份源文件各做全文件字节SHA，再按role解码，字节校验不等于解码其他角色。HDF5 guard精确匹配25组task×字段、每个causal slice；没有action0、demo30+、states或attrs/init_state读取，没有拒绝尝试或技术补跑。

G训练元数据扫描320个模型身份，只解码固定30个source记录及Policy配置；其余290个记录和validation曲线未数值解码。四项全support MSE最大绝对复现差为1.249e−15，低于冻结abs1e−10/rel0；有效帧数和符号准确率精确匹配，30份完整首批重复含padding逐值相等。独立审计仅加载65份本次新数组文件，不读取模型/HDF5或运行Policy。

## 末段与补集：全部任务两臂

以下是三seed算术均值，MSE按3维平移动作分量归一化。方向计数不是显著性、成功率或方法接受门槛，不能把重复模型帧当新增独立样本。

| task | arm | 全support平移MSE | 末20帧 | 补集 | 末段高于补集的seed |
|---|---|---:|---:|---:|---:|
| 0 | A | 0.065411 | 0.113748 | 0.052773 | 3/3 |
| 0 | B | 0.064496 | 0.111416 | 0.052230 | 3/3 |
| 1 | A | 0.056020 | 0.072307 | 0.053195 | 3/3 |
| 1 | B | 0.051528 | 0.068253 | 0.048626 | 3/3 |
| 3 | A | 0.045853 | 0.055872 | 0.043413 | 3/3 |
| 3 | B | 0.048146 | 0.061175 | 0.044974 | 3/3 |
| 4 | A | 0.038754 | 0.029523 | 0.040195 | 0/3 |
| 4 | B | 0.038795 | 0.032970 | 0.039704 | 0/3 |
| 7 | A | 0.035977 | 0.030947 | 0.036819 | 0/3 |
| 7 | B | 0.036423 | 0.031797 | 0.037197 | 1/3 |

共19/30个模型的末段误差数值更高，不能推广为全部任务都应重加权。task0的末段/补集比约1.80–2.30，task1约1.23–1.56；task3包含接近相同的反例：B seed20261107只高约0.52%，且只有13/30条示教的末段MSE高于各自补集。均值方向不意味着每条示教同向。

task4 A/B各三个seed的末段均更低；task7 A三个seed更低，B只在seed20261107略高（约0.96%）。低于补集不等于动作已经足够准确；也不能因此直接判定闭环分布偏移为唯一原因。

## 全部15组A/B末段配对

| task | seed | A末段平移MSE | B末段平移MSE | B−A |
|---|---|---:|---:|---:|
| 0 | 20261106 | 0.109954 | 0.121450 | +0.011496 |
| 0 | 20261107 | 0.128717 | 0.118494 | -0.010224 |
| 0 | 20261108 | 0.102574 | 0.094306 | -0.008268 |
| 1 | 20261106 | 0.071693 | 0.068873 | -0.002820 |
| 1 | 20261107 | 0.070576 | 0.062349 | -0.008228 |
| 1 | 20261108 | 0.074650 | 0.073537 | -0.001114 |
| 3 | 20261106 | 0.065970 | 0.083704 | +0.017735 |
| 3 | 20261107 | 0.044010 | 0.040541 | -0.003468 |
| 3 | 20261108 | 0.057638 | 0.059278 | +0.001640 |
| 4 | 20261106 | 0.026103 | 0.034643 | +0.008540 |
| 4 | 20261107 | 0.037045 | 0.036102 | -0.000943 |
| 4 | 20261108 | 0.025421 | 0.028164 | +0.002743 |
| 7 | 20261106 | 0.028995 | 0.028373 | -0.000621 |
| 7 | 20261107 | 0.033190 | 0.037853 | +0.004663 |
| 7 | 20261108 | 0.030656 | 0.029165 | -0.001491 |

task1 B在三个seed都更低；其余任务保留正反方向。task3 B末段均值0.061175高于A的0.055872，2/3 seed更差，但[相同端点的既有来源短前缀](libero-source-candidate-geometry-capture-results-2026-11-07.md)为B 6条原生目标达成、A 0条（各12条、最多150步）。两者是不同观察条件和指标，不能据此计算“离线MSE预测成功”的关系、合并为正式成功率或认定处理因果；它提供了不能按离线排序选方法的反例。

三维平移动作分量误差和bias均已保存。task0/1/3两臂末段的第3个动作分量MSE均值最大，但当前没有对应的目标幅值参照，不能直接认定某个世界轴控制或action head是根因，也不能据此立即增加该分量loss。

## 验证与证据

新增24项软件测试、165项重点测试通过；CPU回归1797项通过，排除可选Crafter相关收集项，不声称全环境pytest通过。四份执行相关Python文件编译和CPU环境检查通过。初次合成Policy测试遗漏两个配置字段，已在真实payload前修正，失败日志保留；实际研究读出没有失败、补跑或额外warm-up。后处理绘图只读审计CSV，没有新增Policy、checkpoint、HDF5或环境调用。

独立审计完成201144项数值/身份/重复/计数核对，输出90行model×stratum、45行配对、30行task×arm×stratum和900行demo分层CSV。完整[对比图PDF](/tmp/libero-source-expert-late-translation-figures-20261107-v1/translation-tail-vs-complement.pdf)中柱为三seed均值、点及线为同seed末段/补集配对，无置信区间或显著性标注。

| 产物 | SHA-256 |
|---|---|
| 科学配置：`configs/libero-source-expert-late-translation-readout_v1.json` | `0349b99cae5b9e79019de54f7e8422b5b850ae52309544f870aa5c51443453e6` |
| 执行绑定：`configs/libero-source-expert-late-translation-execution-binding_v1.json` | `532874caf8ad780f30f796c0dd11df047940b65752f09ca2075e9de84cda28bc` |
| 主结果：`/tmp/libero-source-expert-late-translation-readout-20261107-v1.json` | `b44f498a82d50e6a19a289dfcdcf3bdf308f514ab92f770b3be458890c879b67` |
| 独立审计：`/tmp/libero-source-expert-late-translation-artifacts-20261107-v1/independent-audit.json` | `3424571219e6602da7cd9083e97dde3447025b3a80fb3b6005c57705d0aa39b3` |
| 完成收据：`/tmp/libero-source-expert-late-translation-artifacts-20261107-v1/completion.json` | `05efc8060a4302444db69eff3d31dc4ab64b8b8809279ab829eb275f608b6b1a` |
| 只读汇总：`/tmp/libero-source-expert-late-translation-summary-20261107-v1.json` | `07f0070abb15a8fbe04d809ae3957fe451721c4f9e44f453b6adea825220c40a` |

主结果保留原始pending-audit状态，通过另记independent-audit/completion，不回写字节。原始30个prediction NPY、5个label NPZ、30个全槽repeat NPZ、事件/输入摘要和CSV均在`/tmp/libero-source-expert-late-translation-artifacts-20261107-v1/`；[自动可读结果](/tmp/libero-source-expert-late-translation-artifacts-20261107-v1/readable-results.md)可直接检查全部seed。图及格式化代码的独立provenance在同前缀figures目录的figures.json，绘图不修改已绑定执行代码。

| 审计表 | 行数 | SHA-256 |
|---|---:|---|
| model-strata.csv | 90 | `909a521674e003db5ddb3847fa5450129793252ea75fe240e16e56ca00340fa1` |
| paired-strata.csv | 45 | `b2340f146d2c73f2ee48685cbc8e20ad50a4fd0d00c315d4296eea8802084a7e` |
| task-arm-strata.csv | 30 | `9f9fd9eb907f72842ba363195647b9acc92d7cd3cddb0df759d1009492a54904` |
| demo-tail-translation.csv | 900 | `face512a637f8d1705b68cfbe67e4c070e9287d86fb46d026e2e96c58159e7b6` |

进程/启动记录为`/tmp/libero-source-expert-late-translation-{launch,process}-20261107-v1.json`，控制台日志为同前缀run日志；软件验证元数据为`/tmp/libero-source-expert-late-translation-software-tests-20261107-v2.json`。最终文件完整性和全部预算核对另存`/tmp/libero-source-expert-late-translation-final-verification-20261107-v1.json`，保留原1327个保护文件、357项执行引用和历史结果，不复制checkpoint进仓库。

## 训练决定与下一步

当前不实施统一末段加权，不改变500 updates、5步稳定化或现有loss。source专家输入下确有动作误差，但它在task4/7没有普遍集中于末20帧；相对更低也不能证明精度足够。任务差异、目标动作尺度和闭环观察条件尚未分离，不把本结果升级为SPT更新、Knowledge机制或Module资格。

下一步只读这批已保存的label与prediction/指标缓存，在原末20/补集分层下加入专家平移动作幅值（零平移动作的离线误差参照），并按三个动作分量及有符号偏差分解，检查差异是否主要随目标幅值变化。保留全部模型与seed，0新CNN/HDF5/训练/环境；完成这一分解后再决定是否有理由另冻单变量训练候选。若需物理接触的精确解释，仍要另冻原生判定操作数的有限归档，不能将影子接触当成精确原生标签。
