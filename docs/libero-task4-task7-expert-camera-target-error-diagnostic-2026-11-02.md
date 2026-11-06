# LIBERO专家输入相机干预与目标动作误差诊断（2026-11-02）

本项执行[已冻结科学设计](libero-task4-task7-expert-camera-target-error-protocol-2026-11-02.md)，不修改原配置或算子。固定task4/7、三个seed、六个baseline500，仅读取source-support demos0..29，保持原causal `obs[t-1] → action[t]`、t>=1。闭环默认稳定化5步不变，离线没有reset或额外稳定化，没有新训练。

## 启动前执行绑定

- [执行入口](../experiments/run_libero_task4_task7_expert_camera_target_error_diagnostic.py)，SHA-256 `5a3ed5c2bc611120521a71f9ae821e46b09e4a607c8d1aae6163e6e784793067`。
- [执行绑定](../configs/libero-task4-task7-expert-camera-target-error-execution-binding_v1.json)，SHA-256 `16ebf9946b39abb83f24a87d397b9f1db90acf00459317a27e7e22b652d7aa42`。
- 原科学设计SHA-256 `67654a94e8e41497fa24b0929849f2d9535d9b5ae518bd56c39a863182acfdb4`；算子/校验代码SHA-256 `681a3390bc894e753e945fa0bdab6b79b5dd4ee2a60cc2fccb9e18fb10b7d834`，均保持不变。原配置的pending字段是设计冻结时的历史状态，实际执行由本次独立binding绑定，不改写原记录。
- 绑定UTC时间`2026-11-02T09:44:19.278942+00:00`；CPU、Torch单线程、CUDA隐藏，OMP/OpenBLAS/MKL=1。Python3.10.22、Torch2.4.1+cu124、NumPy1.22.4、h5py3.11.0，MKLDNN开启，分析seed20261022。
- 元数据预检`/tmp/libero-expert-camera-target-error-execution-preflight-20261102-v1.json`，SHA-256 `9bdb979642ce89c6f20738b5db90649e1e0d847c79b04dcc384c3897673b07ba`。预检没有新Policy前向或HDF5数值读取。

## 固定预算与执行门槛

八个条件、干预幅度、像素匹配、误差恒等式、三seed描述flag全部沿用原设计，不搜索强度或调规则。主前向7200批／207120行；每task×seed首批八条件全部重复，共48批／1536行。支持集8630个专家输入、8690行原动作（含60个仅作命令mask的t0）、六个checkpoint加载；optimizer、backward、环境、初态/private数值读取均为0。

角色授权在HDF5打开之前；仅访问source-support组的actions和四种公开观察，不读query组或init/private属性。原目标、mask、索引和300个共享输入摘要精确匹配旧support记录，900次原预测、900次uint8/float恒等检查及48次重复均须精确一致。像素匹配和有符号误差/分量恒等式沿用1e-7/1e-12，不放宽原数据与预测精确门槛。

每次前向记录attempt、完成、行数和预测摘要。算子参数在前向之前写入独立JSONL；失败时保存当前预测、有效行mask和failed结果，不覆盖、不追加预算。成功时逐行保存五个误差项、原预测及全部分层/等demo汇总；这些符号项是误差代数，不是因果贡献。

启动前新增11项执行测试、连同冻结算子及原专家误差测试共49项通过。真实Policy类只在合成图像上验证uint8/float精确一致，没有额外使用真实专家帧做Policy预跑。入口py_compile和CPU隔离环境检查通过；测试中的缺失binding文件夹具已在冻结执行绑定前修复。本项研究结果与实际成本在执行后追加。

输出`/tmp/libero-task4-task7-expert-camera-target-error-20261102-v1.json`，预测/误差项/事件/算子参数在同名前缀artifacts目录，均拒绝覆盖。qualification、reserved、validation、held-out、final evaluation继续封存；不更新SPT/Knowledge、不注册Module。

## 执行结果与主判断

**已完成并独立复核，实际7200主前向＋48重复与绑定精确一致。** 两个任务、两个相机均未满足预设的平移主指标“三seed一致外观误差放大”或“空间细节依赖”flag。弱干预的方向依赖seed；较大输入响应不直接等于误差放大。完整空间信息损失probe使六模型的误差都上升，但其幅度未与弱控制匹配，也会损失纹理并可能分布外，不能由此单独证明几何必要性。

下表为眼在手相机的全support逐步加权平移MSE。Delta均为相对原观察的**绝对MSE差**，正为变差、负为改善；spatial−photo是两个matched条件的配对差，不是相对百分比。

| task | seed | 原平移MSE | photo matched Delta | spatial matched Delta | spatial−photo | spatial full Delta |
|---|---:|---:|---:|---:|---:|---:|
| 4 | 20261022 | 0.038715443 | +0.000560096 | +0.000808446 | +0.000248351 | +0.036357479 |
| 4 | 20261023 | 0.033709614 | -0.000087910 | -0.000437320 | -0.000349410 | +0.021465699 |
| 4 | 20261024 | 0.037131512 | -0.000012376 | -0.000747886 | -0.000735510 | +0.023692601 |
| 7 | 20261022 | 0.037785056 | -0.000118916 | +0.000704826 | +0.000823742 | +0.071588161 |
| 7 | 20261023 | 0.034418914 | +0.000075063 | +0.000667483 | +0.000592420 | +0.065411194 |
| 7 | 20261024 | 0.037530424 | -0.000595342 | -0.001479342 | -0.000884000 | +0.052642652 |

眼在手photo的平移Delta在每个任务都仅1/3 seed为正；matched spatial−photo在task4为1/3正、task7为2/3正，均未达到三seed规则。原定等demo权重的次级均值对上述六组photo/spatial Delta的正负方向一致，不能通过换汇总口径获得一致结论。

外部相机同样保留完整结果，不能只看眼在手：

| task | seed | agentview photo Delta | agentview spatial Delta | agentview spatial−photo | agentview full Delta |
|---|---:|---:|---:|---:|---:|
| 4 | 20261022 | +0.000790261 | +0.001079634 | +0.000289373 | +0.009887082 |
| 4 | 20261023 | -0.000036039 | +0.000554902 | +0.000590941 | +0.020650376 |
| 4 | 20261024 | -0.000054122 | -0.000146050 | -0.000091928 | +0.007028053 |
| 7 | 20261022 | +0.000212646 | +0.000576928 | +0.000364283 | +0.006097971 |
| 7 | 20261023 | -0.000040212 | -0.000113457 | -0.000073245 | +0.007587008 |
| 7 | 20261024 | -0.000161972 | +0.000292629 | +0.000454601 | +0.003973984 |

## 次级线索与误差方向分解

下表严格按原判据记录，两类flag可以同时成立。平移是主指标，七维不能覆盖主判断。

| task / 相机 | 平移：外观放大 | 平移：空间细节 | 七维：外观放大 | 七维：空间细节 |
|---|---|---|---|---|
| 4 / eye-in-hand | 否 | 否 | 是 | 否 |
| 4 / agentview | 否 | 否 | 否 | 是 |
| 7 / eye-in-hand | 否 | 否 | 否 | 是 |
| 7 / agentview | 否 | 否 | 否 | 否 |

task7眼在手的七维空间flag为是，表示spatial−photo的七维Delta在三seed均正且full Delta均正，**不表示matched spatial本身三seed均比原观察差**：其七维Delta分别为`-0.000390723/+0.001847013/+0.000128243`。这类局部反证必须保留。

一个具体的分量取舍是task7/seed24的眼在手matched spatial：平移Delta `-0.001479342`，夹爪Delta `+0.005312492`，旋转Delta `+0.000007744`，七维总Delta `+0.000128243`。不能用平移改善宣布整体改善，也不能用七维变差改写平移主判据。

同组平移响应能量为`+0.000722383`，原误差与响应的交叉项为`-0.002201725`，相加得到`-0.001479342`。非零动作变化在这里朝专家目标方向修正了原误差。task4/seed22的同类交叉项则为`+0.000586155`、能量`+0.000222291`，合为`+0.000808446`，方向相反。这正是不能只用敏感性大小判断错误的原因；这些是固定映射的误差代数，不是模块或环境因果贡献。

所有task×seed×条件的七类命令分层、30条逐demo统计、等demo汇总、逐行误差项和相机配对差均已保存。命令分层不代表真实物理阶段；当前没有新闭环结果，不能从支持域MSE推断qualification或执行成功率。

## 实际干预量与反证

| task | matched有效帧 | 共同像素MAE：均值/中位数 | 共同MAE范围 | 眼在手full MAE均值 | 外部相机full MAE均值 |
|---|---:|---:|---:|---:|---:|
| 4 | 4442/4442 | 0.00402349 / 0.00409913 | 0.00287319–0.00504541 | 0.04298556 | 0.04740712 |
| 7 | 4188/4188 | 0.00381217 / 0.00380334 | 0.00242167–0.00536800 | 0.03995885 | 0.04387445 |

四种matched干预在同一帧使用同一个像素MAE预算，不按目标或预测选强度；本轮无零预算帧。最大实际匹配误差为`1.52633e-8`，小于冻结1e-7。full的破坏量明显大于matched，不能将两者的误差幅值直接当作纯几何效应量。

full probe在眼在手与外部相机的六个固定模型中都增加平移及七维MSE，支持当前映射依赖原视觉信息、强信息移除会损害拟合；它没有分离空间几何、纹理及分布外输入影响。[原4像素增强负结果](libero-runtime-native-prefix-cross-task-eye-hand-augmentation-diagnostic-2026-10-28.md)继续保留，不能因为本次少数弱干预改善就重启或扩大该训练臂。

## 实际成本、来源与独立审计

北京时间 **17:45:04–17:52:10**，主程序elapsed **426.5063s（7.11分钟）**。实际六个checkpoint加载，两个source HDF5打开，60条支持示教、8690行动作、8630个causal公开观察，排除60个t0；7200主前向/207120行，48重复/1536行，总208656行预测。分析seed20261022；训练seed仍为原三seed。没有环境被创建，本轮没有新environment seed或官方初态索引。

900次原预测、900次uint8/float恒等、48次重复、300条输入摘要、8630行目标/命令mask/前一夹爪命令均精确匹配。180次逐demo权重/grad检查通过，六模型权重不变；没有optimizer、backward、Policy控制、初态/private或封存角色数值读取。整文件SHA校验与组数值访问分开：HDF5数值解码只在授权support组，不解码query组或初态属性。

| 证据文件（`/tmp/`下） | SHA-256 |
|---|---|
| `libero-task4-task7-expert-camera-target-error-20261102-v1.json` | `787bf42f4b66d472d4a73e09fb9f9d09713ad87f671474c762678a7de26c6b13` |
| `libero-expert-camera-target-error-independent-audit-20261102-v1.json` | `dc2f01442e5bbd9a2f1df8804b8feaa21376bb236617cb20d740cd1b915bc7e4` |
| `audit_libero_expert_camera_target_error_20261102_v1.py` | `99374bce63d1b9ba6908adcc3dfd8e35f8d370a851491b54c44050beea811af6` |
| `libero-expert-camera-target-error-20261102-v1.log` | `377340ada38df03f283218beb969f997ca6542862f4161aca9f387e2ee660ee4` |

两份原始预测/逐行误差NPZ、两份参数JSONL及事件日志的完整路径/hash在结果中。事件文件SHA为`aafe8b5d228466f432fe41f025eb308d090ac8187352f1ab461733469bb9d7e4`。没有覆盖旧设计、配置、checkpoint或历史结果。

独立审计elapsed **2.07168s**，不调用执行入口/算子，不加载Policy或重新前向，不再解码HDF5/RGB。核验原参考目标/索引/mask、原预测和全部重复、7248个前向事件/预测摘要、300条原输入摘要及参数代数；从保存预测独立重算逐行五项误差、全部命令/逐demo/等demo统计和flag，共289550项递归核对。浮点最大绝对差`4.44e-16`，统计复算沿用1e-12；索引、原预测、目标、mask和hash仍精确检查，不放宽历史门槛。

审计范围限制：未额外重读原RGB或独立重做图像变换；图像依据为原批输入摘要、冻结算子、合成算子测试和已保存参数代数，不冒称图像二次独立复现。主程序最大误差恒等式残差`3.33e-16`、分量恒等式残差`2.22e-16`，均低于冻结门槛。

## 验证与下一步

新增11项执行测试、相关49项通过；CPU回归 **677 passed in 15.46s**，排除当前环境可选Crafter依赖的`tests/test_crafter_*.py`、`tests/test_spatial_gather_spt.py`和`tests/test_smoke.py`，不声称全量pytest。新入口py_compile、审计内存compile、环境检查和`git diff --check`通过；最终文件/provenance核验在`/tmp/libero-expert-camera-target-error-final-verification-20261102-v1.json`。

按冻结规则，本轮没有产生统一的平移误差放大或空间细节主线索，不选择新增强、损失或checkpoint。保持原baseline500与闭环5步；旧局部夹爪加权和4像素增强的停止决定保持，不将次级七维flag升级为训练许可。

下一步先只读已有逐demo/命令分层，定位task7平移改善而夹爪恶化的分量取舍是否有跨seed共同模式，再决定是否有依据设计单一训练变量；不追加本轮前向或环境。需要真实闭环纠偏时仍须合法状态—动作证据，不能用这些合成观察、支持域拟合结果或无标签自由运行数据伪造专家监督。本项不进入SPT/FOMAML/Knowledge Evolution或Module qualification。

后续只读更新（2026-11-03）：[task7分层定位](libero-task7-expert-camera-component-tradeoff-diagnostic-2026-11-03.md)已完成，未追加本轮前向或环境。共同局部模式见于expert close的非切换窗口，幅值偏离占正向夹爪恶化量较大部分；存在共同帧和demos3/10的取舍，但全局方向不一致。原主flag与本报告结果保持。下一步先核对公开夹爪动作映射，暂不改训练。
