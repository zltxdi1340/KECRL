# 固定Policy局部夹爪状态受控读出结果（2026-11-04）

**已完成2457个主前向批次＋54个重复批次，全部18个固定端点通过精确复现；主事件锚点与次级lag1均未形成预设联合改善线索。** 两条件各18模型的局部平移MSE和夹爪MSE均数值增加，符号错误率没有任何模型下降。不采用回退/保持夹爪输入作为训练或执行修复，继续500 updates、闭环默认稳定化5步及现有loss/观测。没有新增训练或环境交互。

## 设置与研究边界

遵循[科学协议](libero-source-internal-input-sensitivity-review-and-gripper-state-control-protocol-2026-11-03.md)和[执行前冻结记录](libero-source-internal-gripper-state-control-execution-protocol-2026-11-04.md)。只用task0/1已暴露source-support demos0..29，三seed20261022/23/24，A_original/B_fomaml/C_pooled_bc全部18个固定500-update端点。A为原始初始化适应后的Policy，B为隔离FOMAML候选初始化适应后的Policy，C为普通预训练初始化适应后的Policy；本轮三者均固定权重，没有SPT更新或新适应。

样本为全部open事件后1..5步且专家仍open的149/157帧；按原专家事件/命令选帧，不按模型错误、成败或有利seed筛选。原正确/错close/neutral、共同错误、同帧混合组、lag和逐demo统计在读出前固定。原批次内全部6954个source步保留，未处理6648帧及65个at-event参照不替换。

输入/目标仍为`obs[t-1] -> original action[t]`、t>=1。仅两维公开gripper_states变化：主条件取同open事件e的pre-action `obs[e-1]`，次级取过去一步 `obs[t-2]`；锚点来自不可变原缓存，同demo、严格过去。两相机、joint、原专家目标、权重及其他行固定。离线额外稳定化0，闭环默认5步未变。

## 主结果：全部task、arm和seed

以下均为替换后减identity。符号错误率列为百分点（pp）；MSE为绝对变化，不是相对百分比。正值表示相对原专家目标的误差增加。每组均保留三个seed；均值不合并任务或作为独立样本显著性结论。

| task | arm | 符号错误率Δ（三seed均值，pp） | 平移MSE Δ均值 | 夹爪MSE Δ均值 | 正确→错误读出数 | 错误→正确读出数 | 主cue |
|---|---|---:|---:|---:|---:|---:|---|
| 0 | A_original | +0.4474 | +0.000338 | +0.005802 | 2 | 0 | false |
| 0 | B_fomaml | +0.2237 | +0.000555 | +0.004724 | 1 | 0 | false |
| 0 | C_pooled_bc | +0.4474 | +0.000486 | +0.004962 | 2 | 0 | false |
| 1 | A_original | +0.2123 | +0.000674 | +0.013601 | 1 | 0 | false |
| 1 | B_fomaml | +1.2739 | +0.000488 | +0.010380 | 6 | 0 | false |
| 1 | C_pooled_bc | +1.0616 | +0.000580 | +0.012229 | 5 | 0 | false |

主条件18模型中12个符号错误率增加、6个不变；合计17个“模型×帧”由原正确变错，0个由错变对。17不是17个独立专家场景，多个模型可能重复同一源帧。18个模型的平移MSE和夹爪MSE均增加；变化较小，未执行或声称统计显著性。六个task×arm的三seed联合主cue全部false。

逐seed结果（22/23/24分别指20261022/23/24；每模型task0 n=149、task1 n=157）：

| task | arm | seed后两位 | 符号错误率Δ（pp） | 平移MSE Δ | 夹爪MSE Δ | 正确→错误 | 错误→正确 |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | A_original | 22 | +0.0000 | +0.000252 | +0.001135 | 0 | 0 |
| 0 | B_fomaml | 22 | +0.0000 | +0.000274 | +0.002357 | 0 | 0 |
| 0 | C_pooled_bc | 22 | +0.0000 | +0.000403 | +0.003472 | 0 | 0 |
| 0 | A_original | 23 | +0.6711 | +0.000156 | +0.008164 | 1 | 0 |
| 0 | B_fomaml | 23 | +0.0000 | +0.000801 | +0.005771 | 0 | 0 |
| 0 | C_pooled_bc | 23 | +0.6711 | +0.000446 | +0.004095 | 1 | 0 |
| 0 | A_original | 24 | +0.6711 | +0.000607 | +0.008107 | 1 | 0 |
| 0 | B_fomaml | 24 | +0.6711 | +0.000589 | +0.006044 | 1 | 0 |
| 0 | C_pooled_bc | 24 | +0.6711 | +0.000610 | +0.007319 | 1 | 0 |
| 1 | A_original | 22 | +0.6369 | +0.000903 | +0.017869 | 1 | 0 |
| 1 | B_fomaml | 22 | +1.9108 | +0.000688 | +0.014116 | 3 | 0 |
| 1 | C_pooled_bc | 22 | +0.6369 | +0.000926 | +0.015138 | 1 | 0 |
| 1 | A_original | 23 | +0.0000 | +0.000550 | +0.012188 | 0 | 0 |
| 1 | B_fomaml | 23 | +0.6369 | +0.000179 | +0.005171 | 1 | 0 |
| 1 | C_pooled_bc | 23 | +0.6369 | +0.000185 | +0.008883 | 1 | 0 |
| 1 | A_original | 24 | +0.0000 | +0.000568 | +0.010746 | 0 | 0 |
| 1 | B_fomaml | 24 | +1.2739 | +0.000595 | +0.011853 | 2 | 0 |
| 1 | C_pooled_bc | 24 | +1.9108 | +0.000628 | +0.012665 | 3 | 0 |

## 次级条件和输入响应

lag1没有救回主失败：18模型中5个符号错误率增加、13个不变，0个下降；5个模型×帧由正确变错、0个纠正。各18模型的平移/夹爪MSE也都增加，六组次级联合cue全部false。

| task | arm | lag1 符号Δ均值（pp） | lag1 平移MSE Δ均值 | lag1 夹爪MSE Δ均值 | 正确→错误 |
|---|---|---:|---:|---:|---:|
| 0 | A_original | +0.2237 | +0.000155 | +0.003181 | 1 |
| 0 | B_fomaml | +0.2237 | +0.000270 | +0.002729 | 1 |
| 0 | C_pooled_bc | +0.0000 | +0.000228 | +0.002824 | 0 |
| 1 | A_original | +0.2123 | +0.000341 | +0.006908 | 1 |
| 1 | B_fomaml | +0.0000 | +0.000254 | +0.005394 | 0 |
| 1 | C_pooled_bc | +0.4246 | +0.000298 | +0.006296 | 2 |

两个条件各18模型的夹爪连续输出平均变化均为正，即平均向close方向移动；这是固定Policy对该分量的局部输入响应，不能解释为夹持力增大，也不代表每个帧都同向。原符号错误未被修复，不能因为响应存在就认定它是闭环错误的主因。

全部306个目标帧在两种替换下输入均实际变化，没有“选中但输入未变”的帧。L2为原数据两维公开状态的坐标变化：

| task | 主锚点L2均值／中位数／最大值 | lag1 L2均值／中位数／最大值 | 主／次无变化帧 |
|---|---|---|---:|
| 0 | 0.005654／0.004026／0.019722 | 0.002871／0.002732／0.006617 | 0／0 |
| 1 | 0.005677／0.003567／0.019988 | 0.002838／0.002575／0.006309 | 0／0 |

事件锚点和lag1同时改变时间距离与输入变化幅度；两者不是严格等幅对照，不据此搜索更合适强度或挑锚点。

## 解释和下一步

本结果不支持“把当前公开夹爪状态回退/保持，就能解除这些open后错误”的具体修复假设。它也不是夹爪反馈完全无用的证据：Policy确实对替换有响应，只是相对于原专家目标没有得到所需改善。

原图像仍包含手指外观，替换后的夹爪状态与RGB/joint可能不符合真实状态；原专家动作不是混合观察上的有效新标签。因此不能证明真实夹爪反馈有害/有益、物理释放成败、动作head或视觉是唯一原因，也不能从source-support读出推断query泛化或闭环成功。旧task4/7审计与本task0/1结果保持分开。

下一步先只读这次已保存的预测，检查新增错号读出的原输出离0多远、所在lag及原错误/正确对照的响应方向，区分临界帧翻号与持续错误；不加前向或训练，再据证据选择另一个单分量机制假设并另冻协议。当前不调整loss、sequence length、增强或训练预算。

## 实际预算、provenance与验证

| 项目 | 实际值 |
|---|---:|
| 固定checkpoint加载／权重不变检查 | 18／18 |
| 主调用尝试／完成 | 2457／2457 |
| 重复调用尝试／完成 | 54／54 |
| 主／重复有效预测行 | 187758／3996 |
| 主／重复含padding槽 | 196020／4320 |
| 原预测逐元素精确行／原batch摘要核对 | 62586／819 |
| 未处理有效行／padding／at-event精确比较 | 119664／5508／1170 |
| 重复完整输出精确核对 | 54 |
| 局部主评分行（有效行子集） | 8262 |
| 新共享source观察每模态／action行 | 6954／6954 |
| action0／初态／private state读取 | 0 |
| 新训练更新／backward／Policy控制步／环境调用 | 0 |

Runtime：physicalGPU1 RTX4090，UUID `GPU-d90bd7b2-16f6-c0fe-fbc1-7ed4060e95ff`，使用前32MiB/0%且无该卡compute进程；进程cuda:0，GPU0保持其他作业。Python3.10.22、Torch2.4.1+cu124、CUDA12.4、NumPy1.22.4、h5py3.11.0；threads1、analysis seed20261022、确定性开启、TF32关闭，启动环境与绑定一致。没有Policy warmup或额外pool前/反向。

入口耗时 **41.77s**；cache 0.99s、模型读出和检查 22.56s、数值汇总 17.29s。host cache峰值415117180 bytes，CUDA tensor峰值149238784 bytes；CUDA值是PyTorch分配峰值，不是整卡总占用。完成时间2026-11-04 01:15:21（Asia/Shanghai），原UTC时间和启动命令见结果。耗时短是固定模型批量读出，本轮训练为0，不是降低了500-update端点预算。

执行前90项相关专项、三个新Python文件编译和CPU环境检查通过；环境检查显式隐藏CUDA，device_count0不表示CUDA路径失败。独立审计在固定代码下完成 **605232项标量核对**，重算所有分层、逐demo/等demo、主次cue、剂量和原始数组parity。神经预测0容差；1e-12只用于描述性算术复算。审计未新增Policy/HDF5/RGB/环境读取，仅核对缓存数组和checkpoint文件hash。未修改共享框架，不扩大到缺少可选Crafter依赖的全量pytest。

证据：

- 结果：`/tmp/libero-source-internal-gripper-state-local-control-20261104-v1.json`，SHA `01f8b58b9bf0a880629778547b4bc745b8c757f5ed35c69baaf4702997519ffa`。
- 独立审计：`/tmp/libero-source-internal-gripper-state-local-control-independent-audit-20261104-v1.json`，SHA `b81214ffadd6e53682f445a068f53c2c6fa8d5e84ce8f88d93ede08319914096`。
- 科学配置SHA：`bf38389d06d2d43c74991365d4763cc0cbbe94ee8517ef1c9cf51ca4d4d40206`。
- 设计绑定SHA：`e81809ebff82d4adebf39aafb2b13cbd963db9d69a4c30aceed155ddca8257d8`。
- 执行绑定SHA：`89975352f247f5b8e8a03421294efdadc7cb2596cfeba11d6e09a785c03cb199`，见[绑定文件](../configs/libero-source-internal-gripper-state-local-control-execution-binding_v1.json)。
- Artifact目录：`/tmp/libero-source-internal-gripper-state-local-control-artifacts-20261104-v1`，含18份三条件预测/含padding与重复数组、两份公开gripper与输入剂量、逐调用与源切片事件。各hash在结果中。
- 最终保护与文档检查：`/tmp/libero-source-internal-gripper-state-control-execution-final-verification-20261104-v1.json`。

没有异常中止、补样本、重试、容差放宽、checkpoint选择或追加预算。validation、held-out、qualification、reserved及final evaluation保持封存；无活动SPT接受、Knowledge更新或Module注册。本结果仅为固定source Policy的输入响应诊断。
