# 来源candidate close与几何只读对齐口径（2026-11-07）

用户授权只读对齐已有close段、接触代理、碗高度和目标距离，再决定训练修改。依据[已完成的采集与独立审计](libero-source-candidate-geometry-capture-results-2026-11-07.md)，本次只使用这120条来源前缀及359个命令段；先前结果和task0/3反例已知，不称为盲分析或新正式假设检验。编写前只检查了一个episode的两类JSON字段及已存命令段表，未新运行Policy或环境。

## 固定口径

- 全部task0/1/3/4/7、seed20261106/07/08、A_original/B_fomaml、candidate0..3保持，共120条、60对、15个task×seed组。所有30个权重仍是500端点，5步稳定化和均匀七维masked action MSE保持。无训练、checkpoint/初态/HDF5/RGB读取、额外环境/影子计算；不访问validation、held-out、qualification或reserved。G停止及H未开展保持。
- 重算全部359个正/负/零最大命令段，与已存CSV身份、起止、符号、长度、删失逐项一致后，分析全部170个close段。动作k从状态k−1到k；close段[s,e]的状态窗口固定为[s−1,e]，时长为sim.time[e]−sim.time[s−1]。右端没有观察到换号的段仍为右删失；末尾单步正号不称完整脉冲。
- 对每段保存pre/end/min/max的末端至碗XYZ/XY/有符号Z、碗相对reset高度和碗至plate XYZ/XY。另取段内首次双侧代理、连续代理块数、各块边界数、首块是否在仍close时结束、段末是否有代理。段前边界已有代理单列，不改写为新接触。代理块边界数与经历时间不同，不称稳定抓取或接触力。
- 首次代理至该close段末的高度和目标距离变化，以及首次代理至整个已存前缀末的变化，分别保存；后一口径可能跨后续open/close，不当作同一close效果。仅当下一动作直接为负号时保存immediate-open后的状态；若先出现0，不跨越0合并。单次close换号为open不意味着真实物体释放。
- 保留120条episode级和170条run级的不同权重口径；全部task×seed×arm、task×arm及总体输出均值、中位数、范围和实际观察分母。距离、碗高度、目标距离均为连续量，不搜索接近、抬起、运输或抓稳阈值。
- 配对逐状态表覆盖每对两臂的保存范围并保留availability，缺失填空而非0/上一帧。比较仅使用同一绝对状态、共同可见截止min(nA,nB)，不把两臂不同终止时刻作处理效应。每对输出共同截止的高度最大值及目标距离，保留完整原生前缀终止信息但不换成300步成功率。
- 每个close的pre-onset、post-end、段内首次代理和直接后继open的post-state作锚点，合并相同绝对状态的别名。固定±5个状态窗口，分别列reset左截断和终止右截断，不延长、不补帧。与旧采集报告中的首次事件窗口定义分别标明。

## 执行、审计与产物

[分析入口](../experiments/analyze_libero_source_candidate_close_geometry_alignment.py)复用原标准库保存字段验证器，不导入数值框架或环境；读取绑定的主结果、审计、completion、解释JSON、旧命令段CSV及其列明的240份新增sidecar。独立审计入口不导入分析实现，另读同一240份JSON，重新按逐动作扫描构造段、核对全部派生单元格、配对共同时间、窗口及汇总分母。两个计算pass均只处理既有保存状态，不能追加仿真。旧主结果、协议、执行绑定、源码、checkpoint及原始轨迹不修改。

先完成合成边界测试、编译和metadata校验，再将本口径、配置及源码单独绑定后运行。配置与绑定为`configs/libero-source-candidate-close-geometry-alignment-{readonly,binding}_v1.json`。新主结果和独立审计分别存于`/tmp/libero-source-candidate-close-geometry-alignment-20261107-v1.json`及`/tmp/libero-source-candidate-close-geometry-alignment-audit-20261107-v1.json`，五张新CSV存同前缀tables目录，全部拒绝覆盖。

审计通过后，[绘图入口](../experiments/plot_libero_source_candidate_close_geometry_alignment.py)仅从新配对标量表生成5页PDF和SVG，覆盖全部15个task×seed格：close/代理比例、末端至碗距离、碗高度、碗至plate XY距离及共同可见场景数n(t)。仅画相同绝对状态下共同可见样本的中位数/比例，成功停止后的样本不填充；state k的命令是产生该状态的动作，pre-onset另查事件表。绘图使用标准Matplotlib，不读模型或原始侧录。

不做显著性检验、因果归因、跨环境合并、阈值搜索、挑seed或checkpoint。只根据描述性支持、反例和未解问题，决定是否已有足够依据另冻单变量训练诊断；本次不实施训练修改，不恢复G准入或更新SPT/Knowledge/Module。
