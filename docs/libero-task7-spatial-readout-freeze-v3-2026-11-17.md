# task7视觉读出3A v3：科学冻结与合成验证（2026-11-17）

**唯一3A候选已科学冻结、模块/协议合成验证通过；状态`scientific_frozen_execution_not_bound`。真实训练、闭环、源载荷/SHA读取均0，未建立任何阶段执行绑定。** 这是准备记录，没有新研究成功率。

方案、具体源字段/预算、熵口径与停止规则见[规范协议](libero-task7-spatial-readout-protocol-v3-2026-11-17.md)。科学配置`configs/libero-task7-spatial-readout_v3.json`，科学绑定`configs/libero-task7-spatial-readout-freeze-binding_v3.json`，模块`src/algorithms/libero_spatial_readout_policy_v3.py`，协议校验`experiments/libero_task7_spatial_readout_protocol_v3.py`。

## 锁定内容

两臂均268维、XYZ内部生成0，不读EEF。控制为原共享Flatten/Linear/ReLU，候选为Conv1×1(512→256,bias=False)→4×4 Spatial Softmax→512期望坐标→Linear512/1408→ReLU→Linear1408/128。按用户结构，候选末层没有ReLU。温度固定非训练τ=1.0，坐标是特征网格归一化位置，不是已标定物体中心。

控制/候选可训练参数精确为1514120/1499144（差−14976，−0.98909%），共享视觉读出参数1048704/1033728，总参数12690632/12675656。共同骨干、fusion、frame MLP和动作头逐项同初始化；读出不同形状，不能称完整初始化相同。读出使用局部seed+9200001，数据使用局部seed+7×10000+777；3seed固定20261106/07/08，500批×8完全配对。

熵从已有前向取得，排除填充帧、视图/有效帧/通道等权；控制为null，候选记录两视图熵和/均值及归一化均值。记录必须先于parity/optimizer，不新增前向、无熵loss或选点/调温/继续门。高熵或低熵都不是定位正确性的证明，均匀分布不按技术失败处理。

未来训练为6模型×500更新、CPU单线程、原Adam3e-4/clip1/uniform masked七维MSE、batch8/seq10/stride10、last500。先44条同状态交接，上限15980物理步，净增≥4/22且每seed不退化、独立审计通过后才可新绑120条完整source开发评估（上限36600步）；full后续门为净增≥6/60且各seed不退化。旧8/22、10/22、3/22均不得充当本轮新配对控制。

## 实际验证和未完成的执行工作

165项定向测试通过：新增79、父组件回归86，CPU日志记录5.36秒；未跑全量pytest。覆盖：

- 空白真实ResNet18结构的精确参数量，3seed共同层state逐项相同、构造不改变调用方RNG，不加载原始/公开权重；
- NumPy独立坐标/熵公式、均匀/饱和边界、轴方向/平移不变性、双精度gradcheck；
- 人造RGB/缓存动作及梯度一致、1×1卷积梯度非零、骨干梯度None/BN不变、XYZ三列梯度0、在线/序列一致；
- mask排除填充、快照detach且不入state_dict、无效/失败前向拒绝旧诊断、读出guard与重复诊断不追加前向；
- 500步成对batch、两阶段成对ID/固定seed覆盖、门边界与seed退化、最大预算整数重算、配置/依赖篡改和科学冻结不能开启真实阶段。

编译、CPU环境、差异检查通过。首轮模块22通过；首轮协议49通过/3失败，扩大回归162通过/3失败，均因full ID尾部初态编号被当作交接控制臂后缀。已分别严格验证两类ID，未放宽门；研究载荷访问前修复，旧失败日志完整保留。

本阶段尚无新执行入口/访问及调用预算guard/独立训练和闭环审计，也未做“完整运行器6模型训练+44/条件120条”的合成流程。现有模块/协议测试不冒充该端到端验证。后续先补齐这些执行工作，再锁定新阶段执行代码与授权；不能用本科学绑定直接打开研究载荷。`require_execution_binding()`对train/handoff/full均明确拒绝。

## 绑定和证据

942项代码/元数据依赖及3份最终验证日志已绑定，继承v2交接931项闭包并核验已审计v2父链；没有把源HDF5、权重、NPZ或初态文件放进科学冻结依赖。源描述符沿用既有metadata，不重算原数值SHA。

科学配置SHA：`95b0b7e1cf18349b3fd851c8fd298d9d2788431d24e501924376ab2acdabc009`。

科学绑定SHA：`7763db8609559363a9ca36b4df13ffff49e8c06d338fef1b77c0d407b25e146b`。

[结构化冻结证据](evidence/libero-task7-spatial-readout-freeze-2026-11-17.json)；完整初始/失败/最终测试、环境和冻结回执在`logs/libero-task7-spatial-readout-freeze-20261117-v3/`。未创建研究输出目录或checkpoint目录。

LIBERO继续，500更新、稳定化5、现loss与正式数据封存保持。未将图像网格转换为厘米精度下限，未声称恢复亚像素或证实视觉根因；无Module、Knowledge/SPT更新、部署或checkpoint/seed择优。
