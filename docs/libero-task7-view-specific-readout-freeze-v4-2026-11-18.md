# task7视角专用读出v4（候选A）：科学冻结记录（2026-11-18）

**唯一候选A已科学冻结，状态`scientific_frozen_execution_not_bound`。本轮只做空白网络/人造输入验证；研究载荷及其SHA读取、真实训练和环境物理步均0。未创建任何v4阶段执行绑定、真实结果或checkpoint目录。**

[规范协议](libero-task7-view-specific-readout-protocol-v4-2026-11-18.md)定义全部角色/预算/初始化与拒止规则；[结构化证据](evidence/libero-task7-view-specific-readout-freeze-2026-11-18.json)提供哈希和验证索引。

## 锁定的三项边界

1. **读出组件级对照。** 候选两视角各自独立的Conv512/256→SpatialSoftmax(τ=1)→512/608/128坐标MLP，末层Linear无ReLU；与本轮新训flat_zero_xyz配对。候选1507272/控制1514120可训练参数，差−6848（−0.452276%）。每视角520928参数。独立参数与宽度1408→608同时改变，不能单独归因取消共享；共享3A仅历史背景，不冒充第三臂。骨干仍共享冻结，268维接口与XYZ内部置零保持。
2. **确定性独立初始化与视图熵。** agentview=`seed+9400001`、eye_in_hand=`seed+9500001`，六个分支seed不同、参数不别名、不镜像复制，局部RNG不改变调用方/批次流。共同层逐项同初始化。分别记录两视图熵和/均值，排除padding、不额外前向、不入loss或继续门；熵差不证明定位或专门化。两分支最近熵加两份Policy副本上限81920标量，逐视角调用/元素预算独立记账。
3. **新配对控制与原门。** 三seed20261106/07/08，每臂500实际更新、同500批×8；冻结ResNet18、Adam3e−4、uniform masked七维MSE、稳定化5。原22来源/44条交接，时间轴≤300、物理步上限15980。候选−本轮新控制≥4/22且各seed不退化、独立审计通过后才可新绑120条完整source开发评估。12/22仅是控制恰为8时的例子，不能改成固定候选门；控制9则需13。full上限36600步，后续门仍净≥6/60且逐seed不退化。

动机是三条seed07来源的近盘水平偏位；主终点仍为原生成功，未来44条私有对象状态查询预算0，成功率提升不能直接当作量出了XY误差或证明视觉根因。当前不训练候选、不新增几何或开放任何封存数据。

## 实际验证与执行限制

136项定向测试通过：v4新增57、父v3回归79，最终记录2.70秒；未跑全量pytest。空白真实ResNet18结构核对精确参数量、三seed共同层相等、两个视角分别按固定offset复现、不加载/下载权重；人造RGB/缓存动作与梯度一致、两支卷积均有梯度、骨干无梯度/BN不变、XYZ梯度0、正确视图路由、熵mask/快照/失败后拒旧诊断及81920存储上限均通过。

协议测试核对500批完全配对、44/120条与逐视角最大预算、净增3/4边界、控制0/8/9下的相对门、seed退化、配对/布尔类型、科学定义篡改、源载荷禁止纳入依赖、凭证身份篡改和科学绑定拒绝三个真实阶段。CPU环境检查通过。初版真实父链预检发现v3 completion不含science字段，已在科学绑定前改为现有audit中的science/binding/result链验证并新增篡改拒止测试；旧凭证未改。

这不是完整运行器6模型训练＋44/条件120条的端到端合成验证；本轮尚未实现v4执行入口/guard、独立训练或闭环审计。后续先补齐并验证这些执行工作，再建立阶段绑定；不得改旧SHA或放宽科学拒止函数来执行。

## 文件与哈希

- 模块：`src/algorithms/libero_view_specific_readout_policy_v4.py`。
- 协议校验：`experiments/libero_task7_view_specific_readout_protocol_v4.py`。
- 科学配置：`configs/libero-task7-view-specific-readout_v4.json`。
- 科学绑定：`configs/libero-task7-view-specific-readout-freeze-binding_v4.json`。
- 验证日志/冻结回执：`logs/libero-task7-view-specific-readout-freeze-20261118-v4/`。

975项代码/元数据依赖继承已审计v3与9条定点几何的闭包，三个最终验证文件另列。依赖不含原HDF5、权重、NPZ、初态载荷；原数值描述符仅沿用既有元数据。

科学配置SHA：`41288dc87132847bd6f4a220a5c45020e8f7c4643b56b9b4fd7182906bd65608`。

科学绑定SHA：`ee55b62f7a3823ce4ff17857c22fcf8863cb719eed4811ec77afdc31f99c5599`。

LIBERO继续；Policy500、稳定化5、现loss和正式封存保持。仍是已用source开发域的读出组件诊断，不声称正式Skill Evolution迁移、SPT/SPI或完整KECRL验证，不注册Module、不更新Knowledge/SPT。
