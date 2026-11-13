# 来源candidate原生目标代码与保存终点位姿只读核对（2026-11-07）

承接[close与几何对齐](libero-source-candidate-close-geometry-alignment-results-2026-11-07.md)，只读核对“靠近目标却未完成”。当前任务只授权已有记录和成功判定源码，不训练、重放、采集、选择checkpoint或解封任何角色。

## 固定数据与科学边界

全部task0/1/3/4/7、seed20261106/07/08、A_original/B_fomaml、candidate0..3，共120条已知source短轨迹、60个场景对；保持30个500-update端点、5步稳定化、现有loss和G停止/H未开展。原始已审计capture/对齐结果与completion作谱系输入，SHA在新配置固定；原始轨迹、checkpoint及历史文件不改写。

每条只派生其保存终点及最后6个控制边界[n−5,n]的目标代理（不足时保留实际数量，不补帧），120个终点、最多720行。读取同一120份公开JSON和120份geometry sidecar，原有17247个几何边界由既有纯标准库验证器核验；不把其他中间帧搜索为新“最佳终点”。主计算一次、独立审计一次，均不导入Torch、NumPy、MuJoCo、LIBERO或robosuite，不读取checkpoint、初态、HDF5或RGB，不创建任何真实环境/影子数据。可读取已固定源码和这五个source BDDL；validation、held-out、qualification、reserved和final payload预算0。

这是已知结果背景下的描述性核对，不是盲检验或新的formal result。终点时刻依赖成功提前停止，不把不同终点两臂作处理效应，不换算300步成功率。

## 从原生代码取定义

五个BDDL的目标均为`And(On akita_black_bowl_1 plate_1)`，从problem的`_check_success`经predicate registry、`On.__call__`到物理ObjectState的`check_ontop`，不同于SiteObjectState。绑定当前LIBERO revision、源码SHA及旧采集相关依赖SHA；只做文本/AST检查，不执行原生模块。

本轮代码检查已发现原生条件为：plate root z≤bowl root z、碗与盘的原生contact数组中有匹配geom记录、root原点XY距离严格小于0.03米。3cm来自源码，不是根据结果调出的诊断阈值。原生contact匹配不检查dist≤0，因此同时报告影子plate的全部碰撞记录、非正距离记录和正margin记录。不可用双侧手指代理替代碗/盘接触。

不额外加入gripper open、脱离、接触力、朝向或持续时间门槛；朝向只作为保存姿态的描述量，不解释为原生独立条件。源码结构、参数顺序、比较符号或contact语义变化按fail-closed拒绝，不能迁就数据改谓词。

## 计算与解释

每个保存边界输出碗/盘world root坐标、XYZ/XY距离、原生半径减XY的有符号margin、bowl z−plate z、原生式plate记录门及三个门的合取，另列dist≤0口径作对照。分开保存手指双侧代理、命令符号、碗相对reset高度、两物体up轴夹角。全部组保留，各task×seed×arm及task×arm/总体报告重叠失败项、互斥组合、均值/中位数与分母，不做显著性、阶段阈值或训练变量搜索。

原生成功函数读native缓存body_xpos/contact，本次sidecar是当前状态在独立MjData上位置重算。原生判定输入及确切缓存时刻未保存，不能由影子位姿逐项精确复算；上一个0.05秒控制边界也不能冒充某个原生物理子步缓存。保留原生目标标签，并单列影子代理不一致，不自动改判成功或认定环境bug。

入口和独立审计分别为`experiments/analyze_libero_source_candidate_native_goal_pose.py`与`experiments/audit_libero_source_candidate_native_goal_pose.py`。新增边界测试通过、编译后，将配置、源码、测试及本说明绑定，再运行只读核对。主结果、三张CSV及独立审计均写入新的`/tmp/libero-source-candidate-native-goal-pose-*20261107-v1*`路径，拒绝覆盖。结果另写dated report，保存完整provenance及零研究调用凭据。

最后根据原生条件与保存位姿的支持、反例和不可重建项判断训练修改是否有依据；不因成功标签与代理一致就宣布根因，不把几何或目标阈值输入Policy/loss，也不恢复G准入或更新SPT/Knowledge/Module。
