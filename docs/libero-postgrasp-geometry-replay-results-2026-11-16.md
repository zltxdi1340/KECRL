# task7几何补录v1：技术中止与只读核查（2026-11-16）

**已建立新执行绑定并真实启动；6/36条完成，第7条在第153步腕部图像不一致，按零误差门停止全部后续。29条未启动，整轮独立审计未运行，不能产出36条物理归因比例或选择结构v2。** 不是训练/控制器效果的新结果。

## 各步目的、配置和结果

| 阶段 | 目的及配置 | 实际结果 |
| --- | --- | --- |
| 实现与人造验证 | 全36条、434 reset、12970 native最大包络；像素/1ULP关节差、额外原生/零动作、On和预算攻击 | 99项定向与相关回归测试通过；编译、环境和diff检查通过，未跑全量pytest |
| 执行绑定 | 新36条/19行source初态/四项观测口径、事前互斥首项顺序；原500/5/MSE保持 | 886项代码/静态源依赖和新授权锁定，真实输入前通过preflight |
| 真实补录 | task7、GPU1/EGL1、seed20261022、单线程；固定原动作，不运行Policy | 6条完整，第7条flip01-confirm5-frame_mlp在153步停止；无重试/跳例 |
| 只读故障核查 | 单独绑定889项依赖；7项人造测试；仅两份容器腕部152、153帧，核对两份journal | 152步精确相等；153步仅4像素的5个通道不同，全部原值+1；0新增环境/初态/模型 |
| 独立整轮审计与v2 | 必须整36条完成且通过原公开门 | 未准入、未运行；没有归因分桶研究结果，没有v2决定 |

主预算为36环境、434reset、2170零动作+10800保存动作=12970 native；实际仅7环境、34reset、170零动作+1953保存动作=2123 native。主阶段62.823960秒，驱动64.444395秒。7次episode尝试/6次完成，未把失败的第7条计为完整episode。

实际读取1次task7初态容器、只选择性解码0..18共19行，源36NPZ全在环境前校验/数值缓存，因此源容器虽36个均读取，不代表36条都运行。216个数值成员、205524项数值，容器463475921字节。主角色原RGB解码0、Policy/模型创建/权重读取/训练/优化器/HDF5/额外render全部0。所有正式数据继续封存，未更新Knowledge/SPT或注册Module。

保留1987份新公开观察、3974相机帧、195330048原始公开RGB字节；1209份诊断样本、32643项几何输出投影、7份碰撞体校准（3850项）、6045次外层诊断谓词请求。部分样本均未通过整轮独立审计，不把它们当36条归因证据。

## 零误差门为什么拒绝

第7条交接a=150，0..152公开观察与原记录一致；153步的主相机、joint7、gripper2仍精确一致，joint/gripper最大绝对误差均0，reward和终止标志也相同。腕部eye_in_hand_rgb摘要不等，故整个公开观察门失败。该步图像/公开状态已保存，但153步私有几何未采样；几何片段只有150、151、152三帧。

两个journal逐项对照1987个保留事件，1986份摘要一致，唯一不同组件是该153帧腕部。此核查使用旧/新记录的组件摘要，不冒称重新独立解码了全部1987份图像。

原腕部153摘要：`f203ea15e77702ccc0728672fab92ad134504b1577258af6be285f6a17dd7160`。

新腕部153摘要：`77b6e0c72d338a4c5df6a38b452fa29b73020b7f895d46c5eb7f2ed51d22cf3b`。

有限像素核查逐个比对原保存数组，未翻转、重采样或采用容差：

| 原数组行、列、通道 | 原值 | 新值 |
| --- | ---: | ---: |
| 86,108,G | 166 | 167 |
| 86,109,R | 166 | 167 |
| 86,110,G | 163 | 164 |
| 86,110,B | 160 | 161 |
| 87,111,B | 161 | 162 |

总4像素/5通道，最大通道误差1，平均通道绝对误差5/49152。152帧完全相等。该原RGB解码是**另列的只读故障核查角色**：一份原轨迹及一份本轮partial，2成员、共196608个uint8通道值，ZIP流共消费15139072个成员未压缩字节（跳过前152帧字节但未转为像素数组）。它不改写主v1 `source_rgb_decodes=0`，不读取其他原帧/相机，更没有环境步。

旧/新声明并检查的environment seed、稳定化、GPU1/EGL1、线程、package版本和LIBERO revision相同；目前没有定位具体确定性配置错误。差异位置集中且仅+1，可以作为渲染/数值链路线索，但**尚未证明EGL舍入或排除未保存隐藏状态差异**。关节/夹爪是float32公开投影，相等不证明全部高精度内部状态相等。不能把小差异改成容差放行，也不能据此认为新增几何只读导致差异。

## 36条执行状态，不是物理归因统计

以下6条“完成”只指主程序通过各自公开门，整体仍未审计；第7条和29条不完整，物理首项标签一律不发表。

| 轨迹 | 本轮执行状态 | 保存动作步 | 几何样本数 |
| --- | --- | ---: | ---: |
| flip00-unchanged-frame_mlp | 主程序完成，整轮未审计 | 300 | 226 |
| flip00-unchanged-gru_window10 | 主程序完成，整轮未审计 | 300 | 226 |
| flip00-confirm5-frame_mlp | 主程序完成，整轮未审计 | 300 | 226 |
| flip00-confirm5-gru_window10 | 主程序完成，整轮未审计 | 300 | 226 |
| flip01-unchanged-frame_mlp | 主程序完成，整轮未审计 | 300 | 151 |
| flip01-unchanged-gru_window10 | 主程序完成，整轮未审计 | 300 | 151 |
| flip01-confirm5-frame_mlp | 公开门拒绝，中止 | 153 | 3 |
| flip01-confirm5-gru_window10 | 未启动 | 0 | 0 |
| flip03-unchanged-frame_mlp | 未启动 | 0 | 0 |
| flip03-unchanged-gru_window10 | 未启动 | 0 | 0 |
| flip03-confirm5-frame_mlp | 未启动 | 0 | 0 |
| flip03-confirm5-gru_window10 | 未启动 | 0 | 0 |
| flip04-unchanged-frame_mlp | 未启动 | 0 | 0 |
| flip04-unchanged-gru_window10 | 未启动 | 0 | 0 |
| flip04-confirm5-frame_mlp | 未启动 | 0 | 0 |
| flip04-confirm5-gru_window10 | 未启动 | 0 | 0 |
| flip05-confirm5-frame_mlp | 未启动 | 0 | 0 |
| flip05-confirm5-gru_window10 | 未启动 | 0 | 0 |
| flip06-unchanged-frame_mlp | 未启动 | 0 | 0 |
| flip06-unchanged-gru_window10 | 未启动 | 0 | 0 |
| flip06-confirm5-frame_mlp | 未启动 | 0 | 0 |
| flip06-confirm5-gru_window10 | 未启动 | 0 | 0 |
| flip08-unchanged-frame_mlp | 未启动 | 0 | 0 |
| flip08-unchanged-gru_window10 | 未启动 | 0 | 0 |
| flip08-confirm5-frame_mlp | 未启动 | 0 | 0 |
| flip08-confirm5-gru_window10 | 未启动 | 0 | 0 |
| flip09-confirm5-frame_mlp | 未启动 | 0 | 0 |
| flip09-confirm5-gru_window10 | 未启动 | 0 | 0 |
| flip10-unchanged-frame_mlp | 未启动 | 0 | 0 |
| flip10-unchanged-gru_window10 | 未启动 | 0 | 0 |
| flip10-confirm5-frame_mlp | 未启动 | 0 | 0 |
| flip10-confirm5-gru_window10 | 未启动 | 0 | 0 |
| flip12-unchanged-frame_mlp | 未启动 | 0 | 0 |
| flip12-unchanged-gru_window10 | 未启动 | 0 | 0 |
| flip12-confirm5-frame_mlp | 未启动 | 0 | 0 |
| flip12-confirm5-gru_window10 | 未启动 | 0 | 0 |

四类失败及新互斥首项桶的数量均为未能提供，不填0，也不对先完成的3对做总体外推。没有将技术中止混成控制失败、成功率或结构收益。

## 实现、证据与后续

新增主入口、限定对象的GeometryReader、访问/缓存/域/原生/预算guard、全部四项量化指标、互斥首项规则和独立审计；原生On仍为接触、Z关系和严格XY<3cm，不加独立松爪门。Z采用目标50个box的当前姿态外包络，盘上界不冒称准确承托面。完整规则见[事前执行协议](libero-postgrasp-geometry-replay-protocol-2026-11-16.md)，并未因本次失败修改它。

证据：[新科学配置](../configs/libero-postgrasp-geometry-replay_v1.json)、[执行绑定](../configs/libero-postgrasp-geometry-replay-binding_v1.json)、[只读核查协议](libero-postgrasp-geometry-stop-readonly-protocol-2026-11-16.md)、[紧凑证据](evidence/libero-postgrasp-geometry-replay-stop-2026-11-16.json)。原始failure、41份保留NPZ、journal、admission-status.csv在`results/libero-postgrasp-geometry-replay-20261116-v1/`；两帧核查在`results/libero-postgrasp-geometry-stop-readonly-20261116-v1/pixels.json`；日志和测试在同名`logs/`。

下一步应先解决短程复现准入，而不是继续剩余29条或挑v2。已写[最多366物理步的纯动作/带几何短程对照提案](libero-postgrasp-replay-determinism-proposal-2026-11-16.md)，仅提案，未获新增授权、未绑定、未运行。任何后续新仿真都需另冻范围和预算；本轮不自动重试。
