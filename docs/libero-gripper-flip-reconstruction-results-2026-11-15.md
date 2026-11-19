# task7成败互换：26段动作画面重建与审阅（2026-11-15）

**全部13对已完成数值分析及26段原动作画面重建，复现与媒体独立审计通过。13条失败中，10条可见目标随夹爪离开炉面但未完成搬运/放置，1条可见带起后掉落，2条阶段仍不明确。** 这些是按成败互换选择的13对，不代表全部120对或全部失败的分布；视觉分类是单观察者非盲审阅，独立程序验证数据与证据完整性，没有独立证明视觉判断或根因。

## 这次回答了什么

[前一步只读结果](libero-gripper-flip-pair-analysis-results-2026-11-15.md)确认13对在首次暂缓前的动作和公开状态全部相同，首次改变只在夹爪维；同一步手指状态分岔，下一步机械臂命令分岔。首次实际合爪晚5—12步。因而confirm5影响了后续反馈过程，不能把结果只理解成把原轨迹的同一个动作推迟5步。

新画面显示：部分变化发生在抓起之后，不能把所有失败统一归为没接近目标或第一次合爪太早。对10条后续失败，只定位到搬运/放置阶段，尚不能细分对准、下降、目标姿态、释放或接触条件，更不能证明它们在完整失败母集中占多数。

flip06 / seed20261106 / 微调 / 初态17 / confirm5是清楚的掉落例：第85帧碗仍在夹爪附近，87—90帧可见脱离并落向炉边支撑区，第91步才首次出现负向开爪命令；第39—90步applied夹爪命令一直为正。这例不能解释成先发开爪才掉落。具体承托物、滑脱、抓取姿态或碰撞原因均未测定。[密集画面](../results/libero-gripper-flip-reconstruction-20261115-v1/inspection/flip06-confirm5-drop-detail.png)与[命令核对](../results/libero-gripper-flip-reconstruction-20261115-v1/inspection/drop-command-followup.json)。初审对落点的“炉外桌面”说明由密集帧修订为炉边支撑区，类别/计数未变；初审与v2记录均保留，v2为当前记录。

## 全部13条失败，未知保留

下表步号是引用的已审阅画面，不是精确首次抬起或接触事件时刻。每对另有一条源标签成功轨迹，全部26条均进入审阅文件。

| 案例 | seed末两位 / 模型 | 初态 | 失败条件 | 可见阶段 | 证据帧 |
| --- | --- | ---: | --- | --- | --- |
| flip00 | 06 / 微调 | 3 | unchanged | 可见带离炉面，搬运/放置未完成 | [75, 150, 300] |
| flip01 | 06 / 微调 | 5 | confirm5 | 可见带离炉面，搬运/放置未完成 | [150, 300] |
| flip02 | 06 / 微调 | 9 | unchanged | 阶段未知 | [40, 50, 55, 60, 75, 100, 150, 200, 250, 300] |
| flip03 | 06 / 微调 | 11 | unchanged | 可见带离炉面，搬运/放置未完成 | [75, 150, 300] |
| flip04 | 06 / 微调 | 13 | unchanged | 可见带离炉面，搬运/放置未完成 | [75, 150, 300] |
| flip05 | 06 / 微调 | 15 | unchanged | 可见带离炉面，搬运/放置未完成 | [100, 150, 300] |
| flip06 | 06 / 微调 | 17 | confirm5 | 可见带起后掉落 | [55, 75, 85, 87, 88, 89, 90, 91, 95, 100, 300] |
| flip07 | 07 / 冻结 | 10 | unchanged | 阶段未知 | [40, 50, 55, 60, 75, 100, 150, 200, 250, 300] |
| flip08 | 07 / 冻结 | 13 | confirm5 | 可见带离炉面，搬运/放置未完成 | [100, 150, 300] |
| flip09 | 07 / 冻结 | 14 | confirm5 | 可见带离炉面，搬运/放置未完成 | [150, 300] |
| flip10 | 07 / 冻结 | 18 | unchanged | 可见带离炉面，搬运/放置未完成 | [100, 150, 300] |
| flip11 | 07 / 微调 | 14 | confirm5 | 可见带离炉面，搬运/放置未完成 | [150, 200, 300] |
| flip12 | 08 / 冻结 | 5 | confirm5 | 可见带离炉面，搬运/放置未完成 | [100, 150, 300] |

flip02原命令的已审阅帧中碗留在炉面，机械臂后来空手朝盘子移动；不能区分未稳抓与短暂提起后落回。flip07原命令中目标长期在炉圈附近，后段遮挡；无法可靠判定抬起/接触与是否落回。因此两条保留未知，没有用手指关节值代填阶段。

## 实际执行与验证

[事前协议](libero-gripper-flip-reconstruction-protocol-2026-11-15.md)、[科学配置](../configs/libero-gripper-flip-reconstruction_v1.json)、[执行绑定](../configs/libero-gripper-flip-reconstruction-execution-binding_v1.json)、[范围授权记录](../configs/libero-gripper-flip-reconstruction-authorization_v1.json)。代码及父链685项依赖在新payload访问前绑定。全部原代码、配置、旧失败及SHA保持。

- task7全部13对，原命令与confirm5各13段；只执行各自保存的applied动作到各自原终止步，不调用Policy或重新计算命令。
- 26个独立环境，320次顺序reset，每次稳定化5步；1,600个零动作加6,230个原动作，共7,830个native steps，实际361.130秒。环境seed20261022，horizon300，双相机128×128，EGL GPU1，Torch/BLAS/OMP线程1。
- task7初态0..18共19行仅选择性解码一次，19..49未解码，完整access收据保留。新模型构造/权重读取/Policy前向/反向/优化/HDF5/额外render均0。
- 26/26段目标reset摘要及6,230个动作后的公开joint/gripper状态精确匹配，全部终止步、成败、terminal reason和reward sum匹配。未触发技术停止、跳例或重试。
- 主/审计各26份源JSON、5,545,650字节、155,984个公开标量；独立审计132,685项核对通过，不运行模型/环境或读取初态。
- 6,550个公开观察、13,100个相机帧、643,891,200字节保留RGB。26段视频共6,256帧，另含294个warmup观察；媒体审计核对全部帧数、格式及453个无损画面块的像素/方向。
- 重放相关61项定向测试通过（新增25项），此前只读分析46项通过。人造完整包络、授权越界、1 ULP状态差、reset画面差、额外原生调用和媒体篡改拒绝均验证；编译、环境、差异检查通过，未跑全量pytest。

当前视频是新重建的RGB，原运行后续RGB没有恢复。reset、动作、公开状态和终止一致仍不能证明所有私有物理状态完全一致，图像审阅不能代替物体姿态、接触或因果实验。未把这26段重放加入原240条成功率样本，原13/120→14/120及继续门false保持。

## 研究决策与证据入口

当前confirm5继续停止，不扫描新的延迟步数。后续应准备检验抓起后的搬运、对准与放置控制，短历史/时序控制器是待对照的候选；本次没有证明它一定优于当前模型。下一项须先冻结具体结构、单一改动、配对控制与训练/评估预算，再绑定执行，本轮未启动新训练。

[视频与双条件索引](../results/libero-gripper-flip-reconstruction-20261115-v1/review.html)、[当前26条审阅CSV](../results/libero-gripper-flip-reconstruction-20261115-v1/visual-annotations-v2.csv)、[执行/证据摘要](evidence/libero-gripper-flip-pair-summary-2026-11-15.json)。完整capture、initial-access、journal、independent-audit、completion、media-audit、visual-review-v2及review-completion均在`results/libero-gripper-flip-reconstruction-20261115-v1/`；本轮测试和运行日志位于同名`logs/`目录。只读阶段证据位于`results/libero-gripper-flip-pair-analysis-20261115-v1/`。

Policy500、稳定化5步、现loss及正式数据封存保持，LIBERO继续为主环境。不读取query/meta_query/validation/H/heldout/qualification/reserved/final，不更新SPT/Knowledge或注册Module。这是source开发轨迹诊断，非正式迁移或完整KECRL验证。

生成UTC：2026-11-15T05:31:13.121215+00:00。
