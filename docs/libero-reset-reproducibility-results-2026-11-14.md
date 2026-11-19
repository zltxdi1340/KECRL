# LIBERO初态7的reset复现对照结果（2026-11-14）

**直接在新环境载入初态7时，机器人关节和夹爪数值与原记录相同，图像不同；在同一环境按0..7顺序reset时，8个初态的完整公开观测摘要全部与原记录一致。** 本轮9次reset完成，独立审计通过。支持先修正视频回放的环境使用顺序，再检验动作前缀；尚未判断合爪时机。

本轮按用户要求“下一步先核对环境reset能否复现，再继续夹爪时机分析”，执行[新冻结协议](libero-reset-reproducibility-protocol-2026-11-14.md)。读取前完成[科学配置](../configs/libero-reset-reproducibility-diagnostic_v1.json)、[本轮8行授权](../configs/libero-reset-reproducibility-source-initial-authorization_v1.json)与[执行绑定](../configs/libero-reset-reproducibility-execution-binding_v1.json)，绑定612项依赖。未续跑[失败的视频v1](libero-gripper-approach-video-execution-2026-11-14.md)，旧失败及其缺失日志原样保留。

## 已定位的差异

参照为task4、seed20261106、冻结臂原评估。每次观测均在原5步零动作稳定化后保存，比较原四字段组合SHA，零容差。

| 流程 | 关节/夹爪与原记录 | 完整公开观测与原记录 |
| --- | --- | --- |
| 新环境直接reset初态7 | 数值完全一致，最大绝对差0 | 不一致 |
| 另一环境内依次reset0..7 | 全部数值一致 | 8/8摘要一致，含初态7 |

两支初态7的joint/gripper字段字节摘要也相同；两个RGB字段摘要不同。因此本次复现出的不一致可定位到图像部分。两支新画面的agentview有4465个RGB通道值不同、最大差155/255；手眼相机有7880个通道值不同、最大差186/255。这里计的是通道值，不是完整像素数量；比较对象是本轮direct与sequential两份画面。sequential摘要与原记录匹配，但没有找回独立的原RGB文件。

[原尺寸画面对照页](../results/libero-reset-reproducibility-20261114-v1/reset-review.html)展示两路128×128原图，3倍显示不增加细节；4份PNG均回读核对与NPZ逐像素一致。视觉上相近也不改变零容差匹配结果。

## 为什么原流程会被拦截

原评估对每个模型复用同一个环境，按初态0..19顺序运行；视频v1却逐段新建环境，直接跳到选中的初态。该改变未验证是否保留原观测，是回放设计与合成验证的疏漏。本次对照复现了“直接reset7不匹配”，并且顺序reset恢复了原公开观测，因而该流程差异有实际证据支持。

reset包含环境及观测系统的重建/刷新；加载初态和设置seed并不自动证明全部输出已恢复。关节、夹爪公开数值相同，也不能证明图像相同或所有内部状态相同。本轮未读取物体位置、接触或其他私有状态，**尚未确认更底层的渲染/状态机制，也不能把顺序视为旧失败的唯一原因**。旧第三段没有保存失败的观测分项，不能用本轮文件补造其历史细节。本次两支采用同一个8行数组、global/local编号均为7，区别于旧视频的紧凑编号。

原来要用补录画面解释旧动作轨迹。若起始输入已经不同，继续拍摄便无法确认新视频与旧轨迹的对应关系，所以视频v1停止是必要的一致性检查。本轮协议把不匹配明确定义为待测结果，故按预定顺序完成两支9次reset，不是重试直到匹配。

## 实际预算与验证

| 项目 | 实际完成 |
| --- | ---: |
| 新环境 / reset | 2 / 9 |
| native step / 稳定化零动作 | 45 / 45 |
| 初态单次调用 / 数值行 | 1 / 8，仅task4的0..7 |
| 公开观察 / RGB帧 / RGB字节 | 9 / 18 / 884736 |
| 主采集原episode JSON | 8份 / 1714487字节 |
| 独立审计原episode JSON | 8份 / 1714487字节 |
| Policy前向、训练、checkpoint/HDF5读取、保存动作、额外render | 全部0 |
| 内部采集计时 | 9.485秒 |

独立审计以独立字节编码与Python标量重算9份记录，82项摘要/预算核对通过；核验行访问收据、来源、文件、顺序及global/local映射，不重新读取初态或运行物理环境。审计通过表示记录正确，不表示direct分支已匹配。完成收据状态为`complete_audited_reset_reproducibility_probe`。

104项定向测试通过：27项新增人造对照、异常留证/停止、篡改拒绝和绑定检查，加77项相关旧测试。编译、环境与差异检查通过；未跑全量pytest。代码、原始数据及各阶段证据存于：

- `experiments/libero_reset_reproducibility_protocol.py`：范围/授权与分项记录。
- `experiments/run_libero_reset_reproducibility_diagnostic.py`：主执行与异常留证。
- `experiments/audit_libero_reset_reproducibility_diagnostic.py`：独立审计。
- `tests/test_libero_reset_reproducibility_diagnostic.py`：仅人造数据测试。
- `results/libero-reset-reproducibility-20261114-v1/`：9份公开观测NPZ、真实初态访问收据、逐条JSONL、主结果、独立审计、completion及图像。
- `logs/libero-reset-reproducibility-20261114-v1/`：执行前快照、测试、环境、主/审计控制台和最终完整性检查。

## 后续行动与边界

下一版视频回放应保留环境使用顺序，先在相同环境中执行必要的前序reset，并逐项核验起点。先对失败案例初态7的两臂原动作1..100核验每步公开状态和终止信息，再决定是否扩展task7案例与视觉合爪判读。前序reset及初态读取要计入新版本预算和绑定，不能把旧v1剩余预算当作续跑许可。

**本轮尚未执行任何原保存动作，未验证100步前缀、未补充task7动作视频、未检验夹爪时机干预。** 顺序reset足以恢复本次公开起点，不能证明完整原运行历史或后续物理状态均已复现。Policy500、稳定化5步、现loss、正式数据封存及SPT/Knowledge/Module边界保持，LIBERO继续为主环境。

精简机器可读证据见[结果摘要](evidence/libero-reset-reproducibility-summary-2026-11-14.json)。612项依赖及执行前1813个工作区文件保留检查见`logs/libero-reset-reproducibility-20261114-v1/final-integrity.json`；项目状态入口仅追加本次进展，不回改历史冻结方案或结果。
