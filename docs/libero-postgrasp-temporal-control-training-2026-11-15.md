# task7抓起后控制对照：真实训练完成（2026-11-15）

**两臂、3个seed、6个新模型均完成500更新，共3000次真实Adam更新；独立训练审计3972项核对通过。** 内部耗时132.104秒，使用CPU单线程，未使用CUDA训练。这里没有新的搬运/放置成功率结论；后续交接评估需GPU1空闲并经审计后报告。

| 步骤 | 目的 | 配置 | 实际结果 |
| --- | --- | --- | --- |
| 1：既有能力核对 | 避免误把逐帧模块当成已有记忆 | 只读源码、现协议和上轮报告 | 确认实际cnn_only为MLP；recurrent只是接口名字。最初口头误读为GRU已在研究输入读取前更正 |
| 2：限定新候选 | 比较后段控制是否受益于短历史 | 同冻结ResNet视觉、逐帧MLP对10帧GRU，其他非记忆初始化、batch/优化/loss匹配 | 实际参数构造：总12689864/12689863，可训练1513352/1513351，仅差1参数；估算512差值已更正 |
| 3：实现与合成验证 | 确认访问、因果窗口与停止规则 | 人造模型3000次Adam；44交接/120闭环替身；原策略/数据/初态回归 | CPU版86项通过；原初21和扩展85也通过；真实研究payload未用于测试 |
| 4：读取前绑定 | 锁定科学配置、角色、预算和代码 | 706项训练依赖，stage-specific授权，原封存不变 | 绑定通过后才读取一个task7 source文件与公共权重 |
| 5：真实训练与审计 | 获得可比较的新500终点 | task7 meta_support demos0..29，t≥1，batch8/sequence10/stride10，Adam3e−4，原七维MSE/clip1 | 432窗口、2 opens、2160切片、3000前向/反向/优化；6checkpoint及其实际Adam step均500；冻结权重/BN、非记忆初始化与批次配对通过 |
| 6：后半程交接 | 比较相同抓起状态下能否完成 | 11对源轨迹的22状态×两控制器，共44条；零新训练 | 已建立独立绑定，空闲GPU1准入后执行，结果未在本训练报告中宣称 |
| 7：条件性完整闭环 | 检查从头运行是否也有改善 | 只有交接净增≥4/22且各seed不退化，才评估120条/60对 | 未过前关不运行；完整门为净增≥6/60且各seed不退化 |

## 训练拟合读出

下面是训练batch loss，不能代替任务成败。两臂三seed均明显下降；GRU末50步均值在06/07较低，在08略高，不能据此选择seed或checkpoint。

| seed | 控制器 | 实际更新 | 前50均值 | 后50均值 | 梯度裁剪次数 |
| --- | --- | ---: | ---: | ---: | ---: |
| 20261106 | frame_mlp | 500 | 0.079578 | 0.024855 | 32 |
| 20261106 | gru_window10 | 500 | 0.080923 | 0.023400 | 55 |
| 20261107 | frame_mlp | 500 | 0.076694 | 0.027368 | 51 |
| 20261107 | gru_window10 | 500 | 0.080056 | 0.023110 | 60 |
| 20261108 | frame_mlp | 500 | 0.078549 | 0.028597 | 43 |
| 20261108 | gru_window10 | 500 | 0.088480 | 0.029177 | 69 |

CPU训练设备选择在真实payload读取前锁定：GPU1已有其他用户任务，未借用或停止他人进程。闭环保持当前GPU1/EGL1栈与严格reset/前缀摘要门，资源等待不改初态或动作。队列最多等待3600秒/阶段；等待不产生研究环境步，真正运行后任何技术失败立即停止、不自动重试。

训练实际7个模型构造（含1缓存模型）、120次冻结骨干计算、6次像素/缓存核对、1次源容器SHA、公共权重SHA/读取各1；初态与物理环境访问0。原缓存424954080字节，最终卷积缓存283115520字节，host进程peak RSS 1631248384字节。CPU/BLAS/Torch各1线程，确定性开启；完整版本和预算见配置。独立审计读取6个新checkpoint、实际Adam矩和step、重建初始化与采样，不重训、不读HDF5/初态或运行环境。

[事前协议](libero-postgrasp-temporal-control-protocol-2026-11-15.md)、[科学配置](../configs/libero-postgrasp-temporal-control_v1.json)、[训练绑定](../configs/libero-postgrasp-temporal-control-training-binding_v1.json)、[交接绑定](../configs/libero-postgrasp-temporal-control-handoff-binding_v1.json)、[紧凑训练证据](evidence/libero-postgrasp-temporal-training-summary-2026-11-15.json)。原始训练、optimizer states和独立审计在`results/libero-postgrasp-temporal-control-20261115-v1/train/`，测试/运行/队列在`logs/libero-postgrasp-temporal-control-20261115-v1/`。

本次不是重启已停止confirm5，也不是GRU有效性结论。Policy500、稳定化5、原loss和query/meta_query/validation/H/heldout/qualification/reserved/final封存保持。LIBERO继续，不开展Continual World，不更新SPT/Knowledge或注册Module；所有后半程与完整闭环结果须按各自实际scope单独解释。

生成UTC：2026-11-15T08:33:11.527720+00:00。
