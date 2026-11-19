# LIBERO新视觉后端：240条配对闭环结果（2026-11-14）

**完整执行与独立审计通过。已用source开发场景上，CNN为8/120（6.67%），冻结ResNet候选为16/120（13.33%），净提升6.67个百分点。** 两个task均值、三个seed跨task均值均未退化，但平均增益未达到事前固定的10个百分点扩大门。存在实际闭环收益，仍不足以自动扩大到全source或进入SPT迁移比较；LIBERO继续作为主环境。

本次执行用户确认的task4/7已用source开发初态0..19，遵循[冻结科学协议](libero-pretrained-vision-control-protocol-2026-11-13.md)，没有新训练、选checkpoint或改变门槛。全部240条为当前同一执行栈的新episode。12个500终点来自[上一阶段真实训练](libero-pretrained-vision-control-training-2026-11-13.md)，CNN/ResNet是控制后端名称，不是FOMAML的A/B/C比较。

## 完整成功率

| source task | CNN对照 | 冻结ResNet候选 | 候选−对照 |
| --- | ---: | ---: | ---: |
| task4 | 0/60（0.00%） | 7/60（11.67%） | +11.67个百分点 |
| task7 | 8/60（13.33%） | 9/60（15.00%） | +1.67个百分点 |
| 两task等权 | 8/120（6.67%） | 16/120（13.33%） | +6.67个百分点 |

所有seed与两臂均保留，没有按有利任务、seed或初态删减：

| task | seed | CNN | ResNet | 净成功数差 |
| --- | ---: | ---: | ---: | ---: |
| 4 | 20261106 | 0/20 | 6/20 | +6 |
| 4 | 20261107 | 0/20 | 1/20 | +1 |
| 4 | 20261108 | 0/20 | 0/20 | 0 |
| 7 | 20261106 | 1/20 | 1/20 | 0 |
| 7 | 20261107 | 6/20 | 6/20 | 0 |
| 7 | 20261108 | 1/20 | 2/20 | +1 |

三个seed跨两个task的差分别为+15、+2.5、+2.5个百分点。净增加8次成功中，task4贡献7次，seed20261106贡献6次；不能概括为每个task×seed都严格改善。每个task只有20个不同初态，被三个训练seed和两个后端重复使用；60/arm不是60个独立未见场景，不作正式泛化或显著性推断。

## 配对场景：平均提高并不等于保留全部旧成功

每个task有60对相同seed/初态；总共120对，对应240条episode。

| task | 两臂均成功 | 仅CNN成功 | 仅ResNet成功 | 两臂均失败 |
| --- | ---: | ---: | ---: | ---: |
| 4 | 0 | 0 | 7 | 53 |
| 7 | 2 | 6 | 7 | 45 |
| 合计 | 2 | 6 | 14 | 98 |

候选新增14个独有成功，也丢失6个CNN独有成功，净增8。task7的seed20261107虽然都是6/20，但仅2对共同成功、各有4对独有成功，因此不能将相同成功数说成逐场景等价或无损替换。全部120对已导出，保留反向结果。

## 按原门判断

| 冻结条件 | 实际值 | 判定 |
| --- | --- | --- |
| 两task/三seed等权增益≥1/10 | 1/15，即+6.67个百分点 | 未通过 |
| 每task三seed均值不退化 | task4 +7/60，task7 +1/60 | 通过 |
| 每seed两task均值不退化 | +3/20、+1/40、+1/40 | 通过 |

独立核对器使用整数成功数和Fraction原样判门，`continue_to_all_source_development=false`。本轮结束该固定冻结骨干版本，不按结果放宽10个百分点门、追加样本或直接扩大SPT实验。它是有效执行且观察到有限收益的开发结果，不是技术失败，也不否定LIBERO环境。

该对照支持继续研究完整控制后端，而非将局部辅助头的未命中解释为环境无效。候选同时改变视觉骨干、外部预训练与可训练容量，不能唯一归因预训练或视觉瓶颈；最终成功率仍低，收益分布也不均。新后端不自动替换活动SPT，不产生Knowledge机制结论，不注册Module。

## 授权与实际预算

[本次用户授权](../configs/libero-pretrained-vision-control-source-initial-authorization_v1.json)与[评估执行绑定](../configs/libero-pretrained-vision-control-evaluation-binding_v1.json)在任何初态读取前建立。授权仅用于本次两个source任务的已用candidate_nonregression初态0..19，原训练绑定保持为空授权的历史状态，没有改写其字节。

| 工作 | 实际完成 / 冻结上限 |
| --- | ---: |
| adapters | 12 / 12 |
| reset / episode | 各240 / 240 |
| Policy前向 / 动作 | 各69009 / 72000 |
| 稳定化零动作 | 1200 / 1200 |
| 受控native step | 70209 / 73200 |
| 初态数值行 | 40 / 40：每task恰好0..19 |
| reset参照 / 严格比较 | 40 / 200 |
| 新训练前向 / backward / optimizer更新 | 各0 |
| 新HDF5数据读取 / source meta_query / 正式角色数值访问 | 均0 |

全部注册操作的attempted/completed相等，没有技术中止、失败重试或补样本。两个容器均使用选择性行reader，未执行pickle/torch初态反序列化、未数值解码其余30行，不保存私有初态向量或live simulator state。完整SHA仍会读取容器所有字节，不能将数值行隔离夸写成物理字节隔离。

评估阶段内部计时1219.305665秒（20分19.3秒），不包含上一阶段训练、预训练成本或最终CPU审计。北京时间2026-11-13 23:57:29启动，2026-11-14 00:17:48完成。Torch allocator峰值112289792字节，不是完整渲染VRAM或host RSS。实际使用物理GPU1/CUDA逻辑0、EGL1，单线程、确定性、TF32关闭；版本与冻结配置一致。

独立审计再次检查12个终点及训练轨迹、全部240个episode文件/SHA、模型与场景身份、reset配对、动作边界、原生终止记录、成功计数、实际预算和原继续门；不重训或重跑物理环境，不宣称从公开轨迹复算全部原生接触谓词。补充作用域核对确认两bank恰好40行、其余数值行未解码；另用Fraction重算任务/seed/总体差，与完成收据一致。

## 证据与接续

- [紧凑闭环摘要](evidence/libero-pretrained-vision-control-closed-loop-summary-2026-11-14.json)：原始文件pin、完整分层、配对四格、实际计数和原门判定。
- 真实产物：`results/libero-pretrained-vision-control-20261113-v1/evaluation.json`、`completion.json`及`episodes/`全部240条轨迹。主结果保持pending-audit写入状态，完成收据单独记录审计通过。
- 同目录`paired-outcomes.csv`（120对）、`per-model-success.csv`（12行）、`per-task-success.csv`（2行）、`per-seed-success.csv`（3行）由新输出派生，没有新模型/环境调用。
- 本轮日志：`logs/libero-pretrained-vision-evaluation-20261113-v1/`，含授权前快照、执行/审计控制台、范围及算术复核。

本轮没有改执行代码或重新运行pytest；沿用已绑定的60项软件验证，真实闭环和独立审计是本轮新增验证。授权绑定脚本首次因字典语法错误在执行前停止，修正后才创建授权/绑定；该准备错误未读取初态或消耗episode，未改写任何旧结果。

继续以LIBERO为主环境。下一步优先准备同架构、同初始权重的“冻结骨干 vs 受限视觉微调”单变量方案，测试能否进一步改善较弱seed/task7并保留原成功场景；这是待验证候选假设，不是本轮已证明的根因或效果。本次授权不自动扩展为下一试验预算，其他source任务、source meta_query、validation/H/qualification/reserved/final继续按原角色封存。
