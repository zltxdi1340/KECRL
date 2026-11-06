# LIBERO G阶段完成复核与准入结果（2026-11-06）

后续只读诊断已完成：[任务/seed差异与全量分层](libero-formal-G-task-seed-differences-2026-11-06.md)覆盖全部20个seed、180行配对和2200个原有指标。该事后诊断不改变本页的G准入未通过结果；下一步仅建议读取已有公开轨迹，仍不追加训练或解封H/qualification。

**G完整执行并通过独立审计；冻结的准入规则判定未通过，当前配置结束，H主检验未开展。** 完成时间为北京时间2026-11-05 23:10:01。此次不是仍在排队或训练未跑完：全部320个Policy各完成500次适应，6400条闭环齐全。此前执行v1的审计字段错误已通过版本化修复；本次执行v2没有异常中止记录。

本报告遵循[科学协议v2](libero-formal-transfer-protocol-v2-2026-11-05.md)、[执行绑定v2](../configs/libero-formal-G-execution-binding_v2.json)和[修复续执行契约](libero-formal-G-execution-continuation-v2-2026-11-05.md)。科学版本、执行版本与早期diagnostic分开记录；本结果是预注册G筛选结果，不是held-out确认性主检验、Knowledge Evolution或Module qualification。

## 做了什么与核验结果

本次只读检查完成收据、后台进程和原始结果；分别重新运行既有独立训练审计及完整闭环审计，均通过。再次从6400条episode描述逐task/seed/arm重算320行成功数，以Fraction精确重算均值和严格胜出，结果与完成收据及冻结决定函数逐项一致。实际负G收据请求H准入被拒绝，未读取H数据或启动H训练。

原始训练审计验证1660份checkpoint、1000次来源配对更新、160000次适应更新和2200份dataset sweep；完整审计验证6400份episode、1731927行公开轨迹和1725527个Policy动作。140个公开reset reference及6260次逐位比较通过。所有attempted/completed计数相等，预算未超限。

本次复查的新增Policy前向、optimizer、环境、初态数值解码和H payload访问均0。CPU权重/公开轨迹/配置/hash读取及输出新的复核摘要单独留痕，不覆盖原结果、checkpoint、事件日志或历史审计。

## validation闭环成绩：同任务、同seed、同初态配对

A为无source反馈更新的原始生成器，B为上下文FOMAML候选，C为同容量/同source批次的pooled BC；三臂目标SPI均适应500次。每task每臂20个seed×20初态=400条，两个task与seed等权。

| validation任务 | A：原始 | B：FOMAML | C：pooled BC |
|---|---:|---:|---:|
| task2：桌面中央 | 118/400（29.50%） | 113/400（28.25%） | 140/400（35.00%） |
| task6：饼干盒旁 | 51/400（12.75%） | 50/400（12.50%） | 23/400（5.75%） |
| 两任务等权均值 | 21.125% | 20.375% | 20.375% |

| 冻结条件 | 实际值 | 判定 |
|---|---:|---|
| B平均成功率≥20% | 20.375% | 通过 |
| B−A均值≥+5个百分点 | −0.75个百分点 | 未通过 |
| B−A严格胜出seed≥16/20 | 11胜／2平／7负 | 未通过 |
| B−C均值≥+5个百分点 | 0.00个百分点 | 未通过 |
| B−C严格胜出seed≥16/20 | 10胜／0平／10负 | 未通过 |

B相对C在task2低6.75个百分点、task6高6.75个百分点，等权均值恰好抵消；B相对A在两任务分别低1.25和0.25个百分点。当前证据未形成候选相对两个对照的稳定平均收益。task6相对C的局部收益保留，但不能覆盖task2的反向结果或改变固定门槛。

G按预设阈值作准入决定，不在validation上补做H的Holm确认性检验，也不按闭环结果选择seed、checkpoint或预算。20.375%是固定validation集合的表现，不是对任意新任务的成功率保证。

## source非退化护栏

每task每臂也是400条，仅A/B；source的0..19已被历史诊断使用，解释为既有来源bank的非退化检查。C未在这部分闭环运行，不补0或合并成三臂总分。

| source task | A | B | 均值B−A（百分点） | 最差单seedB−A（百分点） | 低于−10个百分点的配对 |
|---|---:|---:|---:|---:|---:|
| 0 | 89/400（22.25%） | 106/400（26.50%） | +4.25 | -45.00 | 4/20 |
| 1 | 55/400（13.75%） | 52/400（13.00%） | -0.75 | -55.00 | 3/20 |
| 3 | 102/400（25.50%） | 99/400（24.75%） | -0.75 | -55.00 | 7/20 |
| 4 | 16/400（4.00%） | 24/400（6.00%） | +2.00 | -25.00 | 1/20 |
| 7 | 13/400（3.25%） | 17/400（4.25%） | +1.00 | -25.00 | 2/20 |

五个task的跨seed平均差均满足≥−5个百分点；每个task都至少有一个seed不满足单seed差≥−10个百分点。合计17/100个task×seed配对违反护栏，这不是17个独立seed；最差配对的差为−55个百分点。平均未退化不能代替逐seed护栏，故来源准入条件也未全部满足。全部17个配对列入CSV，未删除失败seed。

## 已保存的离线曲线：仅作次要读出

五个固定checkpoint均保留，以下是500端点selection_query七维action MSE的20seed均值，仅causal t≥1：

| validation task | A | B | C |
|---|---:|---:|---:|
| 2 | 0.042374 | 0.041970 | 0.041374 |
| 6 | 0.047336 | 0.046458 | 0.046943 |

两任务B的query MSE均值略低于A，但对应闭环成功率均值仍略低于A。这些汇总没有形成同一排序，不能用离线MSE替换闭环主指标、恢复已失败门槛或直接归因于视觉/动作头/夹爪。全部0/50/150/300/500的support/query曲线另存CSV；source meta_query属于来源更新反馈，不当独立source评价。

## 预算、时间与访问边界

| 项目 | 实际完成 | 冻结上限 |
|---|---:|---:|
| 适应Policy／updates | 320／160000 | 320／160000 |
| 生成器Adam／临时SGD | 2000／5000 | 2000／5000 |
| 优化更新总数 | 167000 | 167000 |
| 离线评价batch／dataset sweep | 25200／2200 | 25200／2200 |
| 唯一checkpoint文件 | 1660 | 1660 |
| 闭环episode | 6400 | 6400 |
| Policy动作 | 1725527 | 1920000 |
| 稳定化零动作 | 32000 | 32000 |
| native step | 1757527 | 1952000 |
| adapter／reset | 320／6400 | 320／6400 |
| 初态数值行 | 140 | 140 |

累计活动时间29017.178秒，约8小时3分37秒，含继承的失败段与修正审计，低于16小时上限。source训练167.581秒；目标适应含离线评价1392.747秒（约23分13秒），其中评价251.972秒，不能将包含关系重复相加。峰值host tensor cache3174367630字节、CUDA allocated527021568字节。GPU1执行，GPU0未占用；此前v1的全部20组source权重直接引用，0重复source训练。

320份适应后的模型与source/validation在线角色分开。source官方初态只数值解码0..19共100行，validation只解码10..29共40行；qualification、reserved、held-out未数值读取。validation的adaptation_support0..29与selection_query30..49按任务完成条件访问；源示教与官方初态编号互不配对。原始终止记录1068次success、5332次time_limit用于完整性记账，不据此计算跨角色总成功率。

## 证据文件

| 证据 | 路径 | SHA-256 |
|---|---|---|
| 完成收据 | `/tmp/libero-formal-G-artifacts-20261105-v2/completion.json` | `04f62f19fbe92f6d657c1f8393d775c3dfe03d54eb6b80aad4bc79e44cea5ad1` |
| 原始G结果 | `/tmp/libero-formal-G-result-20261105-v2.json` | `0e549dbebfafd8df36e3cea9bd0c0d2a64b9102965c8d2d43a07a27f060fdf99` |
| 本次训练复审 | `/tmp/libero-formal-G-training-audit-check-20261106-v1.json` | `3375a5d5a53b3d90a6b5f4067f633e6710ed55d774aa64f9bcb948dc98ca679c` |
| 本次完整复审 | `/tmp/libero-formal-G-complete-audit-check-20261106-v1.json` | `8282b4db8dc08502925b3d8dc682aa958a2424ff13c7324e8f80e44dce8e7743` |
| 只读重算摘要 | `/tmp/libero-formal-G-results-summary-20261106-v1.json` | `21aeeaeb89a76bc583731d9e1f69b609274071195ed67dfc4eafd9083a631d6f` |

原始独立审计为`/tmp/libero-formal-G-artifacts-20261105-v2/training-audit.json`和`complete-audit.json`，修正source审计为`/tmp/libero-formal-G-source-audit-corrected-20261105-v2.json`。实时状态最终为`complete`，进程已退出，GPU1无遗留本作业计算进程。

| 导出 | 路径 | 行数 | SHA-256 |
|---|---|---:|---|
| per_task | `/tmp/libero-formal-G-task-results-20261106-v1.csv` | 16 | `f8f3aff7618a9afc5e2993a442b4d8f0f57d4db912947fe85dd7f0e14cf6df13` |
| per_model | `/tmp/libero-formal-G-model-results-20261106-v1.csv` | 320 | `2ae622c048e163cbce9c25b9a06c133dd9dbee2a864fdbc982f0d1d6af5270ef` |
| validation_seed_comparisons | `/tmp/libero-formal-G-validation-seed-comparisons-20261106-v1.csv` | 40 | `0cc3e970cac9c67d0406daccfb41af6eeec2e3282e4e1efbe3622260a59e51ce` |
| source_guard_violations | `/tmp/libero-formal-G-source-guard-violations-20261106-v1.csv` | 17 | `a8a6d26b19e80a5706f8c783f92a65b9ffe4323fd18f3698fe960f9092ad54bf` |
| offline_curves | `/tmp/libero-formal-G-offline-curves-20261106-v1.csv` | 110 | `037b9c77c9d79010f0272c27e3f4d374d6687bbc30a1a63c208e041efd486269` |

## 下一步与停止决定

按冻结规则结束本配置，H tasks5/8/9的适应、sealed_query和final初态继续封存，H确认性主检验未开展。不得把G未通过写成held-out负结果，不追加seed、预算或附近超参数，不接受活动SPT、不更新Knowledge、不注册Module。

下一步先归档本轮准入未通过的证据并整理可汇报的负结果；若继续机制研究，先只读已有task2/6的B−C取舍、17个source退化配对及五点曲线，区分任务/seed差异。此报告尚未证明唯一机制，不立即改loss、加updates或重启相邻训练。新训练需要独立科学协议与预算，H仍按原角色封存。
