# LIBERO受限视觉微调：240条配对闭环结果（2026-11-14）

**训练、完整240条闭环和独立审计均完成；冻结臂16/120（13.33%），微调臂12/120（10.00%），总体下降3.33个百分点。原三个继续条件均未通过，结束本受限微调版本，不扩大到全source或进入新的Skill Evolution迁移比较。LIBERO继续作为主环境。**

本轮执行[冻结协议](libero-restricted-vision-finetune-protocol-2026-11-14.md)：同ResNet18架构、同全部初始tensor及批次，候选仅开放layer4五个卷积权重，全部BN/早期视觉阶段冻结。两臂各自重新训练500次并完成新episode；控制的16/120恰与上一轮冻结ResNet成功数一致，但本轮未复用旧终点或episode。训练与角色授权详见[真实训练记录](libero-restricted-vision-finetune-training-2026-11-14.md)。

## 全部任务与seed

| source task | 新冻结臂 | 新微调臂 | 微调−冻结 |
| --- | ---: | ---: | ---: |
| task4 | 7/60（11.67%） | 8/60（13.33%） | +1.67个百分点 |
| task7 | 9/60（15.00%） | 4/60（6.67%） | −8.33个百分点 |
| 等权合计 | 16/120（13.33%） | 12/120（10.00%） | −3.33个百分点 |

| task | seed | 冻结成功/20 | 微调成功/20 | 净成功数差 |
| --- | ---: | ---: | ---: | ---: |
| 4 | 20261106 | 6 | 2 | −4 |
| 4 | 20261107 | 1 | 0 | −1 |
| 4 | 20261108 | 0 | 6 | +6 |
| 7 | 20261106 | 1 | 3 | +2 |
| 7 | 20261107 | 6 | 1 | −5 |
| 7 | 20261108 | 2 | 0 | −2 |

三个seed跨task的冻结→微调合计依次为7→5/40、7→1/40、2→6/40，差为−5、−15、+10个百分点。task4/seed20261108的6次新增成功不能代表一致改善；task7只在seed20261106增加，另两个seed下降。全部任务、seed和场景均保留，未按有利结果筛选。

每task只有20个不同开发初态，被三个训练seed和两臂重复使用；60/arm不是60个独立未见场景。本轮是反复暴露的source开发对照，不作正式泛化或显著性结论。

## 配对场景与原继续门

| task | 两臂均成功 | 仅冻结成功 | 仅微调成功 | 两臂均失败 |
| --- | ---: | ---: | ---: | ---: |
| 4 | 0 | 7 | 8 | 45 |
| 7 | 1 | 8 | 3 | 48 |
| 合计120对 | 1 | 15 | 11 | 93 |

微调新增11个独有成功，也丢失15个冻结臂独有成功，净减少4次。它仅保留本轮冻结臂16次成功中的1次，保留率1/16（6.25%）；task4虽总数多1次，原7次冻结成功全部丢失、另有8次新成功。成功保留是事前声明的描述指标，没有事后把它添加为继续门。

| 冻结条件 | 实际值 | 判定 |
| --- | --- | --- |
| 等权增益≥1/10 | −1/30，即−3.33个百分点 | 未通过 |
| 每task三seed均值不退化 | task4 +1/60；task7 −1/12 | 未通过 |
| 每seed两task均值不退化 | −1/20、−3/20、+1/10 | 未通过 |

独立审计以整数/Fraction判门，`continue_to_all_source_development=false`。另一个仅使用新输出的标准库脚本独立重算全部12模型、120配对、分层与三门，结果一致，不新增模型/环境调用。

末50个相同身份训练批次的平均loss在6组task×seed中有5组更低，但闭环总体仍下降；这是训练minibatch读数，不能替代独立成功门，也不能据此识别具体失效机制。补充只读参数核验确认6个微调模型的五个指定卷积tensor均实际变化，6个冻结模型对应tensor均不变，排除了本次候选完全未更新卷积的情况。

## 授权与实际预算

用户本轮明确授权“建立本轮执行绑定后训练。授权后续240条配对闭环”。[新初态授权](../configs/libero-restricted-vision-finetune-source-initial-authorization_v1.json)绑定本科学配置；[训练执行绑定](../configs/libero-restricted-vision-finetune-training-binding_v1.json)在研究权重/HDF5访问前建立，[评估执行绑定](../configs/libero-restricted-vision-finetune-evaluation-binding_v1.json)在独立训练核对通过后、初态访问前建立。科学冻结中的未绑定字段保留为历史准备状态，没有改写。

| 工作 | 实际完成 / 冻结上限 |
| --- | ---: |
| 新Policy / 每Policy实际Adam步数 | 12 / 12；各500 / 500 |
| 训练前向 / backward / Adam更新 | 各6000 / 6000 |
| adapters / reset / episode | 12 / 12；240 / 240；240 / 240 |
| 闭环Policy前向 / 动作 | 各68592 / 72000 |
| 稳定化零动作 / 受控native step | 1200 / 1200；69792 / 73200 |
| 闭环prefix / layer4前向 | 各137184 / 144000 |
| 初态数值行 | 40 / 40：task4/7各0..19 |
| reset参照 / 严格比较 | 40 / 40；200 / 200 |
| 闭环阶段新训练/HDF5读取 | 均0 |

各注册操作attempted/completed相等，没有技术中止、重试、补样本或换设备。训练只解码source meta_support demos0..29、t≥1；闭环只解码本轮允许的40行，未数值解码或缓冲其余初态行，未使用pickle/torch初态整体反序列化。完整SHA读取全容器字节，不声称物理字节隔离。source meta_query、其他source任务、validation/H/qualification/reserved/final继续封存。

训练内部计时72.918719秒；闭环1276.462864秒（21分16.46秒），北京时间2026-11-14 13:02:58至13:24:14。计时不包含准备、公共预训练及独立核对。训练/闭环Torch allocator峰值分别389,347,840 / 137,806,848字节，不是完整host RSS或渲染VRAM。物理GPU1/CUDA逻辑0/EGL1、单线程、确定性和TF32关闭，运行版本与配置一致。

## 证据与后续

- [紧凑闭环摘要](evidence/libero-restricted-vision-finetune-closed-loop-summary-2026-11-14.json)：配置/执行绑定/授权/新结果pin、完整分层与配对、原门、实际计数及数值行访问记录。
- [训练摘要](evidence/libero-restricted-vision-finetune-training-summary-2026-11-14.json)：全部12终点、初始化/采样/不可训练状态摘要、训练计数、loss与裁剪次数。
- 原始新输出：`results/libero-restricted-vision-finetune-20261114-v1/`的`training.json`、`training-audit.json`、`evaluation.json`、`completion.json`及240个`episodes/`文件。主结果保留pending-audit写入状态，完成收据单独确认审计通过。
- 同目录导出`paired-outcomes.csv`（120行）、`per-model-success.csv`（12行）、`per-task-success.csv`（2行）、`per-seed-success.csv`（3行），均由完整新输出派生。
- `logs/libero-restricted-vision-finetune-execution-20261114-v1/`保留执行前快照、环境、全部控制台、真实卷积变化核验、独立算术脚本/收据及最终完整性核验。独立审计不重训或重放物理环境，不从公开轨迹反推全部原生接触谓词。

本轮没有修改执行代码或重跑pytest，沿用已绑定143项软件验证；真实训练、闭环和独立审计是新增验证。保持500更新、5步稳定化和现loss。结论限于本卷积掩码/优化配置，不证明视觉适配整体无效，不否定LIBERO，也不形成SPT/Knowledge或Module结果。

下一步优先在本轮已保存的公开轨迹中分析冻结独有、微调独有和共同失败的配对行为，明确可检验的控制差异后再冻结下一项LIBERO对照；本轮不追加训练、扩大source或启动正式迁移。
