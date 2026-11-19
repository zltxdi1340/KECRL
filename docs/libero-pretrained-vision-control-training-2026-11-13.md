# LIBERO新视觉后端：真实训练与独立核对（2026-11-13）

**12个完整Policy各500更新已经完成，合计6000次实际Adam更新；独立训练核对通过。** 这是source-support训练结果，目前没有新闭环成功率，不能据此宣称控制效果已经提高。

按用户明确要求继续LIBERO，[当前开发主线](libero-continue-control-development-2026-11-13.md)与[本轮科学协议](libero-pretrained-vision-control-protocol-2026-11-13.md)采用一个新完整视觉后端候选。此前将旧配置收束扩大为转Continual World的安排撤回；旧G及局部候选负结果继续按各自范围保留。

## 实际做了什么

| 项目 | 实际完成 |
| --- | --- |
| 范围 | source task4/7，seeds20261106/07/08，两臂，共12个Policy |
| 控制 | 从头训练的原CNN-only完整七维控制器，6个500终点 |
| 候选 | 冻结ImageNet ResNet18卷积/BN＋可训练空间投影及原fusion/MLP/七维头，6个500终点 |
| 更新 | 6000前向、6000 backward、6000 Adam更新；各可训练参数的Adam step均为500 |
| 配对 | 六组非视觉初始参数相同，500批次身份相同；未选择checkpoint或有利seed |
| 冻结核对 | 6/6候选的卷积权重、BN统计及默认零计数保持不变，与固定公共权重逐项对应 |
| 数据 | task4/7各自meta_support demos0..29、causal t≥1；458/432窗口，共890 |
| HDF5 | 4次打开、4450次数组切片，尝试/完成相同；源文件完整SHA核对2次 |
| 闭环/初态/query | 均未执行或数值读取；正式封存角色保持 |

原MSE、500 updates、未来稳定化5步及完整七维接口保持。骨干特征缓存后供500步投影/控制器训练复用；六个候选各有一次原像素与缓存动作的预定数值核对，全部通过。冻结骨干调用及实际细账见[紧凑真实摘要](evidence/libero-pretrained-vision-control-training-summary-2026-11-13.json)。完整文件SHA会覆盖HDF5中未解码字节，不声称物理隔离query字节。

两种后端的总参数量、可训练容量和外部学习数据不同；候选可训练1513352参数、冻结11176512参数，控制可训练821704。该实验比较实际完整后端，不证明预训练单因素的因果收益，更不等于FOMAML迁移收益。

## 核验与成本

先冻结新科学配置、556项执行依赖和合成验证，再读真实source输入；旧冻结和旧SHA未改。[科学配置](../configs/libero-pretrained-vision-control-diagnostic_v1.json)SHA为`d2a39cb0e1f18f817b582d2013ec9a68a049ee90c78aea09a2d3b9657b634a92`；[训练执行绑定](../configs/libero-pretrained-vision-control-training-binding_v1.json)SHA为`dbfcded732ffc6f68f80317b0129005512c759465bc5f93addcbfd39637b2e73`，其中source初态授权为空。

主阶段内部计时37.590706秒，包含输入核对、缓存及训练，不是预训练成本、完整进程/人工成本或闭环耗时。冻结卷积复用特征使大量重复卷积不必在每个Adam步重做；不能用短耗时断言训练未执行或已经充分收敛。Torch CUDA allocator峰值363711488字节，不是整机VRAM或host RSS。

独立核对器读取新checkpoint、全部训练轨迹和固定公共权重，重建参数哈希、采样摘要、身份、500步记录、实际Adam配置、冻结骨干/BN及预算。没有独立重训6000步或重新读HDF5；核对结果单独保存为`training-audit.json`，不回写主结果的pending-audit状态。

60项定向测试通过，包含发明的12模型6000次真实Adam更新及240个模拟槽位工作流；物理环境和训练审计在该合成流程中有明确替身，不当作真实研究成绩。公共权重接口另用发明RGB核对。准备中修正了一项合成配置漏字段，并处理官方旧权重未序列化BN计数的兼容规则；均发生于研究执行前，初始失败日志保留。CPU环境检查通过，主动隐藏CUDA；未跑全量pytest。

## 证据与下一步

- 真实产物：`results/libero-pretrained-vision-control-20261113-v1/`，含12份checkpoint、完整事件、`training.json`及`training-audit.json`。
- 执行/测试/环境日志：`logs/libero-retain-environment-20261113-v1/`。
- 可归档元数据：[真实训练摘要](evidence/libero-pretrained-vision-control-training-summary-2026-11-13.json)。模型、缓存、数据和详细轨迹保留个人数据盘。

下一步按冻结范围执行240条fresh配对闭环，以原生成功率和task/seed护栏判断是否改善；不能用训练loss代替这一阶段。此前用户要求初态封存，因此已单独询问仅开放这两个source任务已反复使用的candidate_nonregression 0..19；当前未获回答，入口在任何初态文件读取前拒绝未授权评估。其余source meta_query、validation/H/qualification/reserved/final保持封存。

通过开发门才扩到全部source并准备同后端A/B/C；若候选未通过，保留结果并继续在LIBERO内提出有依据的下一改进，不退出LIBERO，不转Continual World，不更新活动SPT/Knowledge或注册Module。
