# task1 B同一候选头双输入专家目标误差冻结记录（2026-11-04）

执行状态更新：用户随后授权的6批读出和独立审计现已完成，见[实际结果](libero-task1-b-fixed-head-expert-target-error-results-2026-11-04.md)。整体真实历史更接近专家目标；重复当前平移MSE在三个seed均增加，均值+30.74%，已知open段保留反向收益。以下保留执行前冻结阶段快照，其中“未执行”仅描述当时状态。

**配置、指标、预算、执行入口与代码哈希已冻结，metadata预检通过；本轮未执行研究读出，没有新的目标MSE结论。** 原始缓存数值解码、编码器/模型研究前向、训练、环境及初态读取均0；合成测试调用不计为研究读出。

## 为什么做、用什么比较

[上一轮闭环](libero-task1-b-fixed-head-input-closed-loop-results-2026-11-04.md)真实历史0/60、重复当前4/60，收益仅seed24；60对轨迹均分叉，尚不能说明哪种输出更接近专家动作。本项将权重、专家观察与目标固定，只改变运动6维的输入内容，直接比较目标误差。

仅task1 B的seed20261022/23/24，固定与闭环完全相同的三个完整support候选500头及scaler。两臂均用候选权重：真实历史[t,t-1,t-2,t-3,t-4]或重复当前[t,t,t,t,t]；夹爪始终当前，当前夹爪读出复用。没有baseline头、重训、scaler重拟合或checkpoint选择。

已核对缓存来源JSON和字节hash：source meta-query demos30..49，2598个已有causal pre-action帧；固定全部2518个t>=5完整历史帧，三seed共7554配对。每seed是同一批专家场景，不视为7554个独立未见样本。示教编号与official初态编号不同，本项不打开任何初态容器，也不读取validation、held-out、qualification、reserved或final evaluation。

原500端点、闭环5步稳定化、现有loss保持。离线专家历史长度5并不等于执行5个稳定化动作；本项不运行环境、不对专家观察补零动作或改target。较低目标MSE只表示更接近示教动作，不能直接推出控制可靠性或环境机制。

## 已冻结的指标与预算

| 项目 | 冻结值 |
|---|---|
| 主指标 | 每seed全2518帧的平移MSE差，重复当前减真实历史；负为改善 |
| 主判断 | 三seed均严格降低才记一致改善；均值不能覆盖反向seed |
| 次指标 | 旋转、运动6维、夹爪、七维MSE；逐帧差及逐demo/等权demo汇总 |
| 联合描述flag | 主判断通过，且三seed旋转/七维均不增加、夹爪差严格0 |
| 误差分解 | DeltaMSE=2mean((real-target)delta)+mean(delta²)，delta=repeat-real |
| 固定专家分层 | all2518、open事件23、已知open段232、初始open974、expert close1312 |
| 新重复当前矩阵读出 | 3批/7554行 |
| 历史运动矩阵重放 | 3批/7554行 |
| 总算术 | 6批/15108行，夹爪复用，无隐藏第三路 |
| 数值解码归档 | 目标1、feature3、候选头3、已有真实历史预测3 |
| 新CNN/训练/环境/HDF5/RGB/PT参数/初态 | 全0 |
| 运行设备 | CPU、单线程、NumPy1.22.4，CUDA隐藏 |

完整规则见[协议](libero-task1-b-fixed-head-expert-target-error-protocol-2026-11-04.md)与[配置](../configs/libero-task1-b-fixed-head-expert-target-error-diagnostic_v1.json)。既有真实历史预测须通过1e-5桥接及严格夹爪符号复现，目标误差恒等式容差1e-12。失败保留并使主结论无效，不补样本或放宽门槛。所有输入/代码hash和实际attempted/completed均记录；夹爪零差是合同负对照，不作为夹爪学习收益。

## 实现、验证与provenance

新增[执行入口](../experiments/run_libero_task1_b_fixed_head_expert_target_error.py)复用已有causal帧计划、输入标准化和序列化，不修改历史实现。它固定候选完整support端点、检查与上轮闭环权重身份一致，输出两个输入下的动作/logit、逐元素误差方向/能量/差、固定分层和每demo统计；不运行CNN。

新增[专项测试](../tests/test_libero_task1_b_fixed_head_expert_target_error.py)与相关旧交叉、闭环审计、第5步及路由测试共77项通过（1.71s）。覆盖相同响应能量下改善/恶化两个方向、跨seed反证与旋转取舍、夹爪精确复用、因果索引/跨demo/padding拒收、预算attempted/completed、错误角色、metadata零解码及输出保护。两份Python编译、CPU环境检查通过；未重复全量pytest，可选Crafter不在此次验证范围。环境工具在CUDA隐藏时报告available=false指CUDA不可用，Torch本身正常。

| 对象 | SHA-256 |
|---|---|
| 配置 | 873ba3cb325daa6b3bb71f5c98afa8108a847cb1be5d79816b62391b6a430d2c |
| [代码绑定](../configs/libero-task1-b-fixed-head-expert-target-error-binding_v1.json) | 1ff094a0c9e2487bc7b91016e5931e50ad987ec9e3bb83b51951644da0538714 |
| 执行入口 | eeca3d1b1c9280cfeb983b6364d6c59f0dba21211988c876980cebbcd443c5cb |
| 测试 | 967efc5a3d3e3dda62fe0ab7c1d00e9832883da410cd30a60b463f9325b444fc |
| 协议 | 7146cb11511147673617727d9ea07468f64fd500e3b3ac17928bdecc9bd59574 |

本轮默认预检核对25个输入pin及3份执行文件绑定、上轮闭环与读出fit独立审计，没有调用np.load解码真实NPZ。开始快照为`/tmp/libero-task1-b-fixed-head-expert-target-error-start-20261104-v1.json`，保护1206份既有文件；仅README、roadmap和当前下一步计划追加，用户AGENTS.md/.vscode及无关改动保持。HEAD为e525d19cb45a40cc6df421b6c33b96445500c2b0，不提交或复制权重进仓库。

最终检查记录为`/tmp/libero-task1-b-fixed-head-expert-target-error-freeze-verification-20261104-v1.json`，包含最新metadata预检、25个来源与绑定hash、测试和编译凭据、预算独立核算、文档与文件保全。它不是研究读出result。

未来执行命令如下；本轮只运行了不带--execute的预检：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
conda run -n libero-kecrl python -m experiments.run_libero_task1_b_fixed_head_expert_target_error --execute
```

未来结果`/tmp/libero-task1-b-fixed-head-expert-target-error-20261104-v1.json`及同前缀artifacts目录当前不存在，排他创建；不预写空研究结果。所有原始数据、checkpoint和旧结果保持原位。

## 当前结论和下一步

目前只确认对照已具体冻结且入口通过验证，尚不能判断真实历史或重复当前哪一种更准确。下一步按6批冻结预算完成全部三seed读出，独立重建预测/误差分解及旧预测桥接，报告逐seed、逐demo及预定分层的支持和反证，再决定训练修改。

无论离线方向如何，本项不更新SPT/Knowledge、不注册Module，不把4条闭环成功或较低MSE当作正式结果；保持500端点、稳定化5步、现有loss、旧候选停止和正式封存角色。
