# LIBERO既有几何与接触记录可用性核对（2026-11-06）

**现有记录不足以检验这36对反例中的close是否发生在接近物体之后。** G闭环逐步记录没有末端位置、物体位置、接触或RGB帧；历史source专家示教虽有末端位置，但不能与这些Policy轨迹拼接。本次只核对已有代码、冻结schema、完成审计和文件名清单，没有新读取原始轨迹、HDF5、图像、checkpoint、初态或私有状态，也没有Policy前向、训练及环境调用。

本项承接[36对close段分析](libero-formal-G-close-runs-late36-diagnostic-2026-11-06.md)。[G准入未通过及停止决定](libero-formal-G-results-2026-11-06.md)保持；500 updates、5步稳定化、现有loss不变。H未开展，held-out、qualification和reserved继续封存。以下是记录可用性诊断，不是几何机制结果、SPT更新、Knowledge Evolution或Module资格结果。

## 核对了什么，证据在哪里

| 记录或实现 | 已有信息 | 对当前36对几何问题的限制 |
|---|---|---|
| [LIBERO adapter](../src/environments/libero_adapter.py) | 两路RGB、7维joint、2维gripper作为Policy观测 | 公共接口没有末端、物体或接触字段；环境返回过图像不代表轨迹归档保存过图像 |
| [G rollout写出](../experiments/libero_formal_rollout.py) | 每步动作、joint/gripper、相对reset的偏移；reset摘要和终止结果 | 没有逐步RGB、末端/物体位置或接触；摘要不能还原图像；成功标志不是逐步抓取或接触记录 |
| [800对schema校验](../experiments/analyze_libero_formal_G_validation_BC_traces.py) | 严格比较episode和每行字段集合，额外字段也拒绝 | 此前1600条轨迹均通过该校验，36对是其固定子集；不能假定归档中还藏有几何字段 |
| [source专家覆盖审计](libero-conditional-coverage-audit-2026-10-24.md) | 历史报告记录示教有obs/ee_pos、obs/ee_ori | 仅专家末端绝对轨迹代理，不是目标相对位姿；不是同一Policy/初态的轨迹，没有合法一一对应关系 |
| [task4/7 RGB短窗采集](libero-task4-ordered-prefix-reproduction-diagnostic-2026-11-01.md) | 特定旧Policy和场景的RGB/action窗口 | 不属于本次task2/6、G seed及B/C端点，不能用来补当前窗口；本次未解码这些archive |
| [geometry-extension任务分层](libero-source-internal-geometry-extension-protocol-2026-11-05.md) | 任务的来源几何类别与覆盖/外推关系 | “geometry”是任务类别分层，不是每步末端—物体距离或接触测量 |

以AST读取冻结写出代码中的两个逐步字典，得到reset行5个字段和动作行8个字段；与严格schema常量逐项相同。动作行完整字段为：

```text
step, action, joint_state, gripper_state,
joint_offset_from_start, joint_offset_l2_from_start,
gripper_offset_from_start, gripper_offset_l2_from_start
```

核对了写出器、adapter、schema及科学协议的SHA，并用已完成的G审计和先前诊断核对阴性决定，未改变任何冻结文件。G归档目录仅做文件名/后缀统计：6405份JSON、1600份PT、1份JSONL；没有解码这些权重或逐条重新读取轨迹，也没有图像archive可直接复用。该清单只说明已归档文件类型，不声称原环境内部从未拥有其他观测。

可复核证据为`/tmp/libero-G-geometry-record-availability-audit-20261106-v1.json`，SHA-256：`d0a47e7367869580cbbc8cf6775d3134d9085fa11f6a391c1c2517b80923fe33`。其中保存源码与metadata哈希、完整schema、写出位置、可用性表和全部新增研究调用为0的记录。

## 现在能说什么，不能说什么

此前31对输家持续close至赢家开始close，说明大多数“赢家较晚close”不是单步脉冲计数假象。但现有数据仍无法区分夹空、尚未到达物体、接触后滑落或运输失败，不能用joint位移大小或命令符号直接填入这些物理阶段。

理论上，经过坐标系、机械臂模型、基座和工具标定验证的正向运动学可以提供末端位置估计；本次未执行该重建，且它仍无法补出缺失的物体位姿和接触。历史HDF5的simulator state也不是这些G Policy轨迹，不能解封私有状态来补写旧记录。

## 下一步

先另行设计来源任务candidate范围内的有限几何观测诊断，明确它是新的观测采集，不能追认为旧36对的几何真值。保持固定500端点和5步稳定化，先定义末端、物体和接触的测量来源、坐标系及动作前后时钟对齐；如引入评估专用私有量，需要明确独立记录边界，不能进入Policy输入、梯度、checkpoint选择或Knowledge更新。采集前须另冻任务/seed/初态、窗口、预算和代码绑定。

该方案尚未冻结或运行。本轮不追加validation试验，不用已知validation结果继续选择方法，不自动重放或启动训练。当前可交付的是“缺哪些证据”的明确清单，不能将几何假设认定为根因。

## 软件验证与归档范围

推送前重点测试115项通过；排除可选Crafter相关收集项的CPU回归1648项通过（40.61s）。命令明确排除`tests/*crafter*.py`、`test_spatial_gather_spt.py`及`test_smoke.py`，不声称运行包含可选Crafter的全量pytest。原始日志及命令保存在`/tmp/libero-geometry-records-push-test-results-20261106-v1.json`及其引用日志。

本次整理412个LIBERO代码、配置、测试和研究文档；194个Python文件可编译、110个JSON可解析、本地文档链接可解析。历史v1/v2及失败记录仍用于provenance追溯，保留。清理了969个未跟踪的Python/pytest缓存文件、14个空缓存目录，约11.64 MiB，不删除实验artifact、数据、checkpoint、日志或冻结协议；明细为`/tmp/libero-project-cache-cleanup-20261106-v1.json`。`AGENTS.md`、`.vscode/`和无关的既有修改保留在本地。
