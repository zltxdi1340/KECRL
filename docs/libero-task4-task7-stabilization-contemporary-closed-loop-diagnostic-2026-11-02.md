# LIBERO 当前执行环境下的 stabilization 0/5 闭环对照（2026-11-02）

本项直接回答用户“为什么不比较两种设置的闭环成功率”：固定六个500-update Policy，在同一当前 CUDA/EGL0 栈中，将两臂全部重新执行，比较 candidate 初态上的成功率。无需先完成 RGB/本体分解，不重训。

**执行状态：两臂240条fresh episode已全部完成并独立复核。0步与5步均为7/120=5.83%。** task4零步少成功1条，task7多成功1条；两个任务均无三seed一致的零步改善。保持现行5步与500 updates，不把原历史复现失败改判成功，不将本诊断升级为正式资格结果。

## 为什么独立于历史复现试验

[前序历史复现试验](libero-task4-task7-stabilization-closed-loop-diagnostic-2026-11-02.md)要求五步臂全部长轨迹逐位匹配历史，然后才运行零步臂。它在 task7/seed22/state6 第149步失败；回到原 device0 后仍失败，而两次当前轨迹彼此完整精确相同，最终均失败。因此换设备并未消除与历史的差异，也不能用旧 `7/120` 直接替代新基线。

前序 device1 与 device0 结果继续保持 **failed**，不改门槛、不改判通过、不纳入本项主分母。本项是另行冻结的**当前执行条件下两种 reset 流程比较**，两臂都重新采集，240条全部为新记录。历史完整轨迹一致性和67条已有当前参考的一致性逐条另报；它们不被称为原历史复现成功。既有20步短窗口接口契约继续是严格门槛，不是把窗口调到149步之前。

## 执行前冻结

- [配置](../configs/libero-task4-task7-stabilization-contemporary-closed-loop-diagnostic_v1.json)，SHA-256 `e3654bf195aa4ea8bdb5b71f74f8920526548c177473ad3afbf946e4cd52fc06`。
- [runner](../experiments/run_libero_task4_task7_stabilization_contemporary_closed_loop.py)，SHA-256 `55a7061794080c406f02e2072851e1364667b8bc6c4e6e06a81b67dbedf5890c`。
- task4/7；seeds `20261022/20261023/20261024`；六个既有 CNN-only baseline500，不选 checkpoint、不改变损失、数据或适应预算。
- 两臂 stabilization=5/0；环境seed20261022、同一物理CUDA/EGL0、相同native horizon305、每episode最多300个Policy动作；每 task×seed×arm 单独adapter，初态0..19递增。
- candidate_nonregression 0..19已被多轮使用，不能称 unseen。沿用只解码候选前20行的初态reader；不反序列化qualification/reserved数值，不创建示教dataset，不访问query/validation/held-out/final evaluation，不保存private simulator state。

## 门槛与实际比较问题

五步臂的reset与既有前20步action/公开状态必须精确匹配历史窗口；120条全部通过后才运行零步臂。同一arm/task/state在不同训练seed中的reset公开观察摘要也须精确一致。权重、初态角色、设备、顺序、horizon或预算不符均停止，完整拒收episode先保存。

历史长轨迹可能在窗口后出现微小分叉，逐条保留full-exact标记和终止结果，不隐藏、不借用历史成功率。主比较仅使用本项同一栈中fresh五步与fresh零步结果，按 task×seed×state 配对，报告两臂成功数/分母，以及 both_success/zero_only/five_only/both_failure。只有同任务三seed成功数均严格增加才记一致零步改善；不做显著性或独立最终测试推断。

完整240条结束后形成主比较。这个干预包括物理演化与固定零动作前缀，不是单纯替换图像；结果不自动选择新默认、不升级为正式SPI/SPT/Knowledge或Module qualification。

## 预算、来源与验证

本项新增240episodes、最多72,000Policy动作、600固定零动作、12adapters、40个候选初态数值读取；不复用任何主比较episode。新增optimizer/backward/HDF5 payload均为0。

前两次历史技术尝试累计74episodes、21,861Policy动作、370固定零动作。连同本项累计上限为314次episode尝试、93,861Policy动作、970固定零动作、120次候选初态数值读取（唯一初态仍为40个）。额外74条只作为历史复现/技术开销，不进入本项分母。另有一次已有帧CUDA前向预检，不执行环境。

device1失败结果 SHA-256 `077e546e63164f39a23ea7771190e1062a084cdf55e3c7469f67cfda46d94f84`；device0失败结果 SHA-256 `1bc2ee77c544d6bcb1460eb0273d56a73dd2c1395395e9d0857cd6d9f2248a5b`。当前分叉轨迹跨设备精确重复，但尚未定位与历史不同的具体数值来源。

启动前39项新协议与原组件测试通过，检验fresh分母、角色、20步精确边界、旧长轨迹失败不会被改判以及artifact保留。冻结时未预填成功率；实际完整结果和复核见后文。

代码完成后 CPU 回归为 `614 passed in 13.77s`，排除当前环境缺少可选 Crafter 依赖的 `tests/test_crafter_*.py`、`tests/test_spatial_gather_spt.py`、`tests/test_smoke.py`，不声称全量 pytest。新入口 py_compile 通过，独立审计脚本以内存 compile 检查语法，避免写入非本任务所有的共享 `/tmp/__pycache__`。

运行环境补充记录保存在 `/tmp/libero-stabilization-contemporary-runtime-supplement-20261102-v1.json`：Torch2.4.1+cu124、NumPy1.22.4、h5py3.11.0、MuJoCo3.2.3、robosuite1.4.0、gym0.25.2，旧协议中的 native code/BDDL hash 均仍匹配。当前尚不能将历史长轨迹分叉唯一归因于某个包或设备。

输出 `/tmp/libero-task4-task7-stabilization-contemporary-closed-loop-20261102-v1.json` 及独立artifacts目录。启动资源和退出信息另记；device0当前资源门槛 utilization<=50%、已占显存<=4GiB，torch/OMP/BLAS单线程、有界推理与渲染，无训练、不停止其他作业。

## 闭环结果与配对反证

两列均来自本次fresh执行，不使用历史成功率补分母。两臂均为同一组source-baseline Policy，不含jitter、局部loss候选或新训练。

| task | 5步：成功/episode | 0步：成功/episode | 零步减五步 |
|---|---:|---:|---:|
| 4 | 3/60 = 5.00% | 2/60 = 3.33% | -1条 |
| 7 | 4/60 = 6.67% | 5/60 = 8.33% | +1条 |
| 两任务合计 | 7/120 = 5.83% | 7/120 = 5.83% | 0条 |

| task | training seed | 5步成功/20 | 0步成功/20 |
|---|---:|---:|---:|
| 4 | 20261022 | 0 | 0 |
| 4 | 20261023 | 2 | 1 |
| 4 | 20261024 | 1 | 1 |
| 7 | 20261022 | 0 | 0 |
| 7 | 20261023 | 1 | 0 |
| 7 | 20261024 | 3 | 5 |

两臂共同成功4组、仅零步成功3组、仅五步成功3组、共同失败110组。总体计数相同掩盖了具体成功场景的改变；它不是统计等效检验，也不支持两种设置普适等价。

| task | seed | candidate state | 仅哪一臂成功 |
|---|---:|---:|---|
| 4 | 20261023 | 8 | 5步 |
| 4 | 20261024 | 5 | 0步 |
| 4 | 20261024 | 17 | 5步 |
| 7 | 20261023 | 3 | 5步 |
| 7 | 20261024 | 12 | 0步 |
| 7 | 20261024 | 14 | 0步 |

零步新增成功的三组都来自seed20261024；seed20261023在两个任务各丢失一组五步成功。任务内均未满足三seed严格增加规则，不能用task7某seed的5/20代替整体结论。每臂其余113条均以time_limit结束，没有异常终止。

本结果不支持“去掉五步即可提高当前整体闭环可靠性”的简单修复。前序source匹配诊断说明稳定化改变观察与模型响应，但并未证明这些变化会提升控制；本轮直接检验后，整体未改善。也不能把稳定化与所有失败完全无关作为结论，因为它确实改变了六组配对结果。两种设置均只有5.83%，主要控制能力缺口仍未解决。

## 历史与当前重复性单列

五步120条reset与原20步接口前缀均精确匹配，117条历史完整轨迹精确匹配。其余为task7/seed22/state6（首个分叉step149）、seed23/state2（step94）、seed23/state14（step262）；三条success、长度、终止类型仍与历史一致。已有67条当前参考均与本次五步完整轨迹精确一致；80个arm×task×state重置组在三个训练seed间摘要全部一致。

`runtime.actual_calls.five_episodes_exact=120`在本入口按冻结analysis定义指reset/20步前缀，不表示120条历史长轨迹均匹配；完整历史一致性另记`historical_full_exact=117`。旧device1/device0两个完整历史复现试验仍为failed，不因本次成功率相同而改判。以上不唯一确定历史数值分叉原因。

## 实际预算、来源和复核

本次运行北京时间 `11:33:14–11:54:21`，主程序elapsed `1267.2459s`（约21.12分钟）。实际240个fresh reset、12个adapter、70,188个Policy动作（5步臂35,160，0步臂35,028）、600固定零动作、70,788个显式环境step；40个candidate初态数值读取。没有optimizer或backward，没有HDF5 payload或封存初态数值读取。PyTorch峰值allocated显存 `13,490,176 bytes`（12.87MiB），该数不包含CUDA/EGL上下文或其他作业显存。

连同两个历史技术尝试，本轮实际累计314次episode尝试、92,049个Policy动作、970固定零动作，即93,019个显式环境step；candidate初态数值重复读取120次，唯一初态仍为40个。累计Policy动作低于预先增记的93,861上限。额外74条技术记录不在本项120/120分母内；单独一次已有帧CUDA预检只增加1次前向、0环境。没有任何新训练。

| 证据文件（均在`/tmp/`） | SHA-256 |
|---|---|
| `libero-task4-task7-stabilization-contemporary-closed-loop-20261102-v1.json` | `0f17ad53f86dce6aa7d38332297174702725bc5388092a4507a0214aadc1a49e` |
| `libero-stabilization-contemporary-independent-audit-20261102-v1.json` | `ac16e8b359bb6c79cb3fbf3660c6a037025cdf34679d852a2689ac625dc3bb4f` |
| `libero-stabilization-contemporary-runtime-supplement-20261102-v1.json` | `f0611fa94945490dc7d8c36d1c9a7973e87291241157a9424684aeb59b511905` |
| `libero-stabilization-contemporary-launch-20261102-v1.json` | `67bcaaccfbb6247977b0d5607478341a9fda8e6dd7f29bb6a9a164d7a0c37978` |

独立审计耗时 `3.3462s`，重新核验240个episode文件、70,428行公开状态、全部配对计数/seed方向、初态角色/文件hash、checkpoint来源、reset一致性、历史/当前重复性标记与实际预算；offset范数重算差为0，没有新的Policy、环境或初态数值读取。旧失败状态和fresh主分母的分离均已验证。

最终CPU回归仍为614项通过；运行后只更新报告与只读审计，没有改变已冻结的训练/执行代码。最终文件和工作区核验另记 `/tmp/libero-stabilization-contemporary-final-verification-20261102-v1.json`。

## 下一步

保持现行stabilization=5和500 updates。原计划的[120对轨迹只读对齐](libero-task4-task7-stabilization-trace-alignment-diagnostic-2026-11-02.md)现已完成：六组相反结果单列，未发现首动作、close时机或持续长度的统一成功方向。五步臂前缀中间状态没有记录，只从共同T=5开始对齐，不补造“前五步变化”。新增Policy/环境/训练均0，631项CPU回归通过；本报告的原运行与旧失败artifact不改写。下一项先冻结source既有缓存的RGB/本体响应分解，不把选中六组或公开状态差当专家误差、物理抓取或视觉根因证据，不提前进入qualification/SPT/Knowledge更新。
