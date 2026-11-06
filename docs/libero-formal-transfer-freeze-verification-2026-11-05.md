# LIBERO正式验证方案冻结核验（2026-11-05）

**科学设计与统计代码已冻结并通过metadata预检；正式训练和封存数据读取尚未开始。** 冻结不代表方法成立，也不代表执行入口、Module资格契约或正式结果已经完成。完整设置与接受/停止规则见[协议](libero-formal-transfer-protocol-2026-11-05.md)。

## 冻结内容

- 主指标：held-out tasks5/8/9的固定500端点闭环成功率；20个新训练seed作为独立比较单位，B−C及B−A同时确认。严格seed胜出概率单侧精确二项检验+Holm familywise α0.05，观测平均增益≥5个百分点，逐task平均差≥−5个百分点。概率检验与观测平均差门槛不是同一统计对象。
- G阶段先对source非退化和validation2/6做完整检查；B平均成功率≥20%、相对A/C均≥5个百分点、每比较至少16/20 seed胜出，source各任务均值退化≤5个百分点且任一seed≤10个百分点。任一失败停止，不读held-out。
- 全通过时上限：500个适应Policy，各500更新；适应Adam250000、生成器Adam2000、临时SGD5000，共257000个optimizer更新。G/H分别6400/5400条，共11800条闭环；稳定化5步，Policy动作上限3540000、固定零动作59000、native step上限3599000。
- 只计划使用source candidate0..19、validation candidate_acceptance10..29、held-out final_evaluation20..49。qualification、reserved和未用checkpoint-selection初态保持封存，qualification预算0、Module注册关闭。
- 10→20 seed的选择来自读取任何封存数据前的纯数学校准：假设严格seed胜出概率0.8，单项通过概率由0.37581升至0.80421；联合检验和实际效应门槛的功效更低，不能声称对5个百分点效应已有80%功效。

## 不可变绑定

| 文件 | SHA-256 |
|---|---|
| [科学配置](../configs/libero-formal-transfer-protocol_v1.json) | `d1b7e5170760f46902947c5084e7071735bd940b6ccdf6154732f4346f557ad9` |
| [冻结绑定](../configs/libero-formal-transfer-freeze-binding_v1.json) | `1e4758fb35266fb1e67065962b2f082f107af913b951946e529a58c46c22b3af` |
| [协议](libero-formal-transfer-protocol-2026-11-05.md) | `0bcc875cf2c706ac71065f40980f281efb543d3eee2a44613ddd9037d86adaba` |
| [metadata与统计代码](../experiments/libero_formal_transfer_protocol.py) | `59d9ba26e7cb9def75a08c5fc5f3a8aef14c8a1d599450aa7f1064bd9a8d1f31` |

当前绑定固定科学配置、协议、统计代码、测试和已有算法/数据/接口源码；`execution_binding_complete=false`，不得将它冒充未来正式runtime的完整执行绑定。今后执行入口必须符合本设计并另行绑定所有实际执行依赖、阶段门槛和访问账本。

## 验证证据

[新增测试](../tests/test_libero_formal_transfer_protocol.py)及原transfer split、causal数据集、FOMAML相关测试：78 passed in1.60s。覆盖元数据预检不导入训练/仿真依赖或打开研究payload，拒收角色/预算/端点/alpha/绑定漂移，缺失/重复/布尔计数，验证失败保持封存，技术失败不伪装为效应失败，平局保留分母、单seed大收益不替代一致性、显著但小于5个百分点拒收，以及宏平均改善掩盖单task退化时拒收。

CPU回归1410 passed in19.60s；日志`/tmp/libero-formal-transfer-freeze-cpu-regression-20261105-v1.log`。排除`tests/*crafter*.py`、`tests/test_spatial_gather_spt.py`和`tests/test_smoke.py`，不声称包含可选Crafter的全量pytest。合成训练/张量测试与研究调用分开，本轮未运行真实实验模型或读取真实研究payload。

独立纯数学核对使用SciPy1.10.1的`binomtest(... alternative='greater')`及单侧97.5%精确下界，对n10/20全部32种胜出数核对：p值最大差0，下界最大差2.6334490144108713e−13，小于预设1e−10；p绝对容差1e−12。它只验证统计实现，不访问任何实验结果或数据。

- 数学记录`/tmp/libero-formal-transfer-statistical-calibration-20261105-v1.json`，SHA-256 `e0be095b56b16f83ceb3d431503db182e7ffd40f6e1c22aa209b966d7a5d7d90`。
- metadata预检`/tmp/libero-formal-transfer-freeze-preflight-20261105-v1.json`，SHA-256 `38e9d5eddd0e7e1397d8adf134ecc9e3f2d9332c363add68ed260bb9d9fe34bb`，UTC05:02:09.467196记录passed，研究调用0，`sealed_access_authorized=false`。
- 最终文件保全/文档/预算核验`/tmp/libero-formal-transfer-freeze-final-verification-20261105-v1.json`；保护1242份既有文件和原HEAD，AGENTS.md/.vscode及无关修改不动。

复核命令：

```bash
conda run -n libero-kecrl python -m experiments.libero_formal_transfer_protocol \
  --config configs/libero-formal-transfer-protocol_v1.json \
  --binding configs/libero-formal-transfer-freeze-binding_v1.json

conda run -n libero-kecrl python -m pytest -q \
  tests/test_libero_formal_transfer_protocol.py \
  tests/test_libero_transfer_split.py \
  tests/test_libero_causal_t1_dataset.py \
  tests/test_libero_bc_fomaml.py
```

下一步是实现并绑定正式执行入口、合法连续行选择性读取及独立审计；先做合成角色/初态映射/预算/统计测试和metadata检查，再按冻结G阶段执行。在此之前，所有封存payload保持未读，活动SPT/Knowledge/Module均不更新。
