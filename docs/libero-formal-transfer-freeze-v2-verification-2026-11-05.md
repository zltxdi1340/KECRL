# LIBERO正式验证方案v2冻结核验（2026-11-05）

**当前有效设计为v2；方法和统计代码已冻结，执行绑定尚未完成，正式实验尚未开始。** 完整设置见[v2协议](libero-formal-transfer-protocol-v2-2026-11-05.md)。本轮研究训练/环境/封存payload读取均为0。

## v1设计错误与修正

v1复用G准入和H最终推断的同一批20个训练seed/生成器。G按训练结果筛选后，H的seed差值可能受准入事件影响，不能直接沿用预期的独立二项模型。v1的metadata/软件测试通过不等于该统计设计正确；v1从未执行或读取封存数据。

v2使用G的20261106..20261125和H的20261126..20261145，两组各20、互不重叠。G只决定方法配置准入；H在独立初始化和采样流上重新训练source生成器，不复用G权重或筛选子集。因此最终20个训练重复不受G的seed筛选条件化。源数据/任务固定的统计假设及效应范围仍按协议限制，不声称任意任务总体效应。G非退化是方法校准证据，不代表各H权重逐个通过非退化或完整资格契约。

v1的配置、绑定、代码、测试和文档六份文件均原样保留。修正记录`/tmp/libero-formal-transfer-v1-cohort-design-review-20261105-v1.json`，SHA-256 `bae00c3ee737b5f7c32359759b57255ab58b8e8b911807eb1d24b304a3ab74f9`。v1不作为正式执行依据。

## 冻结内容与预算

主指标为held-out5/8/9上500端点闭环成功率。B−C与B−A同时满足严格seed胜出概率单侧精确二项检验+Holm familywise α0.05、观测宏平均增益≥5个百分点、任一任务平均差≥−5个百分点。20个H seed是统计单位；概率检验不是平均效应显著性检验，平局计非胜出，离线MSE不能替代主结果。

G阶段320个适应Policy/160000适应更新/6400条闭环；通过全部预注册的validation及来源非退化规则后，H重新来源训练，再做180个适应Policy/90000适应更新/5400条最终闭环。每Policy保持500、稳定化5步、固定500端点，所有seed/臂保留，不因效果续训或挑模型。

全流程上限：500个适应Policy，250000适应Adam+4000生成器Adam+10000临时SGD，**264000个optimizer更新**；11800条episode、3540000个Policy动作、59000个固定零动作、3599000个native step。源数据梯度40000次、VJP20000次；生成器端点120份。qualification预算0，Module注册、活动SPT与Knowledge写入关闭。

| 冻结文件 | SHA-256 |
|---|---|
| [v2科学配置](../configs/libero-formal-transfer-protocol_v2.json) | `74d28d96f5e6dcc29e48baa37ece117706c82278095f211e11829506a70f1440` |
| [v2绑定](../configs/libero-formal-transfer-freeze-binding_v2.json) | `0707c74c6d6224503e3e6d525f48a289afcd75af9d3cc3479a13e75b6e3b49f5` |
| [v2协议](libero-formal-transfer-protocol-v2-2026-11-05.md) | `3fded64b7c89cadc8e7a1d7d6e671849b4e38a2f5cd7f42af99fbe51b79df6de` |
| [v2 metadata/统计代码](../experiments/libero_formal_transfer_protocol_v2.py) | `6cdb9e51eeb419353cd52d4782a4393981b1d33ed319c6785a5b4444a9f96ee9` |

绑定固定科学配置、协议、统计代码、测试及已有算法/接口源码；`execution_binding_complete=false`。下一步实现的训练/rollout/选择性读取和审计入口必须另行绑定，不能把metadata绑定当成已具备执行能力。

## 验证

[v2测试](../tests/test_libero_formal_transfer_protocol_v2.py)与transfer/causal/FOMAML相关测试：**80 passed in1.50s**。覆盖独立G/H seed与来源预算，拒收G训练重复进入H最终推断，角色/预算/绑定/alpha漂移、缺失/重复计数、平局、单seed高分、小实际效应和被宏平均掩盖的任务退化等情况。metadata检查不导入Torch/HDF5/仿真或打开研究payload。

CPU回归**1459 passed in18.05s**，日志`/tmp/libero-formal-transfer-freeze-cpu-regression-20261105-v2.log`。排除`tests/*crafter*.py`、`test_spatial_gather_spt.py`和`test_smoke.py`，不声称全量可选Crafter通过。合成测试与研究调用分开。

独立数学核对：SciPy1.10.1，n10/20全部32种胜出数的精确p值最大差0，单侧97.5% Clopper–Pearson下界最大差2.6334490144108713e−13；容差分别1e−12/1e−10。记录`/tmp/libero-formal-transfer-statistical-calibration-20261105-v2.json`，SHA-256 `a1ab872d3901a64676d3a33079a37f1183523b6f8cbeac860d159238c6ea3ff6`。只是纯数学验证，不是研究统计结果或联合功效保证。

metadata预检UTC05:14:21.549517通过：`/tmp/libero-formal-transfer-freeze-preflight-20261105-v2.json`，SHA-256 `0e46d06210bd2a22d57ba58e4210139c67a885762d6dbfe46f6530c8c2b7560a`。实际研究调用0，封存访问未授权。

最终核验记录：`/tmp/libero-formal-transfer-freeze-final-verification-20261105-v2.json`；1242份任务开始前既有文件及全部v1文件保持不变，HEAD/AGENTS.md/.vscode/无关修改保留，配置/预算/文档/代码hash核对通过。

复核命令：

```bash
conda run -n libero-kecrl python -m experiments.libero_formal_transfer_protocol_v2 \
  --config configs/libero-formal-transfer-protocol_v2.json \
  --binding configs/libero-formal-transfer-freeze-binding_v2.json

conda run -n libero-kecrl python -m pytest -q \
  tests/test_libero_formal_transfer_protocol_v2.py \
  tests/test_libero_transfer_split.py \
  tests/test_libero_causal_t1_dataset.py \
  tests/test_libero_bc_fomaml.py
```

下一步实现并绑定G/H独立来源训练、合法初态行区间注入与阶段门控，先做合成和metadata检查，再启动G。G失败即停止、H保持封存；本次未启动G/H。
