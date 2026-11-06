# LIBERO G执行v2：来源审计字段修复与预算继承（2026-11-05）

本版本只修复执行器审计接口并继续同一次G预算。[科学协议v2](libero-formal-transfer-protocol-v2-2026-11-05.md)、[原执行契约](libero-formal-G-execution-contract-2026-11-05.md)、原代码/绑定/产物全部保留；500更新、loss、seed、初态角色、G门槛和统计方法均不变。用户授权启动G，本修复不重复来源训练、不选择权重或追加科学预算。

## 错误、影响及证据

v1已完成全部20个seed的来源训练，保存60份A/B/C生成器。随后的独立source审计在消费learner日志时抛出`KeyError: 'disposable_source_sgd_updates'`。真实既有learner字段是`disposable_policy_sgd_updates`；审计器误写了字段名，先前合成审计fixture沿用了错误名称，导致70项专项未覆盖真实生产者/消费者的该接口。该错误属于审计实现问题，不能据此判定算法性能。

失败发生在目标适应前：生成器Adam2000、临时SGD5000、来源数据梯度20000、VJP10000、checkpoint60；目标适应、评价、环境、初态数值读取均0。实际失败段活动时间169.72777701774612秒。原始证据为`/tmp/libero-formal-G-artifacts-20261105-v1/source-training.json`、`failure.json`、`events.jsonl`与`/tmp/libero-formal-G-run-20261105-v1.log`，均不修改。v1技术中止，不产生有效G决定。

另建[独立审计v2](../experiments/audit_libero_formal_transfer_v2.py)修正该字段，已只读重审全部60份权重、1000次来源配对更新、采样子流和计数，结果`passed_source_training_audit`。证据`/tmp/libero-formal-G-source-audit-corrected-20261105-v2.json`，SHA-256为`c31b6ccba3cdad9cd083f14169716694b11dfdd97500fb422599742963911b51`；本次审计1.6130098812282085秒，无Policy/优化器/环境/初态调用。没有重新训练、替换seed或比较候选效果。

## 如何继续与门控

[续执行入口v2](../experiments/run_libero_formal_transfer_G_continuation_v2.py)复用原绑定的`Training`实现；[续执行门控](../experiments/libero_formal_source_continuation.py)在GPU或payload访问前核对原失败、source报告、日志、旧绑定和修正审计的SHA，且只允许这个目标训练前的完整source段。任何提前目标/状态访问、部分计数、旧依赖或权重变化均拒绝。

全部60份来源权重直接引用原路径及tensor/file hash；不复制checkpoint、不新增来源更新或筛选seed。仅将旧事件日志原字节前缀复制到新的实时账本并追加续执行标记；原失败记录原样保留，完整审计仍重算全部来源和目标计数。新产物使用`/tmp/libero-formal-G-artifacts-20261105-v2`与`/tmp/libero-formal-G-result-20261105-v2.json`，不会覆盖v1。

所有来源工作attempted/completed计数完整继承，新的生成器Adam和临时SGD额度已耗尽，试图再更新即拒绝。16小时活动wall预算继承失败段和修正只读审计，先前活动时间合计171.34078689897433秒；不重新获得16小时。新增的运行预检和目标/闭环成本继续累计；实现测试与修复等待不冒充研究调用。

目标适应仍为320个Policy各500次，支持窗口采样使用固定独立torch子流；生成器和Policy参数按原checkpoint完整载入，构造时临时随机权重全部覆盖，CNN无随机dropout，不依赖未保存的来源全局RNG状态做目标采样。环境重新固定seed20261022、稳定化5步。完整训练审计通过前不解码任何官方初态；validation只允许10..29，source只允许0..19。H、qualification、reserved继续封存，H不自动启动。

## 验证、绑定与下一步

78项修复相关专项通过，补充直接消费真实既有learner输出的审计联测，以及计数继承、禁止重复来源更新、剩余wall预算和提前数据读取拒绝测试。另有全量非Crafter CPU回归与编译，日志保存至`/tmp/libero-formal-G-execution-cpu-regression-20261105-v2.log`；可选Crafter依赖相关测试仍排除。

[v2执行manifest](../configs/libero-formal-G-execution-manifest_v2.json)保留相同科学配置和G计划，新增原失败段及修正审计的不可变provenance；[v2执行绑定](../configs/libero-formal-G-execution-binding_v2.json)在重新启动前冻结。启动GPU1前重新检查空闲与磁盘；仅续做160000次目标适应Adam、固定离线曲线、完整训练审计和6400条闭环，不扩大预算。

下一步通过v2预检后恢复G，从目标适应开始，完成后独立审计并作一次性G决定。实际启动/状态见[执行记录](libero-formal-G-execution-2026-11-05.md)。当前没有正式效果结论，不接受活动SPT、更新Knowledge或注册Module。
