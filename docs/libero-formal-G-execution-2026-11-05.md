# LIBERO正式G阶段执行与修复记录（2026-11-05）

最新完成状态（2026-11-06复核）：G已于北京时间2026-11-05 23:10:01完整结束，6400条闭环及两次完整复审通过，准入未通过；H主检验未开展，继续封存。最终成绩、预算和停止依据见[完成结果](libero-formal-G-results-2026-11-06.md)。下文保留11月5日启动与修复时的进度快照，不能作为当前状态。

**G已按执行绑定v2恢复，当前正在目标Policy适应。20组来源训练及修正后的独立审计完成，尚无G准入或held-out主检验结论。** 科学设置仍为[冻结协议v2](libero-formal-transfer-protocol-v2-2026-11-05.md)。本次只执行G，完成独立审计和准入决定后结束，不自动启动H。

## 做了什么、为什么做

已实现正式入口、阶段角色门控、选择性初态读取、attempted/completed预算计数，以及来源训练、完整训练、完整闭环三次独立审计。完整训练审计通过后才解码source0..19和validation10..29，分别记录全局与局部编号；qualification、reserved和held-out不在当前访问范围。

首次执行v1完成来源训练后技术中止：审计器误用`disposable_source_sgd_updates`，真实既有learner字段为`disposable_policy_sgd_updates`。旧合成fixture也沿用了错误字段，未覆盖真实生产者/消费者接口。错误发生时目标适应、环境、初态数值读取均0，不能作效果结论。原代码、绑定、日志和失败产物原样保留。

修复另建[执行续接v2说明](libero-formal-G-execution-continuation-v2-2026-11-05.md)及审计器。只读重审全部60份来源生成器、1000次B/C配对外层更新、采样子流和实际计数通过；复用全部20个seed的原权重，不筛选、不复制checkpoint、不重新训练来源。继承2000次生成器Adam、5000次临时SGD以及已消耗的活动时间，16小时总wall预算不重置；目标Policy继续保持500次Adam、既定loss和闭环5步稳定化。

## 当前得到什么

- 78项修复相关专项测试通过，新增直接消费真实learner日志的接口联测及预算/时间继承拒绝测试；1554项非Crafter CPU回归通过（28.34s），编译与`git diff --check`通过。可选Crafter依赖相关测试排除。
- 282份执行依赖完成v2哈希绑定，实际metadata预检通过且拒绝H；预检明确禁止导入Torch/NumPy/HDF5/模拟器，未执行模型、训练、环境或新初态读取。
- G续执行UTC：`2026-11-05T07:09:14.056833+00:00`，北京时间UTC+8；worker PID `2584346`，后台进程组／conda PID `2584326`。
- 本次快照UTC：`2026-11-05T07:11:07.250847+00:00`。已完成`30/320`个目标Policy；最新heartbeat记录累计适应Adam `15000/160000`，生成器Adam2000和临时SGD5000保持不变；没有v2异常中止文件。
- 当前数值初态解码、环境和闭环次数均0；目标适应正从source任务开始，validation payload尚未读取，后续仅依G门控开放。H、qualification与reserved继续封存。

设备为物理GPU1、进程cuda:0/EGL1；两次启动前GPU1均显存30MiB、利用率0、无计算进程。两次确定性池化预检及技术开销均保留provenance，GPU0未占用。CPU测试主动屏蔽CUDA，与正式GPU可用性分开记录。

## 证据在哪里

| 证据 | 文件／SHA-256 |
|---|---|
| [科学配置v2](../configs/libero-formal-transfer-protocol_v2.json) | `74d28d96f5e6dcc29e48baa37ece117706c82278095f211e11829506a70f1440`，不变 |
| [执行manifest v2](../configs/libero-formal-G-execution-manifest_v2.json) | `1956451aca41a06a53c57dfca5e3c62bc650c644ed3a32722a90b81a06d655c3` |
| [执行绑定v2](../configs/libero-formal-G-execution-binding_v2.json) | `d4a4461e44a66278f008995d3f2a970a7c27563bc011a51f7a950e80c62bcf5a` |
| 修正的来源独立审计 | `/tmp/libero-formal-G-source-audit-corrected-20261105-v2.json`；`c31b6ccba3cdad9cd083f14169716694b11dfdd97500fb422599742963911b51` |
| 编译／专项／1554项CPU验证 | `/tmp/libero-formal-G-execution-cpu-verification-20261105-v2.json`、`/tmp/libero-formal-G-execution-cpu-regression-20261105-v2.log` |
| metadata预检／H拒绝 | `/tmp/libero-formal-G-execution-preflight-20261105-v2.json` |
| 启动命令、env、GPU、PID | `/tmp/libero-formal-G-launch-20261105-v2.json` |
| 当前运行快照 | `/tmp/libero-formal-G-running-check-20261105-v2.json`；`eea0019e777796a7ce27ceb7a357bae8018eb767b84fbac2a89d2b7e7843661e` |
| 实时状态／事件 | `/tmp/libero-formal-G-artifacts-20261105-v2/status.json`、`events.jsonl` |
| stdout/stderr | `/tmp/libero-formal-G-run-20261105-v2.log` |
| 完整执行后原始结果 | `/tmp/libero-formal-G-result-20261105-v2.json`，当前尚未产生 |
| 完整审计后完成收据 | `/tmp/libero-formal-G-artifacts-20261105-v2/completion.json`，当前尚未产生 |
| v1失败与原来源权重 | `/tmp/libero-formal-G-artifacts-20261105-v1/`及`/tmp/libero-formal-G-run-20261105-v1.log`，原样保留 |

最初1254份受保护既有文件与v1执行绑定275份依赖均未修改；AGENTS、.vscode及无关工作区修改保留。更新README、roadmap和500-update后续计划指向本记录；冻结科学协议与历史实验报告不覆盖。

## 下一步是什么

完成全部320个500-update适应，记录0/50/150/300/500五点support/query曲线（固定25200评价batch），通过完整训练独立审计后，才解码140行授权初态并运行4000条source＋2400条validation闭环。固定最多1920000个Policy动作＋32000个稳定化零动作；全部完整审计后一次性执行G接受/停止规则。

当前活动wall上限仍为16小时，含继承的171.34078689897433秒；历史吞吐估计G仿真约9小时，训练/审计另计，不保证完成时刻。磁盘启动需至少9GiB、运行保留512MiB，不删除其他实验；资源或技术检查失败即保留partial中止，不追加预算或悄悄计成效果失败。

G拒绝则记录准入失败、H主检验未开展；G通过也仅生成准入收据，本进程不会自动启动H。当前不支持正式效果结论，不接受活动SPT、更新Knowledge或注册Module；qualification预算0。
