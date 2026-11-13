# 维护后真实重跑：A来源与15份原字节重建（2026-11-13）

**task7 A_original三个seed真实重跑及独立审计通过。94项缺失清单中，15份原文件的完整字节已在持久盘重新生成并匹配原SHA；其余79项尚无已核验对应物。** 旧/tmp目录未重新填入文件，没有补造旧实验JSON、测试日志、时间戳或completion。重跑不产生新的正式KECRL/SPT/SPI/Module结论。

本轮按用户要求重新生成实验文件、重跑并上传GitHub。此前query封存要求仍在，因此先完成不读query的A部分；B_fomaml所需source元训练query范围已经单独询问，尚待用户澄清。不能把当前A-only结果写成全部18头或整个缺失链已经恢复。

## 实际完成与核验

| 工作 | 实际范围/计数 | 验证结果 |
| --- | --- | --- |
| 原Policy重跑 | task7、A_original、seeds20261106/07/08，各500 updates，共1500 | 三个500端点tensor hash及完整checkpoint SHA均匹配历史记录 |
| 固定表示采集 | 4188 support合法帧/模型，42主前向+3首批重复 | 全七维logit/tanh及重复检查通过 |
| feature归档格式重建 | 原features/logits/predictions顺序与压缩writer，共3份 | 三个完整NPZ SHA均匹配原记录 |
| λ1辅助头重跑 | 3来源seed×3整demo折，共9头，各500，共4500 updates | 九个完整NPZ SHA均匹配原记录，包含参数、采样、500步trace和快照 |
| 独立CPU审计 | 采样/帧身份、全部快照、feature线性读出、辅助逐步预测/loss | 通过，未再次训练或调用CNN/HDF5 |

15份对应物为3份Policy500 checkpoint、3份current feature归档和9份harm_protected_affine头。核验比较的是原清单SHA与新真实运行输出文件的SHA，不只比较模型分数或文件名。新位置映射、完整摘要和每个原SHA见[可归档真实摘要](evidence/libero-postmaintenance-A-rebuild-summary-2026-11-13.json)。

GPU主段工作时间13.537832秒；CPU辅助训练工作2.998408秒、独立辅助审计0.281105秒。上述是各段内部工作时间，不能当作总人工/进程启动/审计墙钟时间，也不能据此推论预算是否足够。主段使用原物理GPU1，结束后GPU已释放；CPU辅助段intra/inter-op线程各1、CUDA隐藏，峰值RSS431931392字节，NumPy审计96333824字节。

独立feature线性重建最大误差8.848357637702975e−7；辅助训练预测重建最大误差6.963385668079525e−8，分别在预先固定的1e−5内。没有放宽原科学门或使用误差容差挽救结果。

## 访问、预算与谱系

只数值读取task7 meta_support demos0..29的causal obs[t−1]→action[t]、t≥1，432窗口、4188帧。主段HDF5打开2次、公开数组切片2160次；文件完整SHA会物理覆盖未投影字节，source meta_query数组解码0。validation、held-out、qualification、reserved、final、私有状态和环境初态读取0；环境调用0。

主Policy保持500 updates，原loss和未来稳定化5步保持；辅助λ1、257维、fit-only scaler、原PCG64采样和500 updates保持。9个A头全部完成后才开放probe快照；没有λ0效果对照或新的保护cue、闭环结果、checkpoint选择、活动SPT/Knowledge更新或Module注册。

两个阶段均在payload前建立新的科学配置与执行代码绑定，历史协议和原SHA不改：

- [A来源方案](libero-postmaintenance-rebuild-plan-2026-11-13.md)、[执行入口](../experiments/run_libero_postmaintenance_task7_A_rebuild.py)、[独立审计](../experiments/audit_libero_postmaintenance_task7_A_rebuild.py)、[配置](../configs/libero-postmaintenance-task7-A-rebuild_v1.json)、[执行绑定](../configs/libero-postmaintenance-task7-A-rebuild-execution-binding_v1.json)。
- [A辅助头方案](libero-postmaintenance-task7-A-head-rebuild-plan-2026-11-13.md)、[训练/独立NumPy审计入口](../experiments/rebuild_libero_postmaintenance_A_heads.py)、[配置](../configs/libero-postmaintenance-task7-A-head-rebuild_v1.json)、[执行绑定](../configs/libero-postmaintenance-task7-A-head-rebuild-execution-binding_v1.json)。

新谱系直接从仍在的原始合法source数据重跑A，随后从已审计新缓存训练辅助头，不删除或绕过旧冻结依赖。原有基于真实路径的历史执行门仍不能通过，917项路径变化仍需后续有来源的新版本处理。

## 验证和证据位置

来源/模型/数据接口及新重跑guard专项37项通过，新的九头角色/注册梯度/整demo隔离专项3项通过，共40项。使用发明HDF5及小型CPU策略验证拒绝query/private、未绑定执行、超域身份、真实500步计数及可复现采样；这些合成结果没有冒充上面的真实运行。未运行全量pytest，旧151/204项记录仍是历史记录。

全部src/experiments/tests编译通过。提交格式检查对两份历史冻结测试保留原字节例外：`test_libero_task7_residual_harm_protection.py`第80行既有行尾空白，以及`test_libero_task7_stepwise_m_nonincrease_acceptance.py`既有EOF空行；文件SHA分别仍为`5e2c4b2d2b36a160b82b697ff5cf9154d033d35181a6ae1a7201af9d098e9f1b`、`fa4d1d83d48736ac9033cda0e76c46de9c7ec80bbfa04f0e3fb69ea49e1c7b0e`，与原绑定一致。其余提交文件格式检查通过，没有为消除空白而改写旧冻结测试。

供GitHub核对的原字节副本：[37项测试日志](evidence/libero-rebuild-focused-tests-2026-11-13.txt)、[JUnit](evidence/libero-rebuild-focused-tests-2026-11-13.xml)、[3项辅助头测试日志](evidence/libero-rebuild-head-tests-2026-11-13.txt)、[JUnit](evidence/libero-rebuild-head-tests-2026-11-13.xml)、[真实A执行](evidence/libero-rebuild-A-run-2026-11-13.txt)、[真实A独立审计](evidence/libero-rebuild-A-audit-2026-11-13.txt)、[真实辅助头执行](evidence/libero-rebuild-head-run-2026-11-13.txt)、[真实辅助头独立审计](evidence/libero-rebuild-head-audit-2026-11-13.txt)。它们与持久日志逐字节一致，摘要内记录其SHA，不是重新编造的历史日志。

持久证据：

- `results/libero-postmaintenance-task7-A-rerun-20261113-v1/`：新Policy快照、source表示、events、rerun-result、independent-audit、rerun-completion。
- `results/libero-postmaintenance-task7-A-head-rerun-20261113-v1/`：重建feature、九个新头、head-rerun-result、independent-audit、recovery-ledger。
- `logs/postmaintenance-rebuild-20261113-v1/`：新测试日志/JUnit、执行预检、启动/进程和主/审计日志；不使用旧日志文件名。
- `docs/evidence/libero-postmaintenance-A-rebuild-summary-2026-11-13.json`：适合随代码上传的15项原SHA→新路径映射、真实预算、审计摘要与输入谱系。

摘要整理首轮因checkpoint引用额外含有weight_hash字段而在比较完整字典时停止，改为核对path/SHA字段后重新整理；没有修改实验输出或重新训练。

## 未完成部分与下一步

94项是维护后核对快照。现在15份原字节有持久盘上的可验证对应物，原/tmp路径仍未填入；其余79项没有已核验对应物，包括B权重/缓存/头、原父链数值JSON和旧验证证据。旧日志、walltime、PID及completion的原字节无法靠新运行伪造恢复。

B来源须先明确仅为原50步source元训练反馈开放tasks0/1/3/4/7的meta_query demos30..49；此前诊断阶段这些query封存，当前未默认解封。回答后才能另冻B来源重建及后续四参照/有界可行性执行；正式评价和初态继续封存。

GitHub归档代码、配置、文档和真实摘要；原数据、模型、NPZ和详细运行产物保持在个人数据盘。未把A-only重跑写成全部缺失已恢复或正式G/H重新通过。
