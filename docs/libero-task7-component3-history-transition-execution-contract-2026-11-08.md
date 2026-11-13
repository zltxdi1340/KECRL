# task7历史方向对错转移执行契约（2026-11-08）

依据[冻结方案](libero-task7-component3-history-transition-plan-2026-11-08.md)和[科学配置](../configs/libero-task7-component3-history-transition-readonly_v1.json)实现。科学配置SHA为`3731985fc975ce88ace5dad55bbf5c5f53304463a6c7d5ff6b6375a5aadf4962`，其冻结时execution-unbound状态、旧文档和父结果均保持原字节；本执行契约、独立binding及未来completion分别记录实现和完成状态。

## 实现与执行门控

- [入口](../experiments/analyze_libero_task7_component3_history_transition.py)：默认仅执行绑定预检，`--mode run`顺序启动两个独立CPU子进程；`--mode analyze`仅供主分析阶段。分析入口根据共享标签/ID重新核对完整因果帧计划、t≥5历史、material选择、六模型全部预测/邻居和旧混淆计数，然后导出冻结的六张CSV表。
- [字段guard和记录](../experiments/libero_task7_component3_history_transition_access.py)：七文件、75个字段逐一授权，记录attempted/completed，拒绝重复、越权、pickle/mmap及替代读取路径。只共用IO、来源预检和计时/内存设施，不共用数值统计实现。
- [独立审计](../experiments/audit_libero_task7_component3_history_transition.py)：不导入主分析，按demo/step查表重建合法历史，逐帧重新判定四格，使用独立分组逻辑重建全部表、父混淆计数和三seed投票。NumPy均值/线性分位数是共同基础运算，不将其包装成独立的数值库验证。
- [运行测试](../tests/test_libero_task7_component3_history_transition.py)：合成数据覆盖完整7＋7文件链路、共享类分母、所有转移、零gap/null、原Policy零输出、中性断run/左删失、邻居非法/重复/错类/次序、篡改及门控恢复。测试中的缓存均自行构造，不解码七个真实归档。

执行前必须通过冻结元数据验证器的七角色检查，并额外核对固定科学SHA、CPU环境、输出身份、全部执行代码依赖与本契约SHA。角色为analysis、independent_audit、payload_guard、runtime_tests、metadata_protocol、protocol_tests、protocol；任一缺失或改变即在数值payload之前拒绝。元数据协议、其测试和原科学文档不修改；完整角色/依赖清单、实际软件验证、git身份及命令写入独立`configs/libero-task7-component3-history-transition-execution-binding_v1.json`。

## 文件和字段访问

主分析和审计分别只加载固定共享NPZ一次、六份probe NPZ各一次。共享允许9字段，probe各11字段，每阶段恰为7个文件/75个字段；字段列表直接取冻结配置，不追加public_proprio、ordinary、features/logits或新的动作预测。参考checkpoint身份仅作为JSON来源元数据，不打开checkpoint。

guard拦截NumPy两处load入口、NpzFile字段索引（包含`.f`接口）及底层ZipFile.open；后者仅允许由当前获授权字段索引触发，禁止用zip直接取出未授权字段。文件入口同时拦截builtins.open、io.open与Path.open，源归档只有显式np.load或SHA作用域可读，额外文件读取仅限该阶段固定CSV输出。禁止direct fromfile/read_array、非路径load、mmap、pickle、未列文件、重复文件/字段以及模型、HDF5、环境导入；已存在的torch/h5py相关加载/梯度/优化器接口也拒绝。退出时全部hook恢复。

这是受控执行的研究访问契约，不声称替代操作系统安全沙箱。NumPy读取同一字段时可能为格式头和数组内容分别打开zip成员，此为同一次字段解码；实际完整字段解码只有一次，不能把格式头读取另计为新样本。固定文件完整SHA的字节校验也不计为数值数组加载，前后SHA须相同。读取结果数组设只读，不能改写原缓存。

## 输出与审计口径

六表身份和行数沿用科学配置：frame-transitions21588、model-transitions144、demo-transitions4320、seed-agreement-frames7196、seed-agreement-counts192、tail20-neighbors57240。CSV逐行写出，避免把全部邻居字典同时积存在内存；三seed对齐只保留固定帧转移码。

逐模型及demo/时间层/方向四格都保留总类分母、current正确/错误、history正确、净变化、四格计数和各自条件率。逐demo全部30条及空组保留；零分母输出空字段/null，不填0。模型四格再输出目标绝对值、run age、历史中性/相反标签数、时间坐标、两模式gap/归一gap和两类邻居组成的n/均值/中位数/Q25/Q75；未定义归一gap只减少该指标定义数，不删除帧。连续gap只按模型汇总，不跨模型比较绝对距离。年龄0..3/≥4及左删失交叉计数分别保存。

逐帧目标历史仅用于诊断标注，不作为输入。material run在原始连续步上定义，中性打断；t1起始标左删失，不访问action0。每类当前/历史各5个邻居的交集、参考demo数、参考尾窗数均保留。独立审计核对全部431760条保存邻居的ID、demo隔离、正负分块、距离有限/非负、类内排序及同距ID稳定序，重新由保存的距离重建预测/tie/gap；**不重新计算feature距离，也不证明每个保存邻居全局最优**。

所有21588个模型帧、各层/逐demo/三seed输出和57240行末窗邻居明细逐列核对；同时与上轮六模型的current、history及原Policy的三时间层和逐demo all_common混淆计数对齐。表SHA前后不变，整数/字符/布尔/空字段严格一致，浮点绝对容差1e−10、相对0。共享标签、模型行和重复统计都不是新独立样本，不做显著性或方法接受判断。

## 预算、监督与provenance

两阶段均为Python3.10.22/NumPy1.22.4、CPU、CUDA隐藏、PYTHONHASHSEED0及OMP/OPENBLAS/MKL单线程。每阶段活动≤120秒，外部总watchdog300秒。内存用当前地址空间的峰值RSS≤512MiB作保守上界，取固定`/proc/self/status`的VmHWM/VmRSS（仅此遥测路径允许），包含解释器/库基线；它比只核算工作数组更严格，没有扩张科学预算。Linux的ru_maxrss可能保留exec前旧地址空间的峰值，另存来源记录而不用作当前限额；绑定前CPU回归曾因此误拒合成进程，日志保留并有专门回归测试。数组加载7＋7、成员解码75＋75，0训练/backward、CNN/Policy前向、近邻搜索、环境、checkpoint/HDF5、初态/private/query或封存payload。

绑定后的固定命令为：

```bash
timeout --signal=INT --kill-after=5s 300s conda run --no-capture-output -n libero-kecrl \
  python -m experiments.analyze_libero_task7_component3_history_transition --mode run
```

监督进程每阶段使用全新子进程组、独立日志和140秒外围上限（且服从总300秒），子进程自身仍执行120秒活动门控。中断/超时只终止自己创建的子进程组，保留阶段和总进程记录，不干预其它作业。无技术重试、切换seed或覆盖历史输出；入口发现既有launch/process/结果/artifacts即拒绝重跑。失败保留partial和实际完成计数，未通过主阶段不启动审计。

主结果以pending-audit状态一次写入；审计通过另写independent-audit及completion，不重写主结果。所有输出在科学配置指定的新`/tmp`路径，监督日志/launch/process使用同名前缀；配置、绑定、输入/表/结果SHA、Python/NumPy、CPU/环境变量、PID、起止UTC、实际读取计数及耗时/RSS均保存。环境seed和初态索引不适用。

无论新结果如何，保持500 updates、未来稳定化5步、现有loss及G停止/H未执行，正式角色封存。结论仅限source meta_support训练拟合，不能把尾窗当物理阶段、动作符号当世界位移、邻居gap当Policy置信度，或由纠正推断闭环成功、SPT/Knowledge更新或Module资格。
