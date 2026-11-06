# 局部夹爪状态受控读出：执行绑定（2026-11-04）

这是[已冻结科学协议](libero-source-internal-input-sensitivity-review-and-gripper-state-control-protocol-2026-11-03.md)的执行实现，不修改研究条件、训练或判断规则。用户已指示实现、绑定并按冻结预算运行。本文件记录执行前快照，后续数值结果写独立dated report，不能把本快照当作完成结果。

## 设计和入口

- [科学配置](../configs/libero-source-internal-gripper-state-local-control-diagnostic_v1.json) SHA `bf38389d06d2d43c74991365d4763cc0cbbe94ee8517ef1c9cf51ca4d4d40206`。
- [设计/算子绑定](../configs/libero-source-internal-gripper-state-local-control-design-binding_v1.json) SHA `e81809ebff82d4adebf39aafb2b13cbd963db9d69a4c30aceed155ddca8257d8`，旧协议、算子和历史结果保持原样。
- [执行入口](../experiments/run_libero_source_internal_gripper_state_local_control.py)、[独立数组审计](../experiments/audit_libero_source_internal_gripper_state_local_control.py)及[执行专项](../tests/test_libero_source_internal_gripper_state_local_control_execution.py)分别实现读出、结果复算、合成检查。
- [执行绑定](../configs/libero-source-internal-gripper-state-local-control-execution-binding_v1.json)在新源观察、checkpoint payload和Policy前向之前锁定runner、tests、auditor、原GPU资源门槛代码、本执行文档、设计hash、runtime和启动环境。

范围仍为task0/1、三seed、A/B/C全部18个固定500端点，仅source-support demos0..29、`obs[t-1] -> action[t]`、t>=1。306个全部open后expert-open帧（149/157）、原正确/错close/neutral标签、主事件pre-action锚点及次级lag1均不变。两维公开gripper以外的RGB、joint、目标和权重固定。当前不访问query、任何初态/private state、qualification、validation、held-out或final evaluation。

## 执行预算、输入与停止

完整Policy调用全部通过同一个计数入口：2457主调用、54重复，187758/3996有效预测行、196020/4320计算槽；局部主评分8262行是有效行子集。保留原batch8、sequence10、stride10和batch index1重复，不添加真实帧warmup或额外parity前向。先记attempt，再调用，再记完成；失败的尝试不能从预算中消失。

新共享HDF5观察每模态6954行、action6954行，action0读取0。先验证角色和完整文件SHA，再由既有causal dataset仅读授权切片；不重新读取action0生成标签。源目标/ID逐元素对齐旧NPZ，819个原batch摘要与历史逐一比较，identity的62586行预测严格逐元素一致。

所有6648个未处理帧、padding、65个at-event参照及同完整输入行输出必须精确相同。两干预有效未处理行比较119664、padding比较5508、sham比较1170；lag+1两锚点相同由完整输入相等检查覆盖。54个重复比较完整输出（含padding），不将4320槽误写成3996个有效行以外的新样本。18个checkpoint分别核对文件和tensor hash，前后权重不变、grad全None。

输出保存全部条件的有效预测、含padding的完整预测、重复预测、源行映射、原始/替换公开夹爪和逐帧L2变化、所有原标签分层、逐demo及等demo统计。结果路径 `/tmp/libero-source-internal-gripper-state-local-control-20261104-v1.json`，artifact目录 `/tmp/libero-source-internal-gripper-state-local-control-artifacts-20261104-v1`。路径须新建，不覆盖历史。异常保留failed JSON、事件和活动部分数组，立即停止；不更换checkpoint/锚点、放宽容差或自动重试已消费的研究前向。

主cue仍为同task/arm主事件锚点的三个seed同时符号错误率严格下降且平移MSE不增加；次级lag1不能救回主失败。即使cue通过也不自动改训练。混合观察无合法新专家目标，误差相对于原观察的专家动作；不能解释为物理释放原因、闭环改善或新SPT/Knowledge/Module结果。

## runtime与验证

仅空闲physical GPU1、进程cuda:0；启动环境固定`CUDA_VISIBLE_DEVICES=1`、`CUBLAS_WORKSPACE_CONFIG=:4096:8`、`OMP_NUM_THREADS=1`、`OPENBLAS_NUM_THREADS=1`、`MKL_NUM_THREADS=1`。原资源门槛要求使用前memory<=512MiB、util<=5%、无该卡compute进程；不使用GPU0或CPU回退。Python3.10.22、Torch2.4.1+cu124、CUDA12.4、NumPy1.22.4、h5py3.11.0；单线程、确定性算法与cudnn开启、benchmark/TF32关闭，复用原确定性pool，不运行额外pool前向/反向。

执行前新入口/审计及测试编译通过；新执行专项、原算子、事件分解和原逐帧协议合计 **90 passed in 0.53s**。检查用合成数组/FakePolicy，真实Policy调用0。它们覆盖padding不能作为负索引、固定原标签、主次cue不能互相覆盖、预算耗尽拒绝调用、异常尝试仍计数、启动环境不一致拒绝、既有输出保护及preflight失败记录。未改共享框架，无需为验证代码启动训练或完整可选环境pytest。

独立审计仅读取已保存JSON/NPZ及文件hash，不执行新模型或环境。它逐项复算所有分层与逐demo统计、主cue、实际预算、原预测、未处理/padding/sham/重复精确性和输入剂量；只有描述性算术复算允许1e-12绝对差，神经预测parity仍为0容差。执行与审计结果分别保存，不把自报计数替代原始数组核验。

保持500 updates、闭环默认稳定化5步、现有loss/观测；训练更新、backward和环境交互全部0，不接受活动SPT、不更新Knowledge、不注册Module。
