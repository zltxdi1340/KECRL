# task7 原M负界内联合T方向：科学冻结记录（2026-11-11）

已冻结[方向、配对控制、原证书/有限步接受门、预算与停止规则](libero-task7-original-m-bound-joint-t-direction-plan-2026-11-11.md)。本阶段只读完成元数据、源码和人造向量/目标，没有读取研究数值或运行新优化；实际头数、证书/有限步通过率未知。科学绑定状态scientific_frozen_execution_not_bound，不能代替执行绑定。

上一轮各18/18终点线性可行支持检验联合方向，却不能证明已经能走出合法有限步。现有guarded T控制已用κ=.02；新候选使用原ρ=.01负界和固定δ=√εbinary64=2^-26构造内缩的T优先方向，原核验证书/接受门不放宽。R=g(1−δ)、η=((-B/7)/b)/R、κ=η+δ；理想构造满足原球和负界，但只是内缩子集，不能声称原集合精确最优/完整可行。

两臂从相同18个原λ1辅助500头开始，候选只变方向。控制仍须按各自历史复现900/918/7691、0接受M上升，保留原负裕度、零裕度、角平分三个额外参照。原Policy500、未来稳定化5步、loss/表示/数据角色、四参照与原主0/18、G停止/H未开展保持。

先检查候选实测范数≤g及原selected M≤B、T<0，再检查同方向exact M≤B、T<0；有限步仍为20级网格、原严格T/Armijo/M500/逐步M门、首合法步及最后接受点。构造未分辨/不可表示、selected/exact非下降或网格耗尽都按原始值停止，不加epsilon、回退、修补或扩大预算。通过证书不算科学晋级。

未来总≤3600辅助接受更新（P/U各≤1800、S0），5508梯度状态/16524梯度向量/11016 autograd、77073 trial、89925线性形式/251647425 fit行。每状态≤4norm/3dot/4向量、1次方向、1个固定δ、0修补重试；候选几何记录≤2754、半径检查与P记录复核≤3672、公式逐坐标核对≤707778。原归档与CSV预算不扩张，CPU总≤1500秒，所有新数值产物留/tmp。

新增75项、相关118项共193通过，排除2项父实验历史“产物不存在”断言，旧测试/产物不改。覆盖独立二维公式和288组人造方向/圆周约束核对、两臂各50步真实人造clip目标的exact右导数/有限步接受、严格nextafter边界、不可表示/近共线停止、内缩不等于完整可行的反例、四参照、数据角色/预算篡改和阻断研究payload读取/SHA。py_compile、环境检查、git diff --check通过；未运行全量pytest。

证据：[协议](../experiments/libero_task7_original_m_bound_joint_t_direction_protocol.py)、[测试](../tests/test_libero_task7_original_m_bound_joint_t_direction_protocol.py)、[科学配置](../configs/libero-task7-original-m-bound-joint-t-direction-diagnostic_v1.json)（d5849f6e6a7fa54bf767251464d3107396fa4c393121d44574643f1523f57b71）、[科学绑定](../configs/libero-task7-original-m-bound-joint-t-direction-freeze-binding_v1.json)（379c30db0e3b112fae4688fbfd9c572ce029ec467e5d8368d8097f3bcde13474），覆盖910项依赖（完整父902闭包及新协议/计划/测试/配置、完成元数据）。验证日志/XML/环境及1620文件初始工作区快照均在绑定中存哈希；既有1620文件在此阶段哈希未变。

本阶段研究NPZ/CSV/checkpoint/HDF5、控制/纯M/几何数值JSON读取及SHA、真实新方向/梯度/trial/优化/环境调用均0。仅source own20 fit可用于未来计算，encoder已见全部30示教；query、validation、held-out、qualification、reserved、final、初态及其他环境数据继续封存。不更新SPT/Knowledge、不注册Module。

下一步实现主入口、字段/角色/方向/半径/预算guard和独立U/S审计，完整合成及最大预算包络通过后完成执行绑定，再各运行一次，分别报告构造、原证书、有限步与四参照结果。
