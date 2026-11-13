# task7 停止点方向/网格只读分析：执行契约（实现于2026-11-10）

实现[2026-11-09科学冻结](libero-task7-mse-gate-stop-direction-grid-plan-2026-11-09.md)，配置SHA `ab870764b427d77bfc95fe3c6757e87b16d94d9c0232ffcf756079e3c1526c8f`不变。本文路径、产物的20261109前缀保留已冻结标识；实现/结果报告按Asia/Shanghai本地日2026-11-10命名；UTC时间戳按真实时刻保留，不强改日期或回写旧记录。

## 入口和绑定

主入口 `experiments/analyze_libero_task7_mse_gate_stop_direction_grid.py` 默认preflight，evaluate只执行主分析，run在独立进程顺序运行主/审计。CLI与两个payload API都在目录创建、来源CSV SHA或解析前检查新配置/绑定；七角色、本文、科学配置及父745依赖闭包均须逐字节一致。未绑定、不符或已有结果/目录/launch/process即拒绝。研究执行技术重试0，不覆盖。

监管进程记录UTC起止、Python命令、PID、退出码和总墙钟；两阶段各≤120秒/256MiB，总≤300秒，超限终止子进程并保留日志。主与审计均exit0、audit/completion存在且有效才算完成；主JSON保留原始pending状态，不回写成通过。CPU Python3.10.22、CUDA隐藏、PYTHONHASHSEED=0、OMP/OPENBLAS/MKL各1线程；数据分析进程纯stdlib，不导入NumPy/Torch。环境检查另开进程，不能把其Torch版本检查称为研究前向。

## 访问、投影和计账

新guard `experiments/libero_task7_mse_gate_stop_direction_grid_access.py` 只允许固定endpoints/parameters/trials三个来源CSV和四个输出CSV。每阶段按此顺序各一遍；每个来源读取前及阶段末各一次字节SHA，合计主/审计6遍、12次来源SHA。raw open/Path/io.FileIO/os.open、外部CSVreader、ZIP、数组/checkpoint读取或哈希、NumPy/Torch/环境导入、越权写入均拒绝。审计只读主输出，不写主表。

先对所有物理行的ID/condition/优化臂作字符串角色检查，再选择11个固定头的两个点/257坐标/最后12试步，选择后才转换白名单字段；未选数值始终是词法字符串。parameter权重列不在投影中。每阶段物理24022行/380100词法单元，选中5808行/65538投影，合计48044物理行/760200词法单元/11616选中行/131076投影。每阶段22个点、5654个坐标行、22616个保存向量坐标和132个末轮试步。CSV整行会被词法解析，不能宣称未选列字节未接触。

源schema、列顺序、全部18身份、11焦点头覆盖、257坐标无缺失/重复、22点和132试步、单遍/单列转换等均核对；预算超额立即拒绝。7个完整头只验证metadata身份分母，不分析其数值。元数据预检会解析父JSON完整对象，但只使用其manifest/ID/停止步；新分析不得从父JSON中的缓存向量绕过投影。

输出points22、trials132、heads11、strata12共177行，≤16MiB；保存所有空组和null值，所有新增前向/梯度重算/更新/环境访问及sealed访问0。原Policy500、未来5步、loss/表示/方向/网格/预算及旧cue/G/H状态不变。

## 数值复核和解释边界

主分析使用已冻结的标量/向量契约计算梯度几何、clip资格、精确sT与点积gap、余量分类及已有试步贡献。核对G=Gb+Gp、d=-G、三缓存范数、分支计数、T=MSE3/7+P、F=C6+T。零更新两点的所选标量/向量精确相等；短路径目标严格下降。

末轮恰为保存stop_k的j0..11，α=2^-j；固定辅助500上界、保存sT、原Armijo bound及两条件/最终接受/拒绝原因逐项核对，原决定不加epsilon。sM仅无clip边界拐点时从7Gb·d取得，精确惩罚导数用sT−Gb·d，hinge修正单列；clip未知或近零带保持不确定。线性α尺度和泰勒残差只描述已保存试步，不评估新α、不作为安全步长或新控制证据。

独立审计 `experiments/audit_libero_task7_mse_gate_stop_direction_grid.py` 在新进程独立读取三表，自行计算向量点积、范数、clip/hinge资格、余量/方向分类、trial贡献和严格接受条件。不得导入主分析，也不调用科学协议的gradient_geometry/feasibility_evidence/trial_terms或父trial_decision作为数值重建路径；允许共享元数据、字段解析和容差常量。所有177新CSV行/JSON记录/分组、两个进程访问账本重新核对。标量绝对容差1e-10、相对0；角色/计数/标志/决定精确一致。来源与新输出SHA、主JSON前后不可变性均验证。

正余量和负sT仅支持连续数学目标中的局部小步存在性；边界正sM仅约束同方向的足够小邻域。两者均不推出全局无解、loss/表示唯一根因、可用浮点安全步或闭环受益。结局分层11头不能当随机因果实验。

## 合成验证及执行

绑定前验证：完整物理行数但未选数值/权重列为非数值字符串的投影；光滑/hinge/clip与零/正余量；错误范数、向量、分支、坐标、角色、网格、接受标志与旧门上界拒绝；raw/ZIP/数组SHA/重复读取拒绝；未绑定API拒绝；完整规模主/审计及相关/CPU回归。生产无绕过开关。仅pytest子进程显式替换validate_call，且必须确认三个来源CSV均在本次pytest临时目录。验证XML/日志、CPU选择及环境/未绑定证明引用存入绑定validation字段。

```bash
conda run --no-capture-output -n libero-kecrl python -m experiments.analyze_libero_task7_mse_gate_stop_direction_grid --mode preflight
conda run --no-capture-output -n libero-kecrl python -m experiments.analyze_libero_task7_mse_gate_stop_direction_grid --mode run
```

默认输出 `/tmp/libero-task7-mse-gate-stop-direction-grid-20261109-v1.json` 与同前缀artifacts目录；实际日期以process/各阶段时间戳为准。结果另记2026-11-10报告和README/roadmap/500计划，不改已冻结方案、配置或历史产物。
