# task7 五个 fine20 新停止点只读核对：方案与执行契约（2026-11-10）

依据[已独立审计的coarse12/fine20结果](libero-task7-mse-guard-finer-grid-results-2026-11-10.md)，固定旧short5全体的新fine停止点：A/06/f1 k7、B/06/f0 k7、A/07/f1 k8、A/08/f0 k16、B/08/f0 k10（seed完整为20261106/07/08）。它们均先打开旧网格后再耗尽；其余6零/7完整头仅保留18头分母，不能当随机对照。原Policy500、未来稳定化5步、loss和表示保持，正式评价继续封存。

本项直接只读上轮结果JSON，计算五个停止点相对各自辅助500的MSE余量、已存梯度沿d=-G的方向关系，以及最后20次拒绝的分账。先固定白名单、判据、预算与代码，再计算新方向/试步指标；用户本次已授权完成此项分析及结果记录，无需再暂停请求执行许可。

## 输入、计算与解释

唯一研究输入为 `/tmp/libero-task7-mse-guard-finer-grid-20261110-v1.json`，SHA `006090f077ea86a55bb254b3d86f760ac2e2433b4eaa3777eddc1fae13d78c26`；父science/binding/audit/completion/process及dated报告均钉住。主与独立审计每阶段各解析一次、前后各SHA一次，总2解析/4SHA。元数据发现阶段已为检查字段名完整解析过1次，未计算新方向或试步指标，单独留痕；JSON解析会物理解码全部字段（包含未选头和参数），不宣称这些字段未被解码。后续数值投影仅允许本节白名单。

从54个端点中选固定5头的saved500/fine20_final，共10点；选标量、clip/hinge计数及四个257维已存梯度/方向（Gb/Gp/G/d），共10280个坐标。参数theta不参与运算，也不投影旧coarse向量。末轮从6569个既有试步中固定fine20、from_update=k、j0..19共100条；不试新α、不读取特征、不重算梯度。18个pairs只作身份/6-5-7旧组/新停止k及prefix角色检查；所有非选行数值及parameters字段不交给数值函数。

S=M500−Mstop严格≥0，M500是各辅助500的专家误差，不是原Policy误差e0或上一步误差。复核G=Gb+Gp、d=−G、各范数、点积、cosine、T=MSE3/7+P、F=C6+T。sT沿用已存精确右导数，不盲用−||G||²；无clip边界tie时sM=7Gb·d、sP=sT−Gb·d，有clip tie则sM/sP未知。无clip且无hinge tie要求sT≈G·d；hinge tie仅检验非负单侧修正。

严格记录S/sM/sT符号，1e-10仅为描述性数值带，不放宽接受规则或把小正余量重标为0。正余量和严格负sT支持连续数学目标上的足够小步存在性；S=0且正sM只支持当前方向在近0的局部冲突，不能推全局不可行。仅S>0且sM>0时给出线性尺度S/sM以及相对fine最小α=2^-19的比值；不是安全步长，不当作已计算新α。

每个末轮20条逐一复核α=2^-j、固定M500、严格目标下降、c1=1e-4 Armijo及MSE门，无epsilon，必须全拒绝。记录ΔM、ΔP、ΔT、ΔM/7+ΔP守恒、MSE超上界量、Armijo margin、ΔM/α割线、αsM与一阶残差、参数步长范数及四种原因计数。保留“Armijo也不通过”与“仅MSE否决”的反例，不要求预先指定的原因分布；不把线性预测当有限试步事实。

## 执行、guard与预算

stdlib-only CPU/Python3.10.22；CUDA隐藏、PYTHONHASHSEED0、BLAS三项单线程。主/审计各≤120秒/256MiB，总≤300秒；技术重试0、无覆盖。默认preflight，run顺序启动主分析与独立审计，记录命令/PID/UTC/退出码与成本。两进程exit0且audit/completion通过才算完成，主JSON保留pending。

执行前钉住metadata、guard、主/审计、测试及本文与父765项依赖闭包；payload API验证绑定后才创建研究目录和做本轮投影。guard只允许该JSON一次投影及2SHA、声明的新四CSV和自身RSS遥测；拒绝旁路open/os.open/io.FileIO、重复解析、附加投影/parameters字段、NPZ/CSV源文件SHA、Numpy/Torch/环境模型导入和所有新优化。它是研究访问契约，不是OS安全沙箱。

每阶段投影10端点、2570梯度坐标行、100试步，列值总量由配置COLUMNS精确计算；完整JSON物理解码另如实说明。输出四CSV：points10、trials100、heads5、strata6（两来源×三seed，空组保留），共121行/≤16MiB；主结果与independent-audit/completion/process为新JSON。无原NPZ/CSV/头/checkpoint/HDF5/RGB访问或其SHA，无Policy/CNN前向、梯度计算、更新或环境调用。

主复用已审计的标量几何与推论公式，将线性尺度参照正确改为2^-19。审计不导入主数值实现，独立从相同白名单重建向量点积/范数/推论/试步条件与分账，逐项核对新四表及JSON；数值绝对1e-10、相对0，身份/分支/计数/接受决定精确，源JSON审计前后不变。所有结论是已选择5条路径上的fit诊断，不能推正式泛化、闭环成功、Knowledge、SPT更新或Module资格。

绑定前：合成正常/clip/hinge/近数值带、1e-10内的严格门拒绝、20级缺失/重复/错序/改α、梯度或范数/方向篡改、字段访问及两进程完整合成；再做相邻专项、环境/编译/git diff检查。无共享训练代码变化，不重复全量Crafter/重型优化回归；具体测试XML、绑定前证明及工作区快照随binding留痕。产物固定 `/tmp/libero-task7-fine-stop-direction-review-20261110-v1.json` 与对应artifacts目录，历史产物不改。

完成后另记dated结果报告并更新README/roadmap/500计划入口。下一步以实际余量、方向及拒绝分布决定是否收束网格路线、再冻方向层面的单变量假设；本轮不执行方向搜索或修改训练。
