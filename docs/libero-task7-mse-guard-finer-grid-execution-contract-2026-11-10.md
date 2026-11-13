# task7 coarse12 / fine20：执行契约（2026-11-10）

执行[已冻结科学方案](libero-task7-mse-guard-finer-grid-plan-2026-11-10.md)，不改科学配置SHA `03e8d5bd337a40da181100f8fdf1e177edb12b49ec2f152149d3513b16e666e2`。全部18个λ1辅助500头分别从同一起点复制两条路径，只增加固定试步尾部j=12..19。原Policy500、未来稳定化5步、表示/loss/方向/固定起点MSE上界保持。此项是fit优化机制诊断，无环境交互、Policy/CNN训练或前向，probe/query与正式评价继续封存。

主入口 `experiments/run_libero_task7_mse_guard_finer_grid.py` 默认preflight；run顺序启动NumPy主计算和独立Torch Float64审计进程，记录命令、PID、UTC时间、退出码与耗时。主≤300秒/512MiB，审计≤600秒/768MiB，总≤960秒，CPU/CUDA隐藏/单线程和版本按科学配置。已有结果、目录、launch、process或审计记录即拒绝，不覆盖，技术重试0；失败保留partial。主JSON保持pending，以两个exit0和独立audit/completion构成完成证据。

CLI及两个payload API均在研究目录创建、研究归档SHA/解码前验证科学配置、七角色、本文和父754项代码依赖闭包。仅合成测试可在隔离进程替换validate_call，并断言所有源归档和历史JSON在本次测试目录；生产没有绕过开关。

## 访问、预算与数值

新guard `experiments/libero_task7_mse_guard_finer_grid_access.py` 复用既有角色/归档来源/fit注册、只读表示及线性形式计账，覆盖各臂独立的trial_begin、trial_end和finish，避免旧12级常量泄漏。coarse最多12、fine最多20，按j递增且首接受即停止；fine拒绝12级不能提前耗尽，不能跳到新增级或给coarse第13级。方向必须完整负总梯度，接受仍为严格目标下降、精确单侧导数Armijo、MSE≤本头固定辅助500上界，无epsilon或事后选点。每头每臂最多50次辅助更新，0更新也保留；MSE可在固定上界内上升。

每当前状态只能两次线性形式、三梯度，试步一次线性形式且无梯度；开始及每次接受后测状态，最终不重复前向。每阶段≤1836状态、5508梯度、28800试步（coarse≤10800/fine≤18000）、1800更新。18次标准化/18次同对象复用，fit标准化50256行。状态/试步行、线性形式、坐标与参数写入按科学budget核对，新增尾部和耗尽轮全部计账。

IO guard覆盖np.load/成员/ZIP/raw open/Path/io.FileIO/os.open、mmap/fromfile、替代CSV、torch模型/优化器及环境导入。原25NPZ/161成员每阶段各读一次，只允许7个源缓存和18个λ1源头八字段；仅源500用于优化，0校验初始化，100/250不评分。只准本头20个fit示教、t≥1的注册切片。λ0、历史优化NPZ、Adam moments/probe字段、原.pt、RGB/HDF5/private state均禁止。

主新写18NPZ，每个七成员及[起点、coarse末点、fine末点]，Float64参数/int32更新数。写入门核对两个完成路径、参数逐值、来源scaler与fit行，禁止附加快照/成员。审计读取新18NPZ一次。全程68次NPZ读取/448成员解码/154归档SHA；主68SHA，审计86SHA，原归档进程前后保持。来源CSV读取/SHA为0。guard是研究访问契约，不是操作系统沙箱。

七CSV的行数为54端点、13878参数、≤1836状态、≤28800试步、36头、18配对和18scaler核对，总≤44640且≤64MiB；新NPZ合计≤64MiB。审计前后核对新表SHA，不回写主表。

## 历史控制与共同前缀

History仅投影冻结历史JSON的旧guarded12标量，排除参数、梯度向量及parameter_digest；不读旧CSV/优化NPZ。coarse角色标签规范化后，逐条复现394状态、3060试步、36端点和18头；绝对1e-10，相对0，ID/计数/分支/接受决定精确。原始JSON解析时物理包含历史向量，但不交给新数值计算。

Prefix在每进程内保留当前一条coarse路径的已测状态/试步及参数摘要，逐条核对fine共同状态、梯度向量和实际试过的前12级数值，要求精确一致；完成一对后释放。不为核对增加前向。第一分歧必须是coarse首次耗尽位置的j≥12接受，且此前共同记录全部匹配；若未分歧，最后参数、目标和停止类型相同。fine更新数不减、T不升，多更新则T严格降。旧7个50头由前缀及上限结构保证全路径不变；旧6个零头若打开则保留，不用预期结果反向拦截。

每对记录原6/5/7分层、前缀检查量、首次新增接受j/k、T/MSE/P/G差值、相对start的MSE差及试步成本。5短路径打开0/部分/全部只是搜索机制证据。新旧MSE上界参照分开：fine可高于coarse且仍低于start。原联合目标/梯度线索、来源全9头覆盖、旧cue/G停止及H未运行保持，不能因打开网格自动晋升Policy、SPT、Knowledge或Module。

## 独立审计与验证

`experiments/audit_libero_task7_mse_guard_finer_grid.py` 不导入新旧主数值实现。独立Torch Float64 autograd重演全部18对状态、试步和更新，独立计算MSE/P/T/F、单侧方向导数、接受/拒绝与最终线索，再与guard决定及主表逐项比较。每进程独立核对共同前缀；跨实现参数坐标/标量绝对1e-10、分支/决定精确，Float32锚点只作诊断。独立重建seed/source及cohort摘要，保留空组与零分母null。

绑定前验证：新局部测试（真实二次目标首次j12接受、fine不能12级早停、20级耗尽、前缀篡改）、25源归档的完整合成主/审计、1836状态/28800试步NumPy/Torch资源包络、相关重点与CPU回归、环境/编译/git diff。完整合成的6/5/7标签只是摘要角色夹具，数值使用平滑all50路径；零/短/新增级机制另用真实梯度合成测试验证。具体日志、XML、包络和选择清单随执行binding记录；真实payload前无研究归档SHA/解码或新优化。

```bash
conda run --no-capture-output -n libero-kecrl python -m experiments.run_libero_task7_mse_guard_finer_grid --mode preflight
conda run --no-capture-output -n libero-kecrl python -m experiments.run_libero_task7_mse_guard_finer_grid --mode run
```

产物固定 `/tmp/libero-task7-mse-guard-finer-grid-20261110-v1.json` 和同前缀artifacts目录。实际结果/provenance另记dated报告，再更新README/roadmap/500计划入口；已冻结科学文件和历史结果不修改。
