# task7逐lag与固定邻居集合分解协议（2026-11-08）

依据[对错转移结果](libero-task7-component3-history-transition-results-2026-11-08.md)，下一机制问题是：历史距离本身和邻居集合更换，各对应正纠正/负新增错误中的什么变化。本次先冻结再绑定执行，一次性完成主分析与独立审计；保持500 updates、未来稳定化5步及现有loss，G停止/H未开展与所有正式角色封存。

## 数据、四格及解释边界

固定task7、三seed20261106/07/08、A_original/B_fomaml六个现有500端点、source meta_support全部30示教。沿用t≥5、abs(action[t,2])≥0.025的3598共享帧，末20为负402/正75；保留全共同帧/末20/补集及全部demo。编码器见过全部support，本次仍是训练拟合diagnostic，不是未见迁移或Policy闭环结果。

每模型只读匹配的features字段（4188×256 float32），不解码同包logits/predictions；另读旧probe的11个主字段和共享9字段。共6 feature＋6 probe＋1 shared=13归档、81字段/阶段，主与独立审计共26次文件加载/162字段。路径、文件SHA、checkpoint身份与旧结果/审计/完成链逐一绑定；checkpoint仅作为元数据，不打开权重、RGB、HDF5、环境、初态/private/query/封存payload。

固定旧当前邻居集合N_C与旧历史邻居集合N_H，不重新做近邻搜索。四格名的首字母是距离，次字母是邻居集合：CC=当前距离/N_C，HC=五lag平均距离/N_C，CH=当前距离/N_H，HH=五lag平均距离/N_H。C距离取lag0；H距离取对应query/reference各自过去lag0..4距离的等权均值。历史不能跨demo、补齐或读取未来；目标/过去专家动作不作为特征。

每fold重新计算其它29条demo全部t≥5当前特征（含中性）的总体mean/std，与旧fold scaler逐维核对；std≤1e−6置1。同一旧scaler用于所有lag/两集合。距离为float64、64维块直接平方差求和，禁止Gram展开，query chunk64。负/正各5参考，gap=负均距−正均距，gap>0判正，其余判负；严格tie负、无epsilon改标签。CC与HH必须复现旧距离（abs1e−8）和原方向/tie（严格一致），不能选择样本以通过复现。

对四格gap分别记g_CC/g_HC/g_CH/g_HH。两条路径均保存：HC−CC与HH−HC；CH−CC与HH−CH。对称分解T=((HC−CC)+(HH−CH))/2、N=((CH−CC)+(HH−HC))/2，须重建T+N=HH−CC；交互I=HH−HC−CH+CC。固定每集合时，lag1..4各项(g_l−g_0)/5之和必须重建对应时间项。这里是确定性读出规则的代数分解，不是环境机制、Policy模块因果贡献或跨模型可比的绝对距离。

原CC/HH标签不同的帧，根据HC/CH是否各自已达到HH标签分四类：仅时间项、仅邻居项、任一单项均可、必须联合。原端点相同另列endpoint_same，仍保留两个交叉标签，不能隐藏中途翻号再恢复。报告全部四种原始对错转移、两目标方向、所有seed，不只选择正纠正和负新增。不同条件的计数、BA/召回保留分母，空组为0/null，不做p值或接受训练修改。

## 预算、输出、独立审计与停止

每阶段恰计算2158800个已选参考pair×lag距离、552652800个坐标平方差，0全库距离搜索；scaler fit共707832行/181204992坐标，全部fold归一化192983040坐标。两次复算不是新独立样本，源NPZ预算与派生CSV读取分开记账。

六表固定行数：distances21588（每行100个距离，共2158800数值），frames21588、scores144、demo-scores4320、attribution720、lag-summary1440。全部已算距离逐query以CSV归档，不新增派生NPZ加载；独立审计另读派生CSV全部数值，这是声明的结果核对预算，不能漏计为0数值处理。CSV总大小≤256MiB，边写边核对，不缓存全表字典。

独立审计单独进程、不导入主计算，重新计算全部pair-lag距离，逐值abs1e−8核对distance CSV，再从已核验发布距离重建严格四格标签、代数分解/逐lag项及全部导出表；不以独立浮点实现的极小舍入差引入新标签阈值。旧对角距离、标签/tie、全fold scaler也独立核对。新原始特征数组一遍加载后不重复解码字段，前后固定输入SHA不变。

CPU、Python3.10.22/NumPy1.22.4、CUDA隐藏、单线程、PYTHONHASHSEED0；主/审计各120秒、总watchdog300秒，当前地址空间VmHWM/VmRSS≤512MiB，无技术重试。记录配置/科学与执行SHA、seeds、设备/软件、实际文件/字段attempted/completed、距离/坐标计数、PID/时刻/耗时/RSS、表/输出摘要。任何来源、dtype/shape、ID、scaler、控制复现、代数恒等式、预算/时间或审计不一致即停止、保留partial，不重试/覆盖历史。

科学配置与本协议在真实数组之前固定。执行入口、guard、独立审计、runtime测试、metadata协议/测试及本协议七角色须单独绑定配置SHA、依赖SHA和完整预算，未绑定门控拒绝。输出仅新建于/tmp/libero-task7-lag-neighbor-decomposition-20261108-v1.json及对应artifacts目录；主pending-audit状态不回写，审计/completion另存。

无论四格哪项更好，均不自动加入训练、不选checkpoint、不改500/5步/loss或解封正式评价。本轮只解释固定读出取舍，不能推断必要几何信息、闭环根因、SPT/Knowledge更新或Module资格。
