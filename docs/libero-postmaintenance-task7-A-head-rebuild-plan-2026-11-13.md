# 维护后A来源归档与λ1辅助头重跑（2026-11-13）

输入只来自已完成并独立审计的task7 A_original重跑：三个seed的500端点tensor和文件SHA均匹配旧记录。仍建立新的运行ID和谱系，不写回旧/tmp、不重造旧日志或完成收据。本段不依赖未解封的source meta_query，也不读取B、validation、held-out或任何正式评价/初态。

固定三个A来源×三整demo折，共9个harm_protected_affine头，每头257参数、零初始化、λ1、Adam500。沿用原fit-only mean/std64、std≤1e-6置1、标准化转float32，PCG64(seed+100000×fold)的500×64采样、lr0.0003/betas0.9,0.999/eps1e-8、clip norm1、原七维MSE+新增误差惩罚/7。复用原数值kernel和ZIP成员顺序，新建九头角色guard，禁止probe参与梯度、未注册参数/optimizer、额外autograd.grad、checkpoint读取及提前probe评分；原36头契约和历史判据不改。

4500实际辅助optimizer/backward/训练前向、288000训练帧访问，9 scaler/25128fit行。全部9头完成500后才开放全4188帧变换和0/100/250/500快照，共36读出批/150768行；没有λ0效果对照或新保护cue结论。原Policy新更新、CNN、HDF5和环境均0。CPU单线程、CUDA隐藏、原软件版本；主段≤180秒，新独占持久目录。

同时按原已存在的writer定义，以features/logits/predictions顺序和np.savez_compressed重写3份A feature缓存。归档格式转换不新增Policy前向。新头和转换缓存逐份比较原预期文件SHA，符合才能记为原字节已重新生成；不同则保留差异，不改旧SHA。

独立NumPy审计重建整demo采样、scaler、每步线性预测和λ1 loss，容差沿用原1e-5/1e-6；四个保存快照和其他六维逐位验证。它不再次训练、使用Torch/autograd或读取HDF5。所有原始数据/模型/NPZ留数据盘，GitHub只存代码、配置及真实摘要。

新科学配置和代码绑定在本段数组访问前建立。读取上游新完成记录仅固定输入来源/路径/摘要，不依据结果挑seed、fold、loss或预算。旧缺失JSON、测试日志和历史结论仍不被这段新执行替代。
