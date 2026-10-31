# task 4/7 Full Policy 闭环诊断

固定 CNN-only Full Policy、500 updates、三个训练 seed，在 source `candidate_nonregression` 初态 `0–19` 上运行；每个模型 20 episodes，环境 seed `20261022`，stabilization 5 步，最多 300 步。6 个重训模型均精确复现 source query run 的 final weight hash。

结果：整体成功 `7/120 (5.83%)`；task 4 为 `3/60 (5.00%)`，task 7 为 `4/60 (6.67%)`。task 4 三 seed 为 `0/20、2/20、1/20`，task 7 为 `0/20、1/20、3/20`。其余 113 个 episode 均以 `time_limit` 结束，没有异常终止。

这说明离线 query MSE 的稳定排序尚未转化为高闭环成功率；当前瓶颈仍在自由运行后的控制链、状态漂移和动作时序。初态集已有此前诊断使用，不能称为未见测试集，也不构成 Module qualification。未访问 qualification、validation、held-out 或 final evaluation，未更新 SPT/Knowledge。

结果：`/tmp/libero-task4-task7-full-policy-closed-loop-20261030-v1.json`，SHA-256 `03439cc0172f3ba29ee4527d3c64d1da16edd93d2eaf63134344e4c8b15f49ce`。配置和 runner 已由提交 `fc79cd6` 推送。

下一步：在相同闭环协议下分析 7 条成功轨迹与 113 条 time-limit 轨迹的首个动作、首次 close、joint drift 和目标接近阶段，区分动作时序失败与状态漂移失败；不改变训练预算或读取资格初态。
