# M6-E03：10题Conditioning On/Off pilot执行契约

2026-10-08；用户在明确提出10题、开/关各重复一次、40次GPU生成后回复“请执行”。
授权限本pilot，不扩至50题/五seed，不实施Gate、模型算法或训练改动。
使用analysis-campaign的固定比较/停止规则及detached-launch启动验证流程。

父对象M3 `20261002-m3-six-policy-v3`，contract
`a548306d4131e4df0cba1340d3272a4882feae1d9c993c881e281f4551e433c9`。
样本完全沿用E02原冻结10题（每context2题，seed42），包括冲突/歧义/未定位题，
不用AI裁决筛题、不改变dataset gold。原M3六策略、M0–M5证据只读。
exp_id=M6-E03-PILOT；section_id=conditioning；item_id=E03；claim_links=C05/R01/R05/R06；
paper_role=main_required_candidate（完成与可信性检查前不是可用论文结果）。

## 比较、预算与统计

On为full canonical两槽、现有weaver_integrated；Off仍相同full检索，使用已有接口
query_retrieved_memory_conditioning=false与retrieve_but_do_not_condition。
两臂都Weaver生成8新latent，检索两槽16tokens；实际conditioning为16 vs 0，
位置/长度改变属于接口总效应，不解释为纯内容效应。
每次重建CPU副本并完整恢复snapshot Python/NumPy/Torch/CUDA RNG；不重建Bank。
同checkpoint/tokenizer、原prompt、40-token解码、同一物理GPU、单卡可见拓扑。
每题依次on0/off0/on1/off1；主分析只用repeat0，repeat1仅复现，不选最佳。
逐次比较prompt、检索槽/顺序/分数、bank指纹、写入0、Weaver一次、conditioning预算。
当前重复不稳则保存并停止，不重试到一致；旧on重放不一致则禁止旧on复用。

Primary全10题strict on−off。Secondary官方EM/raw recall、parser非空、候选有效、
parsed-only分子/分母；correctness/parser/valid/raw-recall转移与C/W/I 3×3，
每context差值、等权均值和leave-one-context-out。五context独立性未完全确认，
pilot只描述，不做显著性/等价宣称。MEOI与正负数量阈值未审批，故不能据pilot
宣布算法有效/无效或Gate通过；本次成功只指完整性与可重复性，数据用于下一决策。

## 当前资源、实现边界、停止

宿主8×RTX A6000均有其他任务，GPU0检查时约35.6GiB空闲/32%利用率，
GPU5约35.7GiB/77%；其余多为100%利用率。选GPU0单worker固定配对，
不干涉任何已有进程，启动前重查>=12GiB与型号。共享GPU可能影响耗时，不称独占。
预计模型显存按原12GiB门；实际峰值/耗时由pilot记录，暂不可靠外推。
生成预算最多40次，无额外LM smoke；整个worker上限120min、输出<=100MiB，
超时/资源不足/source或snapshot漂移立即停止并保存状态，不自动换卡/重启。
预注册检查/单元测试CPU完成；推理用已有memgen环境，评分用MABench环境。
后台tmux，自动串行；启动验证一次后停止监控，不将启动写成完成。

全部新增实现仅M6/e03实验驱动，复用已有policy_adapter及查询接口，无新增模型hook。
协议与输入hash在生成前冻结；独立输出M6/e03/pilot-20261008-01。
保留每次答案/运行trace及旧on比较；完整结果由脚本评分后写入，不自动进入E04。
