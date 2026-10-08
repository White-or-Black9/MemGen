# 提交版论文基线（M0）

## 权威来源

- 唯一提交版：`review/nference_Time_Latent_Me.pdf`，8 页。
- SHA256：`6b04a3c49b9bb68cccf282059be34f7dadc516e4069e42b204fcae99f68ffed3`。
- 审稿来源：`review/review`；`review/gpt_review_suggest.md` 是候选修改路线，不是已验证的方法事实。
- 本次启动 HEAD：`ee1a94358bd3799b3aa20b2f36e05ff870a49542`，分支 `review`。
- 启动时仅提交 PDF 未跟踪；不得自动 stage、commit 或 push。

## 协议与配置

EventQA `eventqa_65536`，Accurate_Retrieval split，固定 5 个 context，每个 100 题。
同一 500 题以 base seeds 42、142、242、342、442 重复，不能视为 2,500 个独立问题。
正式 P7 每个 context 使用 base seed + context index 重置 RNG；生成上限 40 个新 token。
默认非 strict EventQA query；官方 scorer/parser 未修复输出。

| 项目 | 基线 |
|---|---|
| Backbone | Qwen2.5-1.5B-Instruct |
| Checkpoint | Kana-s/MemGen snapshot `269d9b1741130b94fffa410cdaa3d4bc74081a7f` |
| Checkpoint 子目录 | `Qwen2.5-1.5B-Instruct/triviaqa/weaver-sft/pn=8_pl=8_in=0_il=8/model` |
| Capacity / top-k | 16 / 2 |
| Retrieve / update threshold | 0.05 / 0.10 |
| Decay alpha | 0.05 |
| Storage space / device | Weaver / CPU |
| Query phase | 每题恢复同一 frozen snapshot，阻止写入 |
| Session scope | 单 context；不跨 context 共享 |
| Recent-text budget | 32,256 source tokens；不是压缩 bank-off 的 0.008 EM 行 |
| Latent length | 单 slot 8；检索最多 16；不能等同于 Reasoner 最终注入长度 |

实际 manifest、命令、历史运行 commit、软件版本、checkpoint 路径、context IDs 和证据校验和
位于 `baseline_evidence.json`。各方法的配置以其实际运行记录为准，不继承脚本默认值。

## 正式结果来源

- P7：`outputs/mab/formal_p7_effect_repeats/20260722T120000Z-formal-p7-effect-repeats`。
- Recent-text：`outputs/mab/p7_recent_text_32256_effect_repeats/20260722T153000Z-p7-recent-text-32256-effect-repeats/recent_text_32256`。
- Rolling summary / BM25 / Matched-16：`outputs/mab/paper_baseline_effects/20260722T012259Z-eventqa-paper-baseline-effects` 中对应方法。
- Dense E5：`outputs/mab/dense_top2_effect_repeats/20260726T_dense_top2_paper_repeats`。
- No conditioning / Direct top-1：`outputs/mab/ablation_effect_repeats/20260722T143500Z-eventqa-ablation-effect-repeats`。
- No decay：`outputs/mab/no_decay_effect_repeats/20260725T_no_decay_alpha0_full`，通过论文配套 runner 定位，EM 0.1628。

结果表由逐题记录独立复算：`RESULT_TABLE.md`。正式 P7 EM 为 0.188、Recall 为 0.2308。
旧 `outputs/mab/p7/five_repeat_summary.json` 对应 0.1968；另一组沿脚本默认参数运行的
`p7_recent_text_32256_effect_repeats/.../p7` 对应 0.1656。两者都不得替代正式 P7。
no-decay 对应论文配套的预设消融 campaign，不是后续 `p7_decay_000_corrected` sweep
（EM 0.1476），也不是 update threshold 为 0.08 的 `p7_decay_000` sweep（EM 0.1228）。

## 指标和复现边界

论文 pages 5–6 的 EM 是官方 **substring exact match**，不是 strict exact match。
Recall 检查原始输出是否包含 gold；不能单独证明定位了历史证据。
论文称为 parser failures 的数字实际按 `any(format_flags.values())` 累计。
标记包括空输出、花括号、Answer 前缀、多行和超过 24 个词，**不是 parser 是否成功**。
不能用 `500 - format_flags_count` 直接计算 parsed-only accuracy。

历史正式运行 commit 为 `b907065a22c9f4916c0693488ccf250924bfc2d1`，并记录 `git_dirty=true`。
没有完整历史源码 diff 时，“一个 config + commit 完全复现”的原 roadmap 验收条件不能声称达成。
现有预测可作为可信离线分析起点；新运行必须保存准确命令、源码 hash、环境和数据身份。
manifest 的 repo 内 EventQA data-config 路径当前缺失，官方 checkout 有对应配置；不得悄悄替换并声称原路径可用。
控制方法的 seed 目录名称不构成独立的运行时 seed 证明，需结合命令或日志再核查。

## 效率边界

采用 20260722T013124Z 主 campaign 与 20260726T100413Z Dense campaign，分别为 GPU 5 和 GPU 4。
各自记录 serialized、三次重复和 model-loading excluded。
提交版“所有方法同一 GPU”不满足跨 campaign 的物理设备一致条件；同型号也不能替代同设备。
时间包括方法准备及查询；VRAM 是增量 peak allocation，未计模型加载与 CPU bank 存储。
共享 GPU 的效果实验耗时和 paired bank-on/off peak 不用作独立方法效率证据。

## 重复核验

```bash
/home/baishilong/miniconda3/envs/memgen/bin/python scripts/eval/review_baseline_audit.py --write
```

脚本仅依赖标准库，不加载模型。输出到本目录，历史实验原件只读。
M0 的验收是冻结事实与缺口，不是填补缺失的历史证据。结论以机器报告 gate 为准。
