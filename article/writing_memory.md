# 写作记忆

- 审稿反馈：后续论文写作需要保留少量科研发现过程中的“试错-修正”痕迹，例如参数调试失败、模型收敛困难、编码容量不足等。优先用脚注或表下注释承载，正文只在必要处点到，避免把结果章节改成实验流水账。（此条已持久化到全局记忆 `thesis-trial-error-records.md`，对全文各章生效。）
- 过程记录脚注务必有据可查，不可杜撰。03ISim 的取材来源：`old/` 版本链 (`V1.3-FAIL`、`*-PATCH`、`results_*_v2`/`results_tsp_compare_final` 重跑目录)、各版本间的配置漂移、以及会话 transcript (`~/.claude/projects/<key>/*.jsonl` 可直接读)。
- 第3章已落地的过程脚注：`[^process-tsp-param]`、`[^process-g14-budget]`、`[^process-factor-bits]`、`[^process-factor-betaf]`、`[^process-tsp-correction]` 共5处，分布于Max-Cut/因子分解/TSP三族，密度已足够，勿再加。
- 第3章可复用素材：$\mathrm{G14}$扫数预算从$10^4$扩到$10^5$才首次命中BKS；$\beta_f=2$体现冷却不足、$\beta_f=50$体现过冷退化；$M=51$的$b_q=4/5/6/7$扫描支撑最小覆盖比特分配；因子分解$\beta_f$从$20$提到$30$以消除$r_c<1$ (V1.3配置为佐证)；TSP的$A\text{-margin}$扫描说明惩罚调参不能解决QUBO可行域稀疏，后续改用置换空间簇更新。
- 尚未成脚注、如需可备用的真实试错素材：TSP基准实例从 V1.3 的 `berlin52`/`eil51` ($n\approx52$，$N_{\text{spin}}=2704$，QUBO完全不可行) 退到 `burma14`/`ulysses16`/`gr17` 并加 $n\le20$ 硬限。当前判断为“用力过猛”风险，未加；若审稿仍嫌不足再补一处即可。
