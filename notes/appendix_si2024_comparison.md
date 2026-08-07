<!-- 配套的 .docx 已由用户在WPS中手工排版 (重排标题、删去第7节与对照表末行、调整公式)，与本文件已不同步。
     追加内容时请解包改 word/document.xml 后重新打包 (复用样式 5=小节标题、45=正文、40=Hyperlink字符样式)，
     切勿用 pandoc 从本文件重新生成 docx——那会覆盖用户的排版。 -->

# 附录：本文与Si等 (2024) 工作的逐项对比

对比对象：Si J, Yang S, Cen Y, et al. Energy-efficient superparamagnetic Ising machine and its application to traveling salesman problems. Nature Communications, 2024, 15: 3457. DOI: 10.1038/s41467-024-47818-z。以下“该文”指此论文，“本文”指本论文 (章节号按修改稿)。

## 1. 工作性质：系统实验演示 对 跨层评估方法

该文的贡献是一台可运行的硬件系统加一套问题压缩算法：80器件PCB实验、GP-CTSP分解、交叉杆架构提议，主张以有限硬件低功耗求解大规模问题。本文的贡献是可复现的跨层评估方法与设计规则：同一组实例、退火协议与随机种子贯通算法、器件、电路三层，输出的是灵敏度排序、归一化容差与口径修正 (3.6节明确“不是已经完成的阵列芯片性能”)。两者不构成同类竞争：该文回答“这套系统能做到什么”，本文回答“这类系统的性能由哪些因素决定、各留多少裕度”。

## 2. 器件与随机性的使用方式：直流自由涨落 对 纳秒脉冲概率写入

该文器件是两端垂直sMTJ，在直流偏置下持续热涨落，以0.1 ms量级的保持时间为节拍被动读出，随机性以“连续涨落加慢读出”的方式进入系统，主频被保持时间锁定在10 kHz。本文器件是第二章实测标定的SOT脉冲写入型sMTJ (写支路$R_{SOT}=776\,\Omega$，$V_{th}=0.896$ V，$V_T=23.4$ mV)：每次更新是“复位加0.75 ns概率写”的同步时钟化事件，随机性以纳秒脉冲伯努利采样的方式进入，更新周期由外围建立时间主导 (约14 ns)。两种机制的速度-能量包络相差数个数量级：按该文实验口径折算，每自旋每迭代约0.8 nJ (0.64 mW÷10 kHz÷80)，而本文端到端口径为每更新14.7 pJ ($k=3$、功率门控)。工作点不同，两组能量数字不可直接比较。

## 3. 退火与采样规则：本征退火叙事 对 受控算法对照

该文把优势归因于器件本征随机性省去随机数生成与Metropolis判决的实现资源，属能效与资源叙事，未在同一框架内对照不同采样规则的算法效率。本文3.3.1节的受控对照 (共享调度、扫数与种子，只换单步规则) 给出明确的算法层零结果：Gibbs条件采样相对Metropolis-SA没有可分辨加速，G1上还有与Peskun排序方向一致的小幅劣势，G22的表观3.71×加速经统计功效审计后不成立。两个结论相容但层次不同：器件省去随机数生成的资源收益是实现层的，本文证明它不叠加算法层加速，因此收益成立与否完全取决于外围开销的口径核算 (见第5条)。

## 4. TSP路线：问题侧分解 对 更新粒度改造

绕开$n^2$自旋QUBO规模爆炸，两文走了正交的两条路。该文保持逐自旋条件采样的硬件不动，在问题侧做GP图分割与CTSP滑窗，把大问题切成80节点以内的子问题序列；其9城 (81自旋) 实验以逐自旋独立采样的同步并行更新 (每迭代所有自旋按各自条件概率同时重采样，非簇更新) 在硬件上收敛到最优。本文保持问题不动，在求解器侧把更新粒度从单自旋换成置换空间簇更新 (2-opt/swap/insert)，把burma14/ulysses16/gr17从$p_s=0$推到$p_s=1.00$ (扫数还降一个数量级)，并证明惩罚系数调参 ($A$-margin 1.2～10) 不能改变单自旋QUBO的结构边界。两条路线可组合：GP-CTSP切出的子问题仍可用簇更新求解。规模口径也不同：该文硬件收敛的实例在81自旋 (9城)，本文单自旋失败的最小实例是196自旋 (14城)，两者不矛盾，恰与本文对可行域测度极端稀疏 ($n!/2^{n^2}$，$n=14$时约$10^{-51}$) 的判断一致；该文仅在仿真中给出无GP/CTSP方法成功率随城市数下降的曲线 (其Fig. 5c，512个sMTJ)，未界定不分解方案的最小失效实例及其机理，本文给出了这个边界及其机理 (约束项冻结后可行盆地间不可迁移)。另外该文的CTSP由微控制器在问题矩阵层实现，不是多自旋原子更新；本文3.6节“TSP簇更新尚无对应的晶体管级协同更新电路”的判断不因该文而改变。

## 5. 能耗数字的口径：计算内核口径 对 端到端口径修正

该文Table 1明确声明只计伊辛计算部分 (0.64 mW为"main computer kernel"实验值)，且10 kHz主频下外围静态功耗的摊销压力小。本文的核心发现之一恰是口径问题：器件级0.78 pJ/更新的能量优势，被外围驱动静态功耗与建立时间放大约18.7× (tt角；含H桥复位实测后约19.7×)，局部场数字求和一项 (平均度20～48时7.3～17.6 pJ) 就超过单元能量本身，结论是“任何跨平台能耗结论都必须与口径边界一并陈述” (3.5.4节)。用这一尺度读该文Table 1：各平台数字取自不同文献、不同口径 (表脚注自注多处为估计或仿真外推)，其39 solutions/s/W的排名意义受同样的口径限制。这是本文方法学对这类横向对比表的普遍修正，同样适用于该文Table 1。

## 6. 阵列化分析的层次：架构提议 对 定量瓶颈清单

该文提出1T1SMTJ交叉杆加读出放大器的架构，仿真到4 Kb规模，定性说明同步设计可缓解泄漏、潜行电流与寄生电阻，并指出器件涨落相位 (其时序中的PH1) 的时长需与保持时间可比、从而限制主频。本文对同类阵列给出定量的瓶颈清单：写线IR压降的逐行预畸变补偿及其与逐单元阈值校准的码域预算冲突 (3.5.3节)、误读容限随规模按$N\cdot T$标度收紧至$10^3$自旋下的$10^{-5}$量级，而实测读出比较器失配$\sigma_{off}=18.5$ mV ($0.79V_T$) 对应的误读率 (乐观口径$8.2\times10^{-4}$) 已远超该容限 (3.5.5节)、复位back-hopping的$k\approx2\sim3$工程折中 (3.5.2节)、写入量程须随$N\cdot T$放宽到$\pm10V_T$以上 (3.5.2节)。值得注意的汇合点：该文交叉杆同样采用行序更新，与本文由共享列线IR耦合推出的“更新须按行顺序”结论一致，本文给出了该调度的物理必要性论证。

## 7. 统计与成功判据

该文报告成功概率 (中位数加四分位距)、解质量 (相对最佳演示解定义) 与固定迭代数口径的求解时间/能量，200城“90%成功率”的成功定义是达到95%解质量而非命中最优。本文统一采用$p_s$加Wilson区间、$\mathrm{TTS}_{99}$加自举区间，并做统计功效审计 (低命中数的表观加速被扩容重跑否定)。对比两文数字时须注意两边的成功判据不同，不能把“95%质量成功率”与本文“1%容差命中率”直接并列。

## 对照表

| 维度 | Si等2024 | 本文 |
|---|---|---|
| 成果形态 | 80-sMTJ PCB系统实验+GP-CTSP算法+交叉杆提议 | 跨层仿真评估方法+设计规则 (非芯片实现) |
| 器件 | 垂直两端sMTJ，直流偏置自由涨落，保持约0.1 ms | SOT脉冲写入sMTJ，0.75 ns概率写 (第二章实测标定) |
| 更新节拍 | 10 kHz (保持时间锁定) | 约14 ns周期投影 (外围建立主导) |
| 退火 | 电流幅值承载有效逆温度$c$，本征随机性省随机数生成 | 几何$\beta$调度经$V_{wr}=V_{th}+uV_T$映射；量程/位宽/复位逐项量化 |
| 采样规则评价 | 未做同框架算法对照 | Gibbs对Metropolis受控对照：无算法加速 |
| TSP路线 | 问题侧GP+CTSP分解，80节点解70城 (实验) | 求解器侧置换簇更新；单自旋QUBO界定为下界 ($n\leq20$上限) |
| 能耗口径 | 计算内核0.64 mW、39 sol/s/W (外围不计) | 端到端口径修正约18.7×；突触求和项实测计入 |
| 阵列约束 | 架构级：串扰可忽略、涨落相位PH1限频 | 电路级：IR压降/校准码域/读出失配/复位回跳定量清单 |
| 统计协议 | 成功概率+解质量 (95%质量判据) | $p_s$ (Wilson)+$\mathrm{TTS}_{99}$ (自举)+功效审计 |

## 信与附录所引文献

以下为说明信与本附录提及的全部文献。第1至10条为信中所列新增引文，第11至14条为信中提及的原有引文，第15条为本附录的对比对象，第16条为本附录第3节提及的Peskun排序的出处。

1. Metropolis N, Rosenbluth A W, Rosenbluth M N, et al. Equation of state calculations by fast computing machines[J]. The Journal of Chemical Physics, 1953, 21(6): 1087-1092. DOI: [10.1063/1.1699114](https://doi.org/10.1063/1.1699114).
2. Glauber R J. Time-dependent statistics of the Ising model[J]. Journal of Mathematical Physics, 1963, 4(2): 294-307. DOI: [10.1063/1.1703954](https://doi.org/10.1063/1.1703954).
3. Geman S, Geman D. Stochastic relaxation, Gibbs distributions, and the Bayesian restoration of images[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 1984, PAMI-6(6): 721-741. DOI: [10.1109/TPAMI.1984.4767596](https://doi.org/10.1109/TPAMI.1984.4767596).
4. Camsari K Y, Faria R, Sutton B M, et al. Stochastic p-bits for invertible logic[J]. Physical Review X, 2017, 7(3): 031014. DOI: [10.1103/PhysRevX.7.031014](https://doi.org/10.1103/PhysRevX.7.031014).
5. Rønnow T F, Wang Z, Job J, et al. Defining and detecting quantum speedup[J]. Science, 2014, 345(6195): 420-424. DOI: [10.1126/science.1252319](https://doi.org/10.1126/science.1252319).
6. Wilson E B. Probable inference, the law of succession, and statistical inference[J]. Journal of the American Statistical Association, 1927, 22(158): 209-212. DOI: [10.1080/01621459.1927.10502953](https://doi.org/10.1080/01621459.1927.10502953).
7. Croes G A. A method for solving traveling-salesman problems[J]. Operations Research, 1958, 6(6): 791-812. DOI: [10.1287/opre.6.6.791](https://doi.org/10.1287/opre.6.6.791).
8. Lin S, Kernighan B W. An effective heuristic algorithm for the traveling-salesman problem[J]. Operations Research, 1973, 21(2): 498-516. DOI: [10.1287/opre.21.2.498](https://doi.org/10.1287/opre.21.2.498).
9. Pelgrom M J M, Duinmaijer A C J, Welbers A P G. Matching properties of MOS transistors[J]. IEEE Journal of Solid-State Circuits, 1989, 24(5): 1433-1439. DOI: [10.1109/JSSC.1989.572629](https://doi.org/10.1109/JSSC.1989.572629).
10. Razavi B. The StrongARM latch [A Circuit for All Seasons][J]. IEEE Solid-State Circuits Magazine, 2015, 7(2): 12-17. DOI: [10.1109/MSSC.2015.2418155](https://doi.org/10.1109/MSSC.2015.2418155).
11. Camsari K Y, Sutton B M, Datta S. p-bits for probabilistic spin logic[J]. Applied Physics Reviews, 2019, 6(1): 011305. DOI: [10.1063/1.5055860](https://doi.org/10.1063/1.5055860).
12. Lucas A. Ising formulations of many NP problems[J]. Frontiers in Physics, 2014, 2: 5. DOI: [10.3389/fphy.2014.00005](https://doi.org/10.3389/fphy.2014.00005).
13. Kirkpatrick S, Gelatt C D, Vecchi M P. Optimization by simulated annealing[J]. Science, 1983, 220(4598): 671-680. DOI: [10.1126/science.220.4598.671](https://doi.org/10.1126/science.220.4598.671).
14. Borders W A, Pervaiz A Z, Fukami S, et al. Integer factorization using stochastic magnetic tunnel junctions[J]. Nature, 2019, 573(7774): 390-393. DOI: [10.1038/s41586-019-1557-9](https://doi.org/10.1038/s41586-019-1557-9).
15. Si J, Yang S, Cen Y, et al. Energy-efficient superparamagnetic Ising machine and its application to traveling salesman problems[J]. Nature Communications, 2024, 15: 3457. DOI: [10.1038/s41467-024-47818-z](https://doi.org/10.1038/s41467-024-47818-z).
16. Peskun P H. Optimum Monte-Carlo sampling using Markov chains[J]. Biometrika, 1973, 60(3): 607-612. DOI: [10.1093/biomet/60.3.607](https://doi.org/10.1093/biomet/60.3.607).
