# 领域脉络说明：sMTJ伊辛机方法论的沿革与Si等2024的承继关系

本说明是独立文档，不进入论文正文，服务于审稿意见回复的起草；与[ref_si2024_comparison.md](ref_si2024_comparison.md) (逐维度不同点对照) 配套使用。核心论点有二：该领域的方法论组件在Si等2024之前即已由经典文献定型并完全公开；Si等2024本身也大量沿用这套方法论 (除器件条件概率一行外，均可由其文内引用逐项印证，见第二节表注)，其增量在系统工程与应用层。本文的文献基础取自同一批上游经典 (均在第三章参考文献列表内)，与该文不存在方法论上的承继关系。

## 一、方法论在该文之前已定型：分层时间线

**问题映射层 (2014年前定型)。**组合优化问题到伊辛/QUBO形式的系统映射手册是Lucas 2014，TSP的$n^2$ one-hot编码与双重约束惩罚项即出自该文的标准形式；模拟退火本身是Kirkpatrick 1983。这两篇均为第三章的基础引文 (`[^Lucas2014]`、`[^Kirkpatrick1983]`)。

**器件层：sMTJ作随机比特 (2015～2017定型)。**低势垒纳磁体热涨落的物理刻画见Chaves-O'Flynn等2015；"p-bit"的形式化 (双稳随机单元的sigmoid偏置响应$p=\sigma(2\beta h)$与互联网络方程) 由Camsari等2017年在PRX提出，2019年成为综述级标准框架 (第三章引文`[^Camsari2019]`)；用随机纳磁体网络做本征优化、含16城TSP的SPICE模拟退火仿真，是Sutton等2017——与Si等2024同一方法在仿真层的先行完整版本，早七年。

**系统层：实验架构范式 (2019定型)。**Borders等2019 (Nature，第三章引文`[^Borders2019]`) 给出sMTJ单元加比较器加DAC加微控制器数字计算局部场的异步概率计算机，8器件实验分解整数至945。这确立了此后所有sMTJ概率计算实验的系统范式：随机性由器件产生，耦合矩阵与局部场在数字域计算。规模化方向由Aadit等2022 (Nat Electron，第三章引文`[^Aadit2022]`) 推进到数千p-bit的稀疏伊辛机。

**分解层：有限硬件解大问题 (2008～2020已有)。**硬件容量不足时做图嵌入/问题分解是量子退火时代的标准操作 (Choi 2008的minor embedding)；专门针对伊辛机TSP的聚类分解已有Dan等2020 (DAC)——Si等2024的图4b自己就以该文 (其文献39) 作为对照基线之一。

**平台谱系与横向对比惯例。**CMOS退火芯片2015～2016年即达2万自旋 (Yamaoka等)；相干伊辛机2016年达100/2000节点 (McMahon等、Inagaki等，Science)；忆阻Hopfield网络 (Cai等2020) 与相变纳米振荡器 (Dutta等2021) 各有演示；纳秒保持时间的面内sMTJ单器件2021年已报道 (Hayakawa等、Safranski等)。2022年该领域已有Nature Reviews Physics综述 (Mohseni等，第三章引文`[^MohseniReview2022]`)。Si等2024的Table 1六平台横向对比正是这一成熟领域惯例的延续。

## 二、Si等2024对既有方法论的沿用：按其自身引用逐项对照

下表左列为方法论组件，中列为该文的使用方式，右列为承继来源；其中“其文献N”指Si等2024文内的引用编号，即该文自己标注的出处，可直接核验。

| 方法论组件 | Si等2024中的使用 | 承继来源 |
|---|---|---|
| TSP伊辛映射 ($n^2$ one-hot加双重约束) | 其式(5) | Lucas 2014 (其文献9，式(5)处显式标引) |
| 单自旋条件概率$p_\downarrow=1/(1+e^{-2\Lambda})$ | 其式(2) | p-bit标准方程 (Camsari等2017/2019谱系，见表注) |
| 低势垒sMTJ随机器件 (约50 nm，μs～ms保持) | 器件基础 | 其文自注"similar to previous studies⁴,³⁵" (Borders等2019、Chaves-O'Flynn等2015) |
| 系统架构 (器件加比较器加DAC加MCU数字局部场) | PCB系统 | Borders等2019同范式；其文自注采样模式"similar to an asynchronous probabilistic computer⁴" |
| 器件本征随机性省去RNG与Metropolis判决 | "intrinsic annealing"卖点 | Sutton等2017已论证 (其文献27，其引言以之标注16城SPICE退火仿真)，Borders等2019实验化 |
| 退火调度 (有效逆温度渐升) | 全局退火$c(t)$ | Kirkpatrick 1983一脉的标准SA调度 (其文献7、36为SA算法与收敛性) |
| TSP分解到有限硬件 | GP加CTSP | 聚类伊辛法Dan等2020为其图4b对照 (其文献39)；嵌入/分解思想见Choi 2008 (其文献32) |
| 跨平台横向对比表 | Table 1 | 领域惯例 (综述见Mohseni等2022；各平台原始文献即其表内引文) |

注：式(2)一行是唯一的例外——该式在其文中未标引 (指向其补充材料Note 1)，其参考文献列表也不含Camsari 2017/2019两篇；将其归入p-bit谱系是领域公认出处层面的判断，其余各行均可由其文内标引直接核验。

## 三、该文的真实增量 (公允陈述)

按上表剥离承继部分后，该文的增量集中在五点。实验与算法层四点：80器件全连接的实验规模 (此前sMTJ实验规模为8器件)；CTSP形式化 (负距离约束项固定访问次序、不增加辅助自旋，属其算法贡献)；GP加CTSP滑窗流水线及70城TSP的完整实验演示；计算内核口径下0.64 mW、39 solutions/s/W的能效数据点。仿真层一点：MRAM交叉杆扩展提议 (1T1SMTJ加RSA架构、4 Kb规模40 nm CMOS仿真68 sol/s/W、200城约90%成功率均为仿真结果)。这些属于系统工程与应用层贡献，方法论组件均为公开沿用。这一判断与其审稿周期相印证：该文2022年6月投稿、2024年4月接收，在方法论定型的成熟领域，评审焦点在工程事实而非原理。

## 四、审稿回复口径建议 (可改写进回复信)

以下为可直接改写的回复段落草稿，视审稿意见的具体措辞取舍：

> 感谢审稿人指出Si等 (Nature Communications, 2024) 的工作，我们已在3.3.3节补充引用，并将其作为大规模TSP依赖问题分解的硬件实验代表。需要说明的是，本文的方法论基础取自该领域2014～2019年间定型的经典文献：伊辛映射采用Lucas (2014) 的标准形式，器件随机比特模型采用Camsari等 (2017/2019) 的p-bit框架，系统评估范式参照Borders等 (2019, Nature) 确立的sMTJ概率计算实验架构。按Si等文内引用，其工作沿用的正是同一批上游方法，其增量在系统工程层，即80器件全连接实验与GP-CTSP问题压缩算法。本文的贡献维度为算法、器件、电路三层闭环的评测方法学、采样规则的受控对照与跨平台能耗口径修正，与该文正交。修改稿在3.3.3节以引用注明其分解路线；如审稿人需要，我们可在回复中进一步提供两项工作的逐项对照。

使用注意：回复不宜贬低该文 (其CTSP形式化与实验规模是真实贡献)，重点放在三件事——已补引、方法论同源于公开经典、贡献维度正交。若审稿意见进一步质疑本文相对该文的新颖性，可引用[ref_si2024_comparison.md](ref_si2024_comparison.md)中的七维对照，尤其是该文未涉及的三项：采样规则的受控算法对照 (Gibbs零结果)、更新粒度对可解性的边界判定、外围口径的端到端修正 (约18.7×)。若审稿人追问p-bit概念的更早起源，可注明同组Behin-Aein等2016 (见引用清单) 为概念先导，该文尚无p-bit术语与sigmoid网络方程，形式化完成于Camsari等2017，两者不冲突。

## 引用清单 (含DOI，供回复信取用)

- Kirkpatrick S, Gelatt C D, Vecchi M P. Optimization by simulated annealing. Science, 1983, 220(4598): 671-680. DOI: [10.1126/science.220.4598.671](https://doi.org/10.1126/science.220.4598.671)
- Choi V. Minor-embedding in adiabatic quantum computation: I. The parameter setting problem. Quantum Information Processing, 2008, 7: 193-209. DOI: [10.1007/s11128-008-0082-9](https://doi.org/10.1007/s11128-008-0082-9)
- Lucas A. Ising formulations of many NP problems. Frontiers in Physics, 2014, 2: 5. DOI: [10.3389/fphy.2014.00005](https://doi.org/10.3389/fphy.2014.00005)
- Chaves-O'Flynn G D, Wolf G, Sun J Z, Kent A D. Thermal stability of magnetic states in circular thin-film nanomagnets with large perpendicular magnetic anisotropy. Physical Review Applied, 2015, 4: 024010. DOI: [10.1103/PhysRevApplied.4.024010](https://doi.org/10.1103/PhysRevApplied.4.024010)
- Behin-Aein B, Diep V, Datta S. A building block for hardware belief networks. Scientific Reports, 2016, 6: 29893. DOI: [10.1038/srep29893](https://doi.org/10.1038/srep29893)
- Yamaoka M, et al. A 20k-spin Ising chip to solve combinatorial optimization problems with CMOS annealing. IEEE Journal of Solid-State Circuits, 2016, 51(1): 303-309. DOI: [10.1109/JSSC.2015.2498601](https://doi.org/10.1109/JSSC.2015.2498601)
- McMahon P L, et al. A fully programmable 100-spin coherent Ising machine with all-to-all connections. Science, 2016, 354(6312): 614-617. DOI: [10.1126/science.aah5178](https://doi.org/10.1126/science.aah5178)
- Inagaki T, et al. A coherent Ising machine for 2000-node optimization problems. Science, 2016, 354(6312): 603-606. DOI: [10.1126/science.aah4243](https://doi.org/10.1126/science.aah4243)
- Camsari K Y, Faria R, Sutton B M, Datta S. Stochastic p-bits for invertible logic. Physical Review X, 2017, 7: 031014. DOI: [10.1103/PhysRevX.7.031014](https://doi.org/10.1103/PhysRevX.7.031014)
- Sutton B, Camsari K Y, Behin-Aein B, Datta S. Intrinsic optimization using stochastic nanomagnets. Scientific Reports, 2017, 7: 44370. DOI: [10.1038/srep44370](https://doi.org/10.1038/srep44370)
- Camsari K Y, Sutton B M, Datta S. p-bits for probabilistic spin logic. Applied Physics Reviews, 2019, 6: 011305. DOI: [10.1063/1.5055860](https://doi.org/10.1063/1.5055860)
- Borders W A, Pervaiz A Z, Fukami S, et al. Integer factorization using stochastic magnetic tunnel junctions. Nature, 2019, 573: 390-393. DOI: [10.1038/s41586-019-1557-9](https://doi.org/10.1038/s41586-019-1557-9)
- Cai F, et al. Power-efficient combinatorial optimization using intrinsic noise in memristor Hopfield neural networks. Nature Electronics, 2020, 3: 409-418. DOI: [10.1038/s41928-020-0436-6](https://doi.org/10.1038/s41928-020-0436-6)
- Dan A, Shimizu R, Nishikawa T, Bian S, Sato T. Clustering approach for solving traveling salesman problems via Ising model based solver. 2020 57th ACM/IEEE Design Automation Conference (DAC), 2020. DOI: [10.1109/DAC18072.2020.9218695](https://doi.org/10.1109/DAC18072.2020.9218695)
- Dutta S, et al. An Ising Hamiltonian solver based on coupled stochastic phase-transition nano-oscillators. Nature Electronics, 2021, 4: 502-512. DOI: [10.1038/s41928-021-00616-7](https://doi.org/10.1038/s41928-021-00616-7)
- Hayakawa K, et al. Nanosecond random telegraph noise in in-plane magnetic tunnel junctions. Physical Review Letters, 2021, 126: 117202. DOI: [10.1103/PhysRevLett.126.117202](https://doi.org/10.1103/PhysRevLett.126.117202)
- Safranski C, et al. Demonstration of nanosecond operation in stochastic magnetic tunnel junctions. Nano Letters, 2021, 21(5): 2040-2045. DOI: [10.1021/acs.nanolett.0c04652](https://doi.org/10.1021/acs.nanolett.0c04652)
- Aadit N A, et al. Massively parallel probabilistic computing with sparse Ising machines. Nature Electronics, 2022, 5: 460-468. DOI: [10.1038/s41928-022-00774-2](https://doi.org/10.1038/s41928-022-00774-2)
- Mohseni N, McMahon P L, Byrnes T. Ising machines as hardware solvers of combinatorial optimization problems. Nature Reviews Physics, 2022, 4: 363-379. DOI: [10.1038/s42254-022-00440-8](https://doi.org/10.1038/s42254-022-00440-8)
- Si J, Yang S, Cen Y, et al. Energy-efficient superparamagnetic Ising machine and its application to traveling salesman problems. Nature Communications, 2024, 15: 3457. DOI: [10.1038/s41467-024-47818-z](https://doi.org/10.1038/s41467-024-47818-z)

本说明中Si等2024的文内引用编号 (文献4、7、9、27、32、35、36、39) 与自注引文均取自原文PDF文本；第三章引文标记 (`[^…]`) 均存在于`article/chapter03.md`参考文献列表。
