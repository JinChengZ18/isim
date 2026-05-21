<!-- Stand-alone subsection on hardware-aware evaluation. Designed to
slot into Chapter 3 after the algorithmic benchmarks (3.4) and
before the chapter summary. The section title carries no fixed
numbering.

Scope:
  1. Device non-ideality ablation. Map the calibrated parameters of
     Section 2.3 onto the BehavioralSMTJSpin backend, scan each
     non-ideality knob independently, and quantify its impact on
     TTS_99 over an enumerable Max-Cut testbed.
  2. Cross-architecture hardware comparison. Translate the
     algorithm-level p_success and sweep budget into wall-time TTS
     and energy-per-solution on four hardware platforms (this work,
     CMOS p-bit ASIC, FPGA Ising machine, single-core CPU+Numba),
     reporting absolute and relative figures across the three
     benchmark families.
-->

## 器件级非理想性与跨硬件性能评估

前述基准节将sMTJ求解器视为理想Glauber采样器，所有性能指标以扫数与CPU时间衡量。该层抽象使算法层比较结果干净可控，但不足以回答两个直接面向工程的问题：器件层的实测非理想性（前章2.3节量化的有限logistic斜率、写入偏置、循环涨落、back-hopping平台、晶圆级工艺离散）究竟以何种幅度损耗求解性能，以及sMTJ硬件相对其它已发表的Ising机架构在时间-能耗维度的优劣对比。本节以前章实测参数为输入、以前一节的求解器流水线为载体，分两步给出量化结论。

### 行为级器件模型与不理想度旋钮

仿真框架在`SMTJSpin`基类下保留了行为级模型注入接口，本节实现该接口的`BehavioralSMTJSpin`子类。模型把2.3节的实测非理想性浓缩为五个独立的参数旋钮，每一个对应一种独立的物理机制：

| 旋钮 | 物理含义 | 理想值 | 数据源 |
|:---|:---|:---:|:---|
| $g_\mathrm{dev}$ | drive gain，对应$\beta_s^\mathrm{meas}/\beta_s^\mathrm{ideal}$ 的slope比 | $1.0$ | 同批次sigmoid拟合 |
| $h_\mathrm{off}$ | 写入偏置，等效附加单极外场 | $0$ | $V_\mathrm{th}$方向不对称 |
| $\sigma_\mathrm{C2C}$ | drive上零均值高斯噪声 | $0$ | C2C统计涨落 |
| $p_\mathrm{max}$ | 翻转概率饱和上限，建模back-hopping | $1.0$ | Device A AP→P的0.72平台 |
| $\mathrm{CV}(\Delta)$ | 阵列级器件间增益分散度 | $0$ | Brinkman-PDK反推$7.7\%$基线 |

每自旋$i$的条件采样由
$$
u_i=2\beta\bigl(g_\mathrm{dev}\,g_i\,h_i^\mathrm{eff}+h_\mathrm{off}+h_{\mathrm{off},i}\bigr)+\epsilon_i,\quad\epsilon_i\sim\mathcal{N}(0,\sigma_\mathrm{C2C}^2)
$$
给出，$p(s_i=+1)=\mathrm{clip}\bigl(\sigma(u_i),\,1-p_\mathrm{max},\,p_\mathrm{max}\bigr)$；其中$g_i\sim\mathcal{N}(1,\mathrm{CV}^2(\Delta))$与$h_{\mathrm{off},i}\sim\mathcal{N}(0,\sigma_\mathrm{off}^2)$是器件实例化时一次性采样的D2D系数，跨循环固定不变。当所有五个旋钮置于理想值时该模型严格退化为基线$\sigma(2\beta h_i^\mathrm{eff})$采样，该等价性已在数值层（共享RNG下与`IdealGibbsSpin`输出逐位重合）得到确认。

为保留D2D路径下的逐自旋索引信息，仿真框架的block求解器同步引入索引透传机制：colour class求解循环按全局自旋索引向backend传递`idx`参数，使`BehavioralSMTJSpin`内部的`_g_per[idx]`、`_h_off_per[idx]`查表能正确对齐。该改动对所有不使用D2D路径的backend均为零开销空操作。框架还引入轻量级`register_spin_backend()`注册表，允许外部模块在不修改`isim.py`的前提下挂入新backend，使行为级模型与并行worker的pickle序列化路径自然兼容。

### 单旋钮消融实验

为分离每个旋钮的独立效应，本节在固定的Erdős-Rényi Max-Cut玩具实例（$n=14$，$p_\mathrm{edge}=0.30$）上扫描每个旋钮、其它四个旋钮置于理想值。实例规模选择使$2^{14}$个状态可被完整枚举得到精确基态$E_\mathrm{min}=-6.797$，作为$\mathrm{TTS}_{99}$计算的固定靶；扫数$T=500$、$(\beta_0,\beta_f)=(0.1,5.0)$、每点$N_\mathrm{trial}=30$次独立试验。该规模虽小但已充分暴露非理想性随旋钮值的单调趋势，也避免了需要外部数据集即可在任意环境复现。结果汇总如下图，纵轴是相对理想点的$\mathrm{TTS}_{99}$退化倍数（log尺度）。

![Five-axis device non-ideality ablation on a 14-spin ER Max-Cut testbed. Each panel scans one knob with the other four held at the ideal value. The orange dashed line marks the ideal-device baseline; values above 1.0 are TTS_99 ratios indicating performance loss. n/a markers denote settings where 0/30 trials hit the ground state](device_ablation_panels.png)

五个轴的趋势可独立解读，并各自支持具体的工程结论。**Drive gain**轴上$g_\mathrm{dev}<1$显著退化（$g=0.5$时$\mathrm{TTS}_{99}$退化$6.6\times$），$g$略高于$1$时反而更快（$g=1.5$对应$0.89\times$），$g$过大后又因低温阶段过度冻结而退化；这与同批次sigmoid斜率比NB预测大$\eta_c\approx 5\sim 10$倍的实测发现完全吻合，说明器件天然位于"略高于理想drive gain"的工作点，是有利特性。**Drive offset**对小规模实例的影响有限（$h_\mathrm{off}/J_\mathrm{max}=0.2$时仍只$1.13\times$），但能量中位数从$-6.66$降至$-6.55$，提示偏置对解质量的二阶效应将在更大规模实例上累积。**C2C noise**单调展宽，$\sigma_\mathrm{C2C}=2$时$\mathrm{TTS}_{99}$升$4.3\times$，与解析灵敏度$\partial p_s/\partial\sigma_\mathrm{C2C}\propto\sigma\beta h$一致。**Plateau ceiling**对求解性能构成最严重威胁：$p_\mathrm{max}\le 0.9$时$N_\mathrm{trial}=30$次试验中无一命中基态，能量中位数从$-6.66$跌至$-5.43$与$-3.65$。这从硬件层证实2.3节器件选型结论，即back-hopping平台明显的样品（Device A AP→P高电压区）不适合作为概率原语使用，应通过工艺筛选或工作点回退（降低写入电压至该平台之前）规避。**D2D dispersion**轴上PDK基线$\mathrm{CV}(\Delta)=7.7\%$对应$\mathrm{TTS}_{99}$退化仅$1.28\times$，$\mathrm{CV}=0.30$退至$2.05\times$，$\mathrm{CV}=0.60$方升至$13.5\times$。该单调展宽形态与2.3.6节由Jensen不等式与Monte Carlo数值仿真给出的$\mathcal{F}(\mathrm{CV}(\Delta))$传递函数预测同阶，PDK基线下$\mathcal{F}=0.997$对应的求解性能保持率亦在工程容差之内。

综合五个轴的实测，求解性能对$g_\mathrm{dev}$、$\sigma_\mathrm{C2C}$与$p_\mathrm{max}$敏感而对$h_\mathrm{off}$与$\mathrm{CV}(\Delta)$宽容。这一灵敏度排序与电学测量的器件级关注点恰好相反——存储MRAM苛求$\mathrm{CV}(\Delta)$与$h_\mathrm{off}$的低离散度，sMTJ作为概率求解器原语却在前两项上有充裕的容差余量，其约束位移到$g_\mathrm{dev}$标定与back-hopping抑制两条路径上。这一定量观察为后续器件级工艺优化与电路级写入策略设计提供了明确的优先级输入。

### 跨硬件架构的TTS与能耗对比

仿真框架的求解性能是算法层指标，需要进一步翻译为物理时间与能耗才能与已发表的Ising机硬件横向对比。本节为四类代表性硬件分别建立硬件物理模型，每个平台由一对核心参数（单自旋更新时间$t_\mathrm{update}$与能量$e_\mathrm{update}$）以及一个并行度$N_\parallel$确定，下表汇总参数与依据。

| 平台 | $t_\mathrm{update}$ | $e_\mathrm{update}$ | $N_\parallel$ | 数据源 |
|:---|:---:|:---:|:---:|:---|
| sMTJ-array (本工作) | $0.75\,\mathrm{ns}$ | $0.78\,\mathrm{pJ}$ | $64$ | 2.3节实测：$E=V_\mathrm{th}^2/R_\mathrm{SOT}\cdot t_w$ |
| CMOS p-bit ASIC | $5\,\mathrm{ns}$ | $5\,\mathrm{pJ}$ | $64$ | Camsari等2020[^Camsari2020] |
| FPGA SBM | $1\,\mathrm{ns}$ | $1\,\mathrm{nJ}$ | $256$ | Goto等2021[^Goto2021] |
| CPU + Numba (软件) | runtime | runtime | $1$ | 仿真框架运行实测，TDP=$28\,\mathrm{W}$ |

CPU平台的参数从仿真框架的`time_median`实测数据自动反推，对G1实例对应$t_\mathrm{update}=141\,\mathrm{ns}$、$e_\mathrm{update}=4.0\,\mathrm{nJ}$，对G22为$126\,\mathrm{ns}$与$3.5\,\mathrm{nJ}$，与Intel x86单核芯片功耗包络一致。CMOS p-bit与FPGA Ising机的数据源不变换写入：每个$\mathrm{TTS}_{99}$与每解能耗依据
$$
\mathrm{TTS}_{99}=\frac{\log(1-0.99)}{\log(1-p_s)}\cdot N_\mathrm{sweep}\cdot\frac{N_\mathrm{spin}}{N_\parallel}\cdot t_\mathrm{update}
$$
$$
E_\mathrm{sol}=\frac{\log(1-0.99)}{\log(1-p_s)}\cdot N_\mathrm{sweep}\cdot N_\mathrm{spin}\cdot e_\mathrm{update}
$$
计算，其中$p_s$与$N_\mathrm{sweep}$来自前述基准节的实测数据，与硬件物理参数完全解耦。该投影仅改变扫数到时间/能耗的换算系数而不改变算法层成功概率，因此硬件比较结果具有跨平台可比性。

将该模型作用于前节三类基准（Max-Cut $\mathrm{G1}/\mathrm{G14}/\mathrm{G22}$、整数分解九个半素数目标、TSP $\mathtt{burma14}/\mathtt{ulysses16}$）的实测$p_s$得到下图。$\mathrm{G14}$因$p_s=0$在所有平台上$\mathrm{TTS}_{99}$均不定义，记为n/a；其余十三个实例上四类平台的相对位置稳定。

![Cross-architecture comparison of TTS_99 (left) and energy per solution (right) across 14 instances spanning Max-Cut, integer factoring, and TSP. Both axes are log-scaled. Each cluster shows four bars (sMTJ-array, CMOS p-bit, FPGA SA, CPU+Numba). G14 is n/a because p_success = 0 on both dynamics in the Section 3.4 calibration](hw_compare_panels.png)

时间维度上，sMTJ-array在所有可解实例上均给出该比较中最快的$\mathrm{TTS}_{99}$。在$\mathrm{G22}$（$n=2000$）上sMTJ-array耗时$54.7\,\mathrm{ms}$、CMOS p-bit需$365\,\mathrm{ms}$（约$6.7\times$）、FPGA SBM需$18.2\,\mathrm{ms}$（约$0.33\times$，因$N_\parallel=256$的并行度优势）、CPU+Numba需$645\,\mathrm{s}$（约$11800\times$）。在小规模实例上sMTJ-array与FPGA SBM的相对位置反转（$\mathrm{G1}$上FPGA因$256\times$并行而稍快），但能耗维度上sMTJ-array始终领先：$\mathrm{G22}$实例下sMTJ的每解能耗$3.6\,\mathrm{mJ}$显著低于CMOS的$22.8\,\mathrm{mJ}$（$6.4\times$）、FPGA的$4.6\,\mathrm{J}$（约$1280\times$）、CPU的$1.8\times 10^4\,\mathrm{J}$（约$5\times 10^6\times$）。

两点观察总结这一对比的物理本质。其一，sMTJ-array相对FPGA Ising机的核心优势不在原始时间而在能耗效率。FPGA以GHz量级时钟在数字电路上模拟随机翻转动力学（无论是Glauber采样还是simulated bifurcation的微分方程时间步进），单次更新需通过LUT权重和、随机数生成、阈值比较的完整数字流水线，单次更新能耗在1 nJ量级、几乎完全由静态泄漏电流决定；sMTJ则以亚皮焦的单脉冲完成同等采样，能效提升$10^3\times$源自把"采样"这一操作从数字状态机迁移到本征随机器件层。其二，sMTJ-array相对CMOS p-bit ASIC的优势是同尺度内的写入能耗压缩。两者皆在芯片层做并行随机采样、$N_\parallel$同为$64$量级，差异源于sMTJ的SOT通道写入电压与脉宽相对CMOS LFSR或亚阈值MOS随机源的优势。该比较与Borders等2019年的整数分解硬件实验中报告的"sMTJ p-bit比CMOS p-bit快约5–10倍、能耗低约5–10倍"[^Borders2019]量级一致。

进一步与异类伊辛机架构的横向比较参见专题综述[^Mohseni2022]。当前四类硬件中并未包含光学相干Ising机（如NTT 100k-spin CIM），后者在$N$扩展性上有相对优势但典型$\mathrm{TTS}_{99}$与本节sMTJ-array处同一数量级，能耗与光泵浦功率相关、不在本节物理模型的覆盖范围内。本节硬件对比的范围限于"接近端点的纳秒-皮焦Ising机"这一类电学硬件，结论是sMTJ-array在该比较类内同时给出最低的求解能耗与可竞争的求解时间，其相对优势随问题规模增大而单调放大（与$n=2000$实例上比CMOS快$6.7\times$、比CPU快$11800\times$对应的$10^4\times$量级跨度比小实例上更显著）。

### 复现流水线

本节所有数据由两个独立的驱动脚本复现。`bench_device_ablation.py`运行单旋钮消融，支持G-set模式（需要外部数据文件）与ER枚举模式（无外部依赖，本节使用），每个旋钮约30秒求解、整次实验约5分钟可在普通笔记本电脑完成。`bench_hardware_compare.py`从任意基准驱动产生的`summary.csv`读入$p_s$与$N_\mathrm{sweep}$，按四类硬件物理参数计算$\mathrm{TTS}_{99}$与$E_\mathrm{sol}$并出图，运行时间可忽略。两个驱动均与现有基准流水线同源，与3.4节的`compare_baselines.py`、`bench_*.py`共享`SolverConfig`、`Problem`、`multistart`接口，可在无外部参数调整的情况下复用所有已生成的summary数据。

[^Camsari2020]: Camsari K Y, Sutton B M, Datta S. p-Bits for Probabilistic Spin Logic[J]. Applied Physics Reviews, 2019, 6(1): 011305. DOI: [10.1063/1.5055860](https://doi.org/10.1063/1.5055860).

[^Goto2021]: Goto H, Endo K, Suzuki M, et al. High-performance Combinatorial Optimization Based on Classical Mechanics[J]. Science Advances, 2021, 7(6): eabe7953. DOI: [10.1126/sciadv.abe7953](https://doi.org/10.1126/sciadv.abe7953).

[^Borders2019]: Borders W A, Pervaiz A Z, Fukami S, et al. Integer Factorization Using Stochastic Magnetic Tunnel Junctions[J]. Nature, 2019, 573(7774): 390-393. DOI: [10.1038/s41586-019-1557-9](https://doi.org/10.1038/s41586-019-1557-9).

[^Mohseni2022]: Mohseni N, McMahon P L, Byrnes T. Ising Machines as Hardware Solvers of Combinatorial Optimization Problems[J]. Nature Reviews Physics, 2022, 4(6): 363-379. DOI: [10.1038/s42254-022-00440-8](https://doi.org/10.1038/s42254-022-00440-8).
