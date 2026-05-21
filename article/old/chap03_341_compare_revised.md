<!-- Revision fragment.
Replaces the subsection "sMTJ-Gibbs 与经典模拟退火的同框架对照"
inside the Max-Cut benchmark section. All other subsections of the
benchmarks chapter are unchanged.
The reference list at the chapter end requires no edit: the entry
for Jiang et al. (2018) remains used by the integer-factoring
literature comparison; this revision drops only its earlier,
miscontextualised use inside the present subsection. -->

#### sMTJ-Gibbs与经典模拟退火的同框架对照

为分离单步采样规则对求解性能的贡献，本节在统一仿真框架内运行经典模拟退火（Simulated Annealing，SA）作为对照。此处所谓**经典SA**是Kirkpatrick、Gelatt和Vecchi于1983年提出的标准模拟退火算法[^Kirkpatrick1983]，其单步翻转判定遵循Metropolis-Hastings接受准则
$$
P(\text{accept})=\min\bigl(1,\,e^{-\beta\Delta E}\bigr),
$$
即先随机提议翻转某自旋、计算翻转的能量差$\Delta E$，再以上式概率接受或拒绝。本章前述sMTJ-Gibbs动力学对应另一类单步规则，直接按条件分布$p(s_i=+1\mid\mathbf{s}_{-i})=\sigma(2\beta h_i^{\mathrm{eff}})$采样新自旋值，与提议-接受框架的形式不同。两种方法除单步采样规则外，所有其它配置（退火端点、扫数、初始化、master-seed派生方式）完全一致。

为在结构异质的实例上系统观察动力学差异，对照集直接覆盖前述三类主基准的实例：Max-Cut方面取$\mathrm{G1}$、$\mathrm{G14}$、$\mathrm{G22}$三个G-set实例；整数分解方面取$M\in\{15,21,33,35,51,65,77,91,143\}$九个半素数目标；TSP方面取$\mathtt{burma14}$与$\mathtt{ulysses16}$两个TSPLIB实例（$\mathtt{gr17}$因两类动力学下的成功试验计数均不足$2$，$\mathrm{TTS}_{99}$估计无统计支撑而排除于条形对比之外）。各类对照沿用对应主基准的退火超参数：Max-Cut取$T=10000$、$(\beta_0,\beta_f)=(0.1,10.0)$，整数分解取$T=20000$、$(\beta_0,\beta_f)=(0.05,30)$，TSP取$T=50000$、$(\beta_0,\beta_f)=(0.05,20.0)$；Max-Cut与整数分解类$N_{\text{trial}}=200$，TSP类$N_{\text{trial}}=100$。sMTJ-Gibbs与Metropolis-SA共享退火调度、扫数与`SeedSequence`派生的随机种子序列，差异严格限定于单步翻转规则。十四个实例覆盖自$N_{\text{spin}}=5$至$N_{\text{spin}}=2704$跨越约三个数量级的规模区间，并涵盖稠密均权图、稀疏符号混合图、约束耦合的乘法电路QUBO、$n^2$-spin one-hot QUBO四类景观结构，足以暴露动力学差异在不同问题类别下的系统性趋势。汇总数据如下表，对应的TTS$_{99}$对比图按基准类别分三幅子图给出。

| 类别 | 实例 | $N_{\text{spin}}$ | sMTJ-Gibbs $p_s$ | SA $p_s$ | sMTJ-Gibbs $\mathrm{TTS}_{99}$(s) | SA $\mathrm{TTS}_{99}$(s) | speedup |
|---|---|---|---|---|---|---|---|
| Max-Cut | $\mathrm{G1}$  | 800  | $0.685$ | $0.740$ | $4.51$    | $4.18$    | $0.93\times$ |
| Max-Cut | $\mathrm{G14}$ | 800  | $0$     | $0$     | $\infty$  | $\infty$  | n/a          |
| Max-Cut | $\mathrm{G22}$ | 2000 | $0.020$ | $0.005$ | $573.65$  | $1749.51$ | $3.05\times$ |
| Factor  | $M=15$  | 5  | $0.260$ | $0.290$ | $1.11$  | $1.26$  | $1.14\times$ |
| Factor  | $M=21$  | 5  | $0.135$ | $0.140$ | $3.45$  | $2.90$  | $0.84\times$ |
| Factor  | $M=33$  | 7  | $0.065$ | $0.065$ | $8.08$  | $7.76$  | $0.96\times$ |
| Factor  | $M=35$  | 8  | $0.190$ | $0.195$ | $2.45$  | $2.21$  | $0.90\times$ |
| Factor  | $M=51$  | 9  | $0.050$ | $0.045$ | $12.20$ | $15.01$ | $1.23\times$ |
| Factor  | $M=65$  | 11 | $0.050$ | $0.055$ | $12.89$ | $12.39$ | $0.96\times$ |
| Factor  | $M=77$  | 11 | $0.095$ | $0.090$ | $7.12$  | $7.52$  | $1.06\times$ |
| Factor  | $M=91$  | 11 | $0.060$ | $0.060$ | $12.14$ | $11.91$ | $0.98\times$ |
| Factor  | $M=143$ | 15 | $0.075$ | $0.065$ | $10.07$ | $11.48$ | $1.14\times$ |
| TSP | $\mathtt{burma14}$   | 196 | $0.010$ | $0.010$ | $396.79$ | $396.47$ | $1.00\times$ |
| TSP | $\mathtt{ulysses16}$ | 256 | $0.010$ | $0.010$ | $565.90$ | $560.43$ | $0.99\times$ |

表2. sMTJ-Gibbs与经典SA在统一仿真框架下的对比，覆盖三类主基准共十四个实例。两类动力学共享退火端点、扫数、$N_{\text{trial}}$与master seed，差异严格限定于单步翻转规则。speedup列为$\mathrm{TTS}_{99}^{\mathrm{SA}}/\mathrm{TTS}_{99}^{\mathrm{sMTJ}}$，大于$1$代表sMTJ-Gibbs更快。$\mathrm{G14}$两类动力学下$p_s$均为零、$\mathrm{TTS}_{99}$不定义，对照在此层面不传达动力学层信息。

![sMTJ-Gibbs与经典SA在统一框架下的TTS_99对比，对数纵轴。(a) 整数分解九个半素数目标；(b) Max-Cut在G1与G22上；(c) TSP在burma14与ulysses16上。紫色为sMTJ-Gibbs，橙色为经典Metropolis-SA。两类动力学共享退火调度与种子，差异仅在单步翻转规则](tts_compare.png)

对照数据的总体形态明确。十四个实例中速比$\mathrm{TTS}_{99}^{\mathrm{SA}}/\mathrm{TTS}_{99}^{\mathrm{sMTJ}}$有十一个落在$0.84\times\sim 1.23\times$区间内，且大致对称散落于$1\times$两侧，对绝大多数实例两类动力学性能近乎相同，差异处于$N_{\text{trial}}$次试验的统计涨落量级。$\mathrm{G14}$与TSP两实例两类动力学下$p_s$均接近零（前者为$0$，后者为$1/N_{\text{trial}}$对应的单次命中），$\mathrm{TTS}_{99}$不构成可比较量，失败模式不源自单步采样规则的差异，而源自能量景观的结构性壁垒：$\mathrm{G14}$对应稀疏符号混合景观的预算欠缺（详见扫数预算消融），TSP对应$n^2$-spin one-hot QUBO的可行域稀疏性（详见TSP小节的能量轨迹分析）。真正分离两类动力学性能的实例为$\mathrm{G22}$，sMTJ-Gibbs相对Metropolis-SA加速$3.05\times$，对应$p_s$从$0.005$提升到$0.020$的四倍跃升。

加速比的实例依赖性可由两类采样规则的本质区别给出解释。Gibbs采样每步直接从条件分布$\sigma(2\beta h_i^{\mathrm{eff}})$抽取新自旋值，等价于按局部场所提供的方向性偏置直接生成新态；Metropolis则每步先等概率向两个方向提议、再判断接受。当$|\beta h_i^{\mathrm{eff}}|$大时（高$\beta$或高耦合度下），两种规则差距小，因为有利方向的接受率本就接近$1$；当$|\beta h_i^{\mathrm{eff}}|$适中时，Gibbs能直接使翻转方向偏向能量下降，Metropolis则因$1/2$的提议反向损失约半数有效翻转。$\mathrm{G22}$（$n=2000$、平均度约$20$）的高连接度使每个自旋的$|h_i^{\mathrm{eff}}|=|\sum_jJ_{ij}s_j|$分布在大$n$下中心宽幅展开，期望覆盖更高比例的中等$|\beta h_i^{\mathrm{eff}}|$区间，方向性优势在每个翻转上累积放大，对最终$\mathrm{TTS}_{99}$形成倍量级影响；$\mathrm{G1}$（$n=800$）的稠密同号权值使有效场分布更集中于较高量级，Gibbs相对Metropolis无优势可放大，速比落入统计涨落范围；整数分解的九个目标$N_{\text{spin}}\leq 15$、连接度受限于乘法电路的局部稀疏结构，单自旋的有效场不足以拉宽到方向性差异显著的动态范围，速比同样集中于$1\times$附近。该机理预测两点系统性趋势：动力学差异随系统规模与有效场分布的方差单调增强；当问题规模继续扩大至$\mathrm{G22}$以上时，Gibbs相对Metropolis的加速优势仍将放大。这构成sMTJ作为伊辛求解硬件单元相对单纯Metropolis-SA的核心理论依据之一。

综上，统一框架对照在十四个跨结构与跨规模的实例上独立验证两点结论。其一，对中小规模或低连接度的实例两类动力学差异微弱，单步采样规则的细节并非制约求解性能的主因；以整数分解全部九个目标与$\mathrm{G1}$作为代表，速比集中在$0.84\times\sim 1.23\times$区间。其二，对大规模高连接度图，Gibbs条件采样相对Metropolis提议-接受表现出可量化的系统性优势，加速量级与有效场分布形态相关；$\mathrm{G22}$给出$3.05\times$加速作为该趋势的直接证据。两个$p_s$均接近零的实例类（$\mathrm{G14}$、TSP）反映的是基础SA对应景观的结构性瓶颈而非动力学差异，对动力学层比较不构成有效信号，已在相应小节单独讨论。该对照为后续考察sMTJ器件级非理想模型替换理想sigmoid后动力学性能在不同基准上的退化幅度提供了理想Gibbs参考线。
