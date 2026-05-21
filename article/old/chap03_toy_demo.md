<!-- Stand-alone subsection: end-to-end illustration of the Ising
annealing flow on a 5-node Max-Cut toy. Designed to slot in either
at the close of the Ising-formulation section (as a concrete
illustration of the abstract mappings) or at the head of the
benchmarks section (as a primer before the large-scale tables). The
section title carries no fixed numbering. -->

## 端到端图示：一个5节点Max-Cut玩具实例

前述伊辛形式化与求解器框架在大规模基准上易使读者只见数据而不见机理。本节用一个仅含5个自旋的Max-Cut玩具实例完整走通"问题→编码→采样→退火→解"五个环节，使读者得以在脑中建立Ising机求解组合优化问题的最小工作单元图景。该实例的规模选取使$2^5=32$个自旋构型可被完整枚举、能量景观可被完全刻画、退火轨迹可被原子级地观察，从而把后续基准节中所有以加速比与统计指标呈现的数据都还原为这一最小图景的规模化。

### 问题与编码

取5节点无向图$G=(V, E)$，节点编号$0\sim 4$，边集为
$$
E=\{(0,1),(0,2),(1,3),(2,3),(2,4),(3,4)\},
$$
共$|E|=6$条边，权值统一取$w_{ij}=1$。该图含三角形$2{\text -}3{\text -}4$，因此最大切分严格小于$|E|$。按伊辛形式化中给出的Max-Cut映射$J_{ij}=-w_{ij}/2$、$h_i\equiv 0$，相应哈密顿量化为
$$
H(\mathbf{s})=-\sum_{i<j}J_{ij}s_is_j=\frac{1}{2}\sum_{(i,j)\in E}s_is_j,
$$
其中$s_i\in\{-1,+1\}$。$32$个自旋构型可全部枚举，能量取且仅取$\{-2,-1,0,+1,+3\}$五个层级，简并度依次为$4, 6, 12, 8, 2$，基态能量$E_{\min}=-2$对应切数$\mathrm{cut}=5$（两侧划分$S=\{1,2\}$、$\bar S=\{0,3,4\}$或其$\pm$翻转对称版本）。最差构型为全$+1$或全$-1$，对应$\mathrm{cut}=0$、$E_{\max}=+3$。这一完整的能量层级如下图所示：

![Full enumeration of the toy energy landscape, sorted by H(s). The four ground states sit at the lower plateau (cut = 5), the two worst states at the upper extreme (cut = 0)](toy_energy_landscape.png)

完整的端到端流程框架如下图：左起依次为问题、伊辛编码、p-bit网络、退火调度、解，每段对应仿真框架中可单独识别的接口层。

![End-to-end Ising annealing flow on the toy Max-Cut. Five stages: problem → Ising encoding → p-bit network → annealing schedule → solution](ising_flow_schematic.png)

### sMTJ-Gibbs动力学的展开

将编码完成的$(J, h)$输入异步Gibbs采样器，每个扫描时刻按随机排列依次访问5个自旋，每次访问从条件分布
$$
p(s_i=+1\mid\mathbf{s}_{-i})=\sigma\bigl(2\beta\,h_i^{\mathrm{eff}}\bigr),\quad h_i^{\mathrm{eff}}=\sum_jJ_{ij}s_j,
$$
抽样新值。这正是sMTJ作为p-bit的物理工作模式：$h_i^{\mathrm{eff}}$以电流偏置的形式施加于自由层，外部温度对应热涨落幅度$1/\beta$，新的自旋值由器件随机切换得到。退火调度采用几何衰减$\beta(t)=\beta_0(\beta_f/\beta_0)^{t/T}$，端点取$(\beta_0,\beta_f)=(0.1,5.0)$，扫数$T=200$；以独立master seed派生8次试验。求解轨迹如下图所示：浅紫细线为单次试验、深紫粗线为$8$次试验的中位轨迹、橙色虚线为退火端点对齐的$\beta(t)$参考。

![Eight independent Gibbs annealing runs on the toy Max-Cut. Light traces are individual trials, the heavy line is the per-sweep median, and the dashed orange curve is the geometric beta(t) schedule on the right axis](toy_annealing_traces.png)

轨迹展现出退火过程的两个清晰区段。在前$\sim 50$个扫数（对应$\beta\lesssim 1$，温度$T=1/\beta\gtrsim 1$）轨迹在能量谱上充分混合，单次能量在$\{-2,-1,0,+1,+3\}$五个层级间频繁跳变，与玻尔兹曼分布$p(\mathbf{s})\propto e^{-\beta H(\mathbf{s})}$在高温下的近似均匀采样一致。$\beta$越过约$2$之后翻转概率$\sigma(2\beta h_i^{\mathrm{eff}})$被局部场的符号近乎决定性地拉向能量下降方向，单次试验的能量逐步沉入$E\leq-1$区间；至$t\sim 150$后所有试验的能量稳定在$E_{\min}=-2$。最终8次试验全部命中基态，相应单次成功概率$p_s=1$。这正是基础异步Gibbs退火在小规模景观上的"教科书表现"——高温遍历、低温凝固、最终基态——而3.4节大规模基准上的所有$p_s<1$与$\mathrm{TTS}_{99}<\infty$的统计，都可被理解为同一机制在受能垒结构与扫数预算双重制约下的有限样本表现。

### 解的几何含义

最终自旋构型解码为图分区。基态$\mathbf{s}^*=(-,+,+,-,-)$对应划分$S=\{1,2\}$、$\bar S=\{0,3,4\}$，其切边集为$\{(0,1),(0,2),(1,3),(2,3),(2,4)\}$五条；唯一未切的边$(3,4)$位于三角形$2{\text -}3{\text -}4$内部，对应该三角形在任何二分中至少必有的一条同色边——这是Goemans-Williamson半定规划松弛给出$\le 0.87856$紧致界的几何根源[^Goemans1995]在$n=5$实例上的具体显化。最优分区如下图：

![Optimal partition of the toy graph. Dark nodes form set S, light nodes form complement; orange edges are cut, the single grey edge sits inside the triangle 2-3-4 and is the unavoidable uncut edge](toy_optimal_partition.png)

### 与大规模基准的关系

上述5自旋实例与后续Max-Cut基准节的G-set实例（$n=800\sim 2000$）在仿真框架的接口层上完全同构：均通过`Problem(name, n, J, h)`实例化、由`SolverConfig`配置退火端点与扫数、由`multistart`派生独立试验。规模放大带来的差异不在于流程，而在于状态空间从$2^5$增至$2^{2000}$后能量景观的层级数与浅极小密度均按指数膨胀，单步异步Gibbs在固定扫数预算下已不足以遍历高温阶段所需的有效区域，从而$p_s$从$1$退化至$10^{-1}\sim 10^{-2}$量级。3.4节给出的$p_s$、$\mathrm{TTS}_{99}$、gap三层指标系统地刻画这一退化幅度；本节的玩具实例则提供了规模化前的"对照零点"——在没有任何统计意义损失的小规模实例上，伊辛求解器与几何退火的组合可以确定地以接近$1$的概率逼近基态。后续基准节关注的所有比较（动力学差异、扫数预算、退火端点、编码瓶颈）均可被理解为同一图景在更宽规模与更复杂景观上的展开。

本节涉及的全部数据由附带的`toy_demo_maxcut5.py`脚本独立复现，该脚本仅依赖NumPy与Matplotlib，对应代码量约$200$行；框架图由`schematic_flow.py`脚本生成。读者运行该脚本即可在本地复现本节四幅图的全部数值与图形结果。

[^Goemans1995]: Goemans M X, Williamson D P. Improved approximation algorithms for maximum cut and satisfiability problems using semidefinite programming[J]. Journal of the ACM, 1995, 42(6): 1115-1145. DOI: [10.1145/227683.227684](https://doi.org/10.1145/227683.227684).
