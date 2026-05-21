## 3.1 组合优化问题的伊辛形式化

组合优化问题的伊辛形式化是抽象问题与物理求解器之间的接口层。本节首先阐明伊辛模型作为组合优化求解器工作的统计物理基础，随后将Max-Cut、旅行商、整数分解以及其它典型NP-hard问题统一翻译为伊辛哈密顿量
$$
H(\mathbf{s})=-\sum_{i<j}J_{ij}s_is_j-\sum_ih_is_i,\quad s_i\in\{-1,+1\}
$$
的最小化问题，使后续求解器在接口上仅面对耦合矩阵$J$与外场向量$h$两个对象。

### 3.1.1 伊辛模型作为组合优化求解器的理论基础

伊辛模型源自统计物理对铁磁相变的描述，其哈密顿量在给定耦合矩阵$J$与外场$h$后定义了关于自旋构型$\mathbf{s}$的能量函数。处于温度$T$的热浴中时，系统取任一构型的概率服从玻尔兹曼分布
$$
p(\mathbf{s})=\frac{1}{Z}\exp[-\beta H(\mathbf{s})],\qquad Z=\sum_{\mathbf{s}}\exp[-\beta H(\mathbf{s})]
$$
其中$\beta=1/(k_BT)$为逆温度，$Z$为配分函数。当$\beta\to\infty$即温度趋于零时，分布的概率质量集中于使$H$取最小值的基态构型$\mathbf{s}^*$上。若一个组合优化问题被映射为伊辛哈密顿量且保证最优解对应基态，则原优化问题即归结为在低温极限下寻找玻尔兹曼分布的众数，这构成所有伊辛求解器共同的工作逻辑。

需要指出，任意伊辛模型的基态求解本身仍为NP-hard[^Barahona1982]：哈密顿能量在高维离散空间$\{-1,+1\}^N$上呈高度非凸景观，存在指数多的局部极小。若直接在零温极限下进行贪婪下降，系统常被局部极小所俘获而无法达到全局最优。Kirkpatrick等提出的模拟退火算法（simulated annealing，SA）[^Kirkpatrick1983]给出了一条合理路径：令系统在较高温度下充分遍历构型空间以跨越能量势垒，再以适当速率降低温度并持续采样，最终在低温极限下逼近基态。理论上当温度按$T(t)\propto 1/\log t$衰减时模拟退火依概率收敛至全局最优；工程上则普遍采用几何或线性退火以在解质量与求解时间之间取得折衷。本章将SA作为求解器内核的算法基础，并将其sMTJ物理实现（即p-bit SA）作为后续基准的核心评估对象。

在实现层面，玻尔兹曼分布的采样可通过马尔可夫链蒙特卡罗方法的单自旋更新获得。任一自旋$s_i$在给定其它自旋$\mathbf{s}_{\neq i}$时的条件概率仅依赖于该自旋所处的有效局部场
$$
h_i^{\mathrm{eff}}=\sum_{j\neq i}J_{ij}s_j+h_i
$$
具体形式为
$$
p(s_i=+1\mid\mathbf{s}_{\neq i})=\frac{1}{1+\exp(-2\beta h_i^{\mathrm{eff}})}=\sigma(2\beta h_i^{\mathrm{eff}})
$$
其中$\sigma(\cdot)$为sigmoid函数。该条件概率在形式上与双稳态随机器件在外部偏置与热扰动共同作用下的占据概率分布一致，这正是sMTJ可作为伊辛自旋的物理载体、从而在硬件层面实现玻尔兹曼采样的根本依据。

### 3.1.2 伊辛哈密顿量与QUBO的等价变换

许多文献以二次无约束二值优化（quadratic unconstrained binary optimization，QUBO）形式给出映射[^Kochenberger2014]：
$$
f(\mathbf{x})=\mathbf{x}^{\top}Q\mathbf{x},\quad x_i\in\{0,1\}
$$
其中$Q$为对称矩阵。通过线性变换$s_i=2x_i-1$可将伊辛模型与QUBO相互转化。将$s_is_j=4x_ix_j-2x_i-2x_j+1$以及$s_i=2x_i-1$代入伊辛哈密顿量并整理，可得
$$
H(\mathbf{s})=-4\sum_{i<j}J_{ij}x_ix_j+2\sum_i\!\left(\sum_{j\neq i}J_{ij}-h_i\right)\!x_i+C
$$
其中常数$C=-\sum_{i<j}J_{ij}+\sum_ih_i$不影响最小化。对照QUBO定义可读出
$$
Q_{ij}=-4J_{ij}\ (i<j),\qquad Q_{ii}=2\!\left(\sum_{j\neq i}J_{ij}-h_i\right)
$$
反之，给定QUBO参数时取$s_i=2x_i-1$即可读回等价的伊辛参数。二者在多项式时间内无损互换，且最优解一一对应。本章后续一律采用伊辛形式描述，与以QUBO形式发布的基准实例或算法实现对接时只需按上式完成系数重映射。该转换在仿真框架中被实现为`problems.qubo_to_ising(Q)`，取$J=-Q_{\text{off}}/4$与$h_i=-Q_{\text{off}}\mathbf{1}/4-Q_{ii}/2$，其中$Q_{\text{off}}$表示$Q$剥离对角后的部分。所有QUBO形式的问题（TSP、因子分解等）均先在各自问题模块中装配$Q$矩阵，统一经该转换函数进入求解器。

### 3.1.3 Max-Cut问题的伊辛映射

Max-Cut是NP完全的图划分问题[^Goemans1995]：给定加权无向图$G=(V,E,w)$，寻找将顶点集$V$划分为两部分$(S,\bar S)$的方案，使被切断的边权和
$$
\mathrm{cut}(S)=\sum_{(i,j)\in E,\,i\in S,\,j\in\bar S}w_{ij}
$$
最大。

设自旋$s_i=+1$表示$i\in S$、$s_i=-1$表示$i\in\bar S$，则$i,j$分属两侧时$s_is_j=-1$、同侧时$s_is_j=+1$。于是
$$
\mathrm{cut}(S)=\frac{1}{2}\sum_{(i,j)\in E}w_{ij}(1-s_is_j)=\frac{W}{2}-\frac{1}{2}\sum_{(i,j)\in E}w_{ij}s_is_j
$$
其中$W=\sum_{(i,j)\in E}w_{ij}$。最大化切割等价于最小化
$$
H(\mathbf{s})=-\sum_{i<j}J_{ij}s_is_j,\qquad J_{ij}=\begin{cases}-w_{ij}/2,&(i,j)\in E\\0,&\text{其它}\end{cases}
$$
外场项$h_i\equiv 0$。

该映射有两点使其成为伊辛机基准测试的首选：一是$J$的稀疏模式与原图邻接矩阵一致，问题规模$|V|$即为自旋数，不引入任何辅助变量；二是不含约束惩罚项，全体自旋构型都是可行解，求解器不会因惩罚权重选取不当而陷入不可行区。Goemans和Williamson基于半定规划给出的$0.87856$近似算法[^Goemans1995]为启发式求解器提供了明确的解质量参照上界。由Ye整理并发布于Stanford公开站点的G-set基准集在Max-Cut求解器文献中被反复使用，本章将其作为Max-Cut性能评估的标准实例集。

### 3.1.4 旅行商问题的伊辛映射

对含$n$个城市的对称旅行商问题（traveling salesman problem，TSP），采用$n\times n$二值编码：变量$x_{v,j}\in\{0,1\}$表示城市$v$是否位于巡游的第$j$个位置。合法的巡游需满足每个城市恰被访问一次、以及每个位置恰含一个城市，这在伊辛形式下以惩罚项表示[^Lucas2014]：
$$
H_{\mathrm{const}}=A\sum_{v}\!\left(1-\sum_{j}x_{v,j}\right)^{\!2}+A\sum_{j}\!\left(1-\sum_{v}x_{v,j}\right)^{\!2}
$$
路径长度项为
$$
H_{\mathrm{cost}}=B\sum_{(u,v)\in E}d_{uv}\sum_{j=1}^{n}x_{u,j}x_{v,j+1}
$$
其中下标$j$按模$n$循环，$d_{uv}$为城市间距离。完整哈密顿量为$H=H_{\mathrm{const}}+H_{\mathrm{cost}}$，经$x_i=(1+s_i)/2$代换即化为伊辛形式。

惩罚系数$A,B$的选取直接关系到基态是否对应合法路径。当$A>B\cdot\max_{u,v}d_{uv}$时，任何违反约束带来的能量上升都超过路径代价可能的最大下降，从而保证基态解的可行性[^Lucas2014]。该映射将原本$n$城市的问题放大到$n^2$个自旋，耦合矩阵呈块结构：城市块内部完全耦合以执行单城约束，位置块内部亦然；跨块耦合由距离矩阵填充。规模放大与耦合密集是TSP被用作求解器压力测试的主要原因。

**实现札记：可行性阈值与实际甜点的差距**。可行性条件$A>Bd_{\max}$给出的是基态保证，并未说明退火过程中可行域的可达性。在仿真框架的`problems.build_tsp_problem`实现中，惩罚系数采用参数化形式`A = A_margin * B * dmax`，`A_margin`由调用方显式给定。本章主基准取$A\text{-margin}=2.0$（惩罚是距离最大值的两倍），但后续基准节的实测将显示该取值下`ulysses16`等实例的单次试验可行率仅$3\%$——远低于惩罚足够大则必可行的直觉预期。这一差距的机理将在后续基准节结合实测数据与能量轨迹展开；在问题装配接口设计层面，其直接启示是**将$A\text{-margin}$作为显式超参数而非硬编码常数**，为后续对该参数的系统扫描留下接口：

```python
def build_tsp_problem(D, B=1.0, A_margin=2.0, name=None):
    n = D.shape[0]
    A = A_margin * B * float(D.max())  # penalty strength
    # ... assemble Q via row/column constraint and cost terms ...
    return Problem(name=name, n=n * n, J=J, h=h,
                   meta={"D": D, "n_cities": n, "A": A, "B": B})
```

这一参数化在后续TSP基准的消融实验中被直接使用，扫描$A\text{-margin}\in\{1.2, 1.5, 2.0, 3.0, 5.0, 10.0\}$。

### 3.1.5 整数分解的伊辛映射

给定合数$M$，寻找正整数$p,q$使$M=pq$的问题在密码学与随机计算基准中具有重要背景。将$p,q$按二进制展开为位变量$p_i,q_j\in\{0,1\}$，则$M=pq$按位展开为一系列按位乘法与进位约束：
$$
\sum_{i+j=k}p_iq_j+\sum_{\ell}c_{\ell,k}=M_k+2\sum_{\ell}c_{\ell,k+1}
$$
其中$c_{\ell,k}$为进位辅助变量，$M_k$为$M$的第$k$位。整体哈密顿量取为所有按位约束残差的平方和[^Jiang2018]：
$$
H=\sum_{k}\!\left(\sum_{i+j=k}p_iq_j+\sum_{\ell}c_{\ell,k}-M_k-2\sum_{\ell}c_{\ell,k+1}\right)^{\!2}
$$
展开后含高阶多体相互作用项，可通过引入辅助变量降为两体[^Lucas2014]，从而严格满足伊辛模型的两体耦合要求。

与Max-Cut和TSP相比，整数分解的映射有两个显著特点：其一，自旋数量随$\log M$的平方增长，辅助进位变量的数量与原变量数量可比；其二，耦合结构非对称且局部密集，其拓扑直接反映乘法电路。Borders等在2019年基于sMTJ的p-bit阵列对小规模半素数分解问题完成了硬件实验验证[^Borders2019]，这也是本章将整数分解列为基准实例的主要动因之一。

**实现札记：比特预算的最小覆盖原则**。上述映射留有一个关键自由度：$p$与$q$各自分配的位宽$(b_p, b_q)$。理论上只要$2^{b_p}\cdot 2^{b_q}$的乘积空间覆盖真实因子对即可，但实际求解性能对该自由度高度敏感。以$M=51=3\times 17$为例：若$b_q=4$则$\hat q\in\{1, 3, \dots, 15\}$不含$17$，**无论退火多长都不可能命中真解**；若$b_q$取过大值（如$b_q=7$），可表达集合膨胀至$\{1, 3, \dots, 127\}$，真实最优点$\hat q=17$在允许集合中的相对密度被稀释，随机初始化落入正确吸引盆地的先验概率降低。后续因子分解基准的消融实验将定量展示这两种失效模式。仿真框架中，`problems.build_factoring_problem`对此采用**最小覆盖分配**策略作为缺省：对已知的$M$通过$O(\sqrt M)$试除法找到真实因子$(p, q)$，取$b_p=\lceil\log_2(p+1)\rceil$、$b_q=\lceil\log_2(q+1)\rceil$，恰好覆盖真因子：

```python
def suggest_factoring_bp_bq(M):
    """Trial-division to locate the smaller factor, then minimal
    bit-coverage on each side."""
    for p in range(3, int(np.sqrt(M)) + 1, 2):
        if M % p == 0:
            q = M // p
            return (int(np.ceil(np.log2(p + 1))),
                    int(np.ceil(np.log2(q + 1))))
    raise ValueError(f"{M} is prime or has no small factor")
```

这一做法使不同$M$实例间的$N_{\text{spin}}$规模紧凑可比，且与实际因子的不平衡度无关。对该默认策略的替代做法（模拟实际盲分解场景、显式指定较大$(b_p, b_q)$）可通过关键字参数传入，其性能退化将在后续因子分解基准中量化。此处仅强调：编码层的参数选取同样在本章的评估流水线上游，对下游算法性能的影响不亚于退火调度的调参。

### 3.1.6 其它典型问题的映射综述

除上述三类问题外，3-SAT、图着色、最小顶点覆盖、背包、数分划、哈密顿回路等多类NP-hard问题均可按类似思路映射为伊辛形式，具体推导可参见Lucas[^Lucas2014]。下表归纳这些问题在伊辛映射下的基本规模指标。

| 问题 | 自旋数阶数 | 耦合稀疏性 | 惩罚项个数 |
|---|---|---|---|
| Max-Cut | $O(\lvert V\rvert)$ | 与原图一致 | 0 |
| 图$k$着色 | $O(k\lvert V\rvert)$ | 局部 | 2 |
| 最小顶点覆盖 | $O(\lvert V\rvert)$ | 与原图一致 | 1 |
| 3-SAT | $O(\lvert V\rvert+\lvert C\rvert)$ | 局部 | $O(\lvert C\rvert)$ |
| 0-1背包 | $O(n+\log W)$ | 密集 | 1 |
| 数分划 | $O(\lvert S\rvert)$ | 全连接 | 0 |
| TSP | $O(n^2)$ | 块密集 | 2 |
| 整数分解 | $O(\log^2 M)$ | 结构化 | $O(\log M)$ |

其中$\lvert V\rvert, \lvert E\rvert, \lvert C\rvert$分别表示图的顶点数、边数或子句数，$n, W$分别表示项目数与背包容量。该列表并不详尽，其意义在于表明上述映射方法覆盖面广。由于各类问题在接口上最终都归结为同一对$(J,h)$，本章设计的仿真框架可作为通用求解器使用，具体问题仅在前端的输入加载阶段区别，这也构成将仿真框架设计为问题层与求解器解耦结构的直接动因。

### 本节参考文献

[^Karp1972]: Karp R M. Reducibility Among Combinatorial Problems[M]//Miller R E, Thatcher J W, Bohlinger J D. Complexity of Computer Computations. New York: Plenum Press, 1972: 85-103. DOI: [10.1007/978-1-4684-2001-2_9](https://doi.org/10.1007/978-1-4684-2001-2_9).

[^Lucas2014]: Lucas A. Ising Formulations of Many NP Problems[J]. Frontiers in Physics, 2014, 2: 5. DOI: [10.3389/fphy.2014.00005](https://doi.org/10.3389/fphy.2014.00005).

[^Barahona1982]: Barahona F. On the Computational Complexity of Ising Spin Glass Models[J]. Journal of Physics A: Mathematical and General, 1982, 15(10): 3241-3253. DOI: [10.1088/0305-4470/15/10/028](https://doi.org/10.1088/0305-4470/15/10/028).

[^Kirkpatrick1983]: Kirkpatrick S, Gelatt C D, Vecchi M P. Optimization by Simulated Annealing[J]. Science, 1983, 220(4598): 671-680. DOI: [10.1126/science.220.4598.671](https://doi.org/10.1126/science.220.4598.671).

[^Kochenberger2014]: Kochenberger G, Hao J K, Glover F, et al. The Unconstrained Binary Quadratic Programming Problem: A Survey[J]. Journal of Combinatorial Optimization, 2014, 28(1): 58-81. DOI: [10.1007/s10878-014-9734-0](https://doi.org/10.1007/s10878-014-9734-0).

[^Goemans1995]: Goemans M X, Williamson D P. Improved Approximation Algorithms for Maximum Cut and Satisfiability Problems Using Semidefinite Programming[J]. Journal of the ACM, 1995, 42(6): 1115-1145. DOI: [10.1145/227683.227684](https://doi.org/10.1145/227683.227684).

[^Jiang2018]: Jiang J H, Chancellor N, Zohren S, et al. Quantum Annealing for Prime Factorization[J]. Scientific Reports, 2018, 8: 17667. DOI: [10.1038/s41598-018-36058-z](https://doi.org/10.1038/s41598-018-36058-z).

[^Borders2019]: Borders W A, Pervaiz A Z, Fukami S, et al. Integer Factorization Using Stochastic Magnetic Tunnel Junctions[J]. Nature, 2019, 573(7774): 390-393. DOI: [10.1038/s41586-019-1557-9](https://doi.org/10.1038/s41586-019-1557-9).
