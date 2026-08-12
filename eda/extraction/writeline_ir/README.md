# `writeline_ir/` — 写线IR压降与逐行静态驱动偏移

从版图提取的sky130 met2方块电阻出发，量化64×64伊辛阵列（表3.11的N_par=64硬件投影）中列写线电阻对各行写电压的静态压降，并把该逐行偏移馈入第3章求解器评估其对退火性能的影响。流程移植自smtj_pbnn_sim的`eda/extraction/writeline`。

## 模型

阵列按行顺序更新，每条列写线服务64行（单元间距2 µm，met2，W=1 µm）。驱动器到第r行之间的金属往返电阻（位线+源线回流，沿用PBNN约定）为$$R_\mathrm{line}(r)=2R_s\,(r\cdot\text{pitch})/W$$。若驱动器按最近行标定输出$$V_\mathrm{th}=895.783\ \mathrm{mV}$$，则第r行实际得到$$V_\mathrm{th}-\Delta V(r)$$，其中（精确分压，$$I_\mathrm{write}\approx1.15\ \mathrm{mA}$$）

$$\Delta V(r)=V_\mathrm{th}\,\frac{R_\mathrm{line}(r)}{R_\mathrm{SOT}+R_\mathrm{line}(r)},\qquad R_\mathrm{SOT}=776\ \Omega.$$

在标定Sigmoid上这等效于逐行静态驱动偏移$$u_\mathrm{off}(r)=\Delta V(r)/V_T$$（概率窗$$V_T=23.414\ \mathrm{mV}$$）：该行更新概率变为$$\sigma(u-u_\mathrm{off}(r))$$。

## 流程

1. `gen_strap.py`（KLayout批处理）：生成met2校准条（400方块）与n16/n64/n256三种真实写线几何（L=N×2 µm，W=1 µm），另加一条400方块poly校准条用于验证提取链路，两端打标签→`writeline_straps.gds`。
2. `run_extresist.sh`（Magic）：`extract do resistance → extresist → ext2spice`。自检分两级：
   - poly条经extresist双端口提取47.96 Ω/□，对techfile 48.2 Ω/□偏差−0.50%（偏差来源即两端标签内缩0.5 µm，398/400方块），与PBNN流程的校验点一致；
   - met2低阻网络会被extresist的网络筛选丢弃（任何tolerance下均不输出），故met2方块电阻取自`extract do resistance`写入`.ext`节点记录的集总电阻：校准条50.0 Ω/400方块=0.1250 Ω/□，与techfile一致，n16/n64/n256三条几何交叉一致。
3. `analyze_ir.py`：由met2方块电阻算r=1..N各行的$$R_\mathrm{line}$$、$$\Delta V$$与$$u_\mathrm{off}$$（N∈{16,64,256}），并给出预畸变视角：按6位DAC的LSB逐行补偿码$$\mathrm{code}(r)=\mathrm{round}(\Delta V/\mathrm{LSB})$$，残差$$|\Delta V-\mathrm{code}\cdot\mathrm{LSB}|\le\mathrm{LSB}/2$$。LSB读取`eda/testbenches/update_chain_summary.json`的`per_bits["6"].lsb_mV`（W2扫描完成前用理想值2.97 mV，JSON中标注FALLBACK；文件出现后重跑本脚本即自动切换为MEASURED）。→`ir_drop_summary.json`
4. `ir_solver_impact.py`：把N=64逐行偏移经`eda/interface/circuit_backends.py`的`circuit_chain`后端（`mode="none"`，`u_offset`为逐自旋数组，符号取负表示驱动亏缺）馈入求解器。协议与3.4.2节器件消融完全一致：14自旋ER Max-Cut（p=0.30，seed=0），穷举基态，几何退火β 0.1→5.0、2000扫、块更新、200次多起点（master_seed=2024）。14个自旋均匀映射到行0..63：`round(i*63/13)`=[0,5,10,15,19,24,29,34,39,44,48,53,58,63]，行k距驱动器(k+1)个间距。→`ir_solver_impact.csv`

## 结果

提取侧（`ir_drop_summary.json`，met2 0.1250 Ω/□为MEASURED，逐行公式为ANALYTIC）：

| 列高N | 远端$$R_\mathrm{line}$$ | 远端$$\Delta V$$ | 远端$$u_\mathrm{off}$$ | 补偿码上限 | 残差上限 |
|---|---|---|---|---|---|
| 16 | 8 Ω | 9.14 mV | 0.390 | 3/63 | 0.063 u |
| 64 | 32 Ω | 35.48 mV | 1.515 | 12/63 | 0.063 u |
| 256 | 128 Ω | 126.84 mV | 5.417 | 43/63 | 0.063 u |

求解器侧（`ir_solver_impact.csv`，N=64剖面）：

| 场景 | max\|u_off\| | p_success | TTS99（扫） | 相对基线 |
|---|---|---|---|---|
| 无偏移基线 | 0 | 0.185 | 45024 | 1.00 |
| 未补偿 | 1.515 | 0.065 | 137041 | 3.04 |
| 预畸变残差 | 0.063 | 0.175 | 47878 | 1.06 |

## 结论

- 64行阵列的写线IR压降未补偿时把远端行的驱动窗平移最多1.5个$$V_T$$，成功率从0.185降到0.065，TTS约3倍；该偏移是确定性的静态量，可由驱动侧预畸变消除。
- 6位DAC逐行预畸变（远端行占用12/63个码）把残差压到LSB/2≈1.54 mV即0.063 u，求解性能回到基线（比值1.06，在多起点统计涨落内）。代价是远端行让出约19%的DAC码域，且远端所需驱动电压升至0.933 V（N=64）。
- N=256时远端压降5.4 u、补偿码占43/63，单段写线已不可行，与PBNN流程"高列分段或换更低阻层"的指引一致。

## 复现

```bash
wsl -d Ubuntu-24.04-EDA --cd <repo> -- klayout -b -r eda/extraction/writeline_ir/gen_strap.py
wsl -d Ubuntu-24.04-EDA -- bash -lc 'cd "<repo>/eda/extraction/writeline_ir" && bash run_extresist.sh'
python eda/extraction/writeline_ir/analyze_ir.py
python eda/extraction/writeline_ir/ir_solver_impact.py --jobs 10
```
