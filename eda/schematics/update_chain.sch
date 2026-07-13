v {xschem version=3.4.4 file_version=1.2}
G {}
K {}
V {}
S {}
E {}
C {sym/res.sym} 190 270 0 0 {name=Rs1 value=R_u}
C {sym/res.sym} 190 350 0 0 {name=Rs2 value=R_u}
C {sym/res.sym} 190 430 0 0 {name=RsN value=R_u}
C {sym/nfet.sym} 340 340 0 0 {name=TGn0 model=sky130_fd_pr__nfet_01v8 W=4 L=0.15 nf=1 m=1}
C {sym/pfet.sym} 500 340 0 0 {name=TGp0 model=sky130_fd_pr__pfet_01v8 W=8 L=0.15 nf=1 m=1}
C {sym/pfet.sym} 620 190 0 0 {name=M4 model=sky130_fd_pr__pfet_01v8 W=20 L=0.5 nf=1 m=1}
C {sym/pfet.sym} 790 190 0 0 {name=M3 model=sky130_fd_pr__pfet_01v8 W=20 L=0.5 nf=1 m=1}
C {sym/nfet.sym} 620 310 0 0 {name=M2 model=sky130_fd_pr__nfet_01v8 W=20 L=0.5 nf=1 m=1}
C {sym/nfet.sym} 790 310 0 0 {name=M1 model=sky130_fd_pr__nfet_01v8 W=20 L=0.5 nf=1 m=1}
C {sym/nfet.sym} 690 430 0 0 {name=Mt model=sky130_fd_pr__nfet_01v8 W=20 L=0.5 nf=1 m=1}
C {sym/pfet.sym} 950 190 0 0 {name=Mo model=sky130_fd_pr__pfet_01v8 W=80 L=0.5 nf=1 m=1}
C {sym/nfet.sym} 1080 340 0 0 {name=WEn model=sky130_fd_pr__nfet_01v8 W=4 L=0.15 nf=1 m=1}
C {sym/pfet.sym} 1240 340 0 0 {name=WEp model=sky130_fd_pr__pfet_01v8 W=8 L=0.15 nf=1 m=1}
C {sym/sot_mtj.sym} 1420 350 0 0 {name=N1}
C {sym/cap.sym} 920 320 0 0 {name=Ccm value=2p}
N 190 225 190 240 {}
N 190 300 190 310 {}
N 190 310 190 320 {}
N 190 380 190 400 {}
N 190 460 190 475 {}
N 40 340 80 340 {}
N 190 310 360 310 {}
N 360 310 520 310 {}
N 360 370 520 370 {}
N 520 370 540 370 {}
N 540 370 540 285 {}
N 540 285 755 285 {}
N 755 285 755 310 {}
N 755 310 770 310 {}
N 640 110 810 110 {}
N 810 110 970 110 {}
N 640 110 640 160 {}
N 640 190 640 160 {}
N 810 110 810 160 {}
N 810 190 810 160 {}
N 970 110 970 160 {}
N 970 190 970 160 {}
N 640 220 640 245 {}
N 640 245 640 280 {}
N 810 220 810 250 {}
N 810 250 810 280 {}
N 600 190 585 190 {}
N 585 190 585 245 {}
N 585 245 640 245 {}
N 640 245 740 245 {}
N 740 245 740 190 {}
N 740 190 770 190 {}
N 810 250 920 250 {}
N 920 250 920 190 {}
N 920 190 930 190 {}
N 920 250 920 290 {}
N 920 350 920 390 {}
N 920 390 1010 390 {}
N 1010 390 1010 310 {}
N 640 340 640 370 {}
N 810 340 810 370 {}
N 640 370 710 370 {}
N 710 370 810 370 {}
N 710 370 710 400 {}
N 710 460 710 490 {}
N 970 220 970 310 {}
N 970 310 1010 310 {}
N 1010 310 1100 310 {}
N 1100 310 1260 310 {}
N 1100 370 1260 370 {}
N 1260 370 1380 370 {}
N 1260 370 1260 560 {}
N 1260 560 575 560 {}
N 575 560 575 310 {}
N 575 310 600 310 {}
N 1460 370 1530 370 {}
N 1420 310 1420 280 {}
N 1450 357 1490 357 {}
C {sym/lab_pin.sym} 190 225 0 1 {name=l1 lab=vhi}
C {sym/lab_pin.sym} 190 475 0 0 {name=l2 lab=vlo}
C {sym/lab_pin.sym} 40 340 0 2 {name=l3 lab=code}
L 4 58 348 66 332 {}
T {b-bit} 40 312 0 0 0.18 0.18 {}
L 4 80 200 300 200 {dash=4}
L 4 300 200 300 480 {dash=4}
L 4 80 480 300 480 {dash=4}
L 4 80 200 80 480 {dash=4}
N 320 340 295 340 {}
C {sym/lab_pin.sym} 295 340 0 2 {name=l4 lab=sel}
N 480 340 455 340 {}
C {sym/lab_pin.sym} 455 340 0 2 {name=l5 lab=selb}
N 1060 340 1035 340 {}
C {sym/lab_pin.sym} 1035 340 0 2 {name=l6 lab=wen}
N 1220 340 1195 340 {}
C {sym/lab_pin.sym} 1195 340 0 2 {name=l7 lab=wep}
N 670 430 645 430 {}
C {sym/ipin.sym} 645 430 0 2 {name=l8 lab=Vnb}
C {sym/lab_pin.sym} 450 370 0 0 {name=l9 lab=bin}
C {sym/lab_pin.sym} 640 232 0 0 {name=l10 lab=bd2}
C {sym/lab_pin.sym} 810 235 0 0 {name=l11 lab=bd1}
C {sym/lab_pin.sym} 990 310 0 0 {name=l12 lab=drv}
C {sym/lab_pin.sym} 900 560 0 0 {name=l13 lab=wr}
C {sym/lab_pin.sym} 1330 370 0 0 {name=l14 lab=wr}
C {sym/lab_pin.sym} 1500 370 0 0 {name=l15 lab=com}
C {sym/lab_pin.sym} 1490 357 0 1 {name=l16 lab=st}
C {sym/opin.sym} 1420 280 0 1 {name=l17 lab=rd}
N 360 340 385 340 {}
C {sym/gnd.sym} 385 340 0 0 {name=l18 lab=VSS}
N 640 310 665 310 {}
C {sym/gnd.sym} 665 310 0 0 {name=l19 lab=VSS}
N 810 310 835 310 {}
C {sym/gnd.sym} 835 310 0 0 {name=l20 lab=VSS}
N 710 430 735 430 {}
C {sym/gnd.sym} 735 430 0 0 {name=l21 lab=VSS}
N 1100 340 1125 340 {}
C {sym/gnd.sym} 1125 340 0 0 {name=l22 lab=VSS}
N 520 340 545 340 {}
C {sym/vdd.sym} 545 340 0 0 {name=l23 lab=VDD}
N 1260 340 1285 340 {}
C {sym/vdd.sym} 1285 340 0 0 {name=l24 lab=VDD}
N 640 110 640 82 {}
C {sym/vdd.sym} 640 82 0 1 {name=l25 lab=VDD}
C {sym/gnd.sym} 710 490 0 0 {name=l26 lab=VSS}
C {sym/gnd.sym} 1530 370 0 0 {name=l27 lab=VSS}
T {resistor-string DAC} 85 162 0 0 0.28 0.28 {}
T {code-select TG} 350 248 0 0 0.26 0.26 {}
T {two-stage Miller unity buffer} 660 55 0 0 0.28 0.28 {}
T {write-enable TG (x8)} 1050 248 0 0 0.26 0.26 {}
T {sMTJ p-bit device} 1345 240 0 0 0.26 0.26 {}
T {V_th + 4V_T = 989.4 mV} 60 178 0 0 0.22 0.22 {}
T {V_th - 4V_T = 802.1 mV} 60 498 0 0 0.22 0.22 {}
T {R_u = 100 ohm} 205 288 0 0 0.2 0.2 {}
T {2^b taps} 205 320 0 0 0.2 0.2 {}
T {(0.9 V)} 590 445 0 0 0.18 0.18 {}
T {x2} 725 468 0 0 0.22 0.22 {layer=13}
T {x2} 985 228 0 0 0.22 0.22 {layer=13}
T {feedback: sensed after write-enable TG} 700 540 0 0 0.22 0.22 {}
T {V_wr = V_th + u*V_T} 1280 415 0 0 0.26 0.26 {layer=7}
T {0.75 ns pulse, P->AP probabilistic write} 1280 440 0 0 0.22 0.22 {}
T {R_SOT = 776 ohm} 1280 462 0 0 0.2 0.2 {layer=13}
