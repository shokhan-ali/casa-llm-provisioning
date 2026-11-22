import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({"font.family":"DejaVu Serif","font.size":8})
fig,ax=plt.subplots(figsize=(7.2,3.4),dpi=220); ax.set_xlim(0,100); ax.set_ylim(0,48); ax.axis("off")
def box(x,y,w,h,t,fc="#eef3fa",ec="#4e79a7",bold=False):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.4,rounding_size=1.2",fc=fc,ec=ec,lw=1))
    ax.text(x+w/2,y+h/2,t,ha="center",va="center",fontsize=6.6,weight="bold" if bold else "normal")
def arr(x1,y1,x2,y2,t="",dy=1.2,ls="-"):
    ax.annotate("",(x2,y2),(x1,y1),arrowprops=dict(arrowstyle="-|>",lw=0.9,color="#333",ls=ls))
    if t: ax.text((x1+x2)/2,(y1+y2)/2+dy,t,ha="center",fontsize=6.4,color="#333")
box(1,30,14,10,"Request stream\n(input / output\ntokens per min)",fc="#f5f5f5",ec="#777")
box(20,30,15,10,"Forecaster\n(any model)\n$\\hat Y^P_t,\\ \\hat Y^D_t$")
box(20,15,15,10,"Scale estimate\n$\\hat\\sigma^P_t,\\ \\hat\\sigma^D_t$\n(rolling MAE)")
box(40,15,22,25,"",fc="#fdf0ef",ec="#e15759")
ax.text(51,37.6,"CASA",ha="center",fontsize=9,weight="bold",color="#c0392b")
box(42,27,18,8,"Efficiency loop\nsplit $\\varphi_t$ = argmin cost",fc="#ffffff",ec="#e15759")
box(42,17,18,8,"Validity loop (joint ACI)\nlevel $\\alpha_t$ → margin $\\theta_t$",fc="#ffffff",ec="#e15759")
box(67,30,14,10,"Prefill pool\n$n^P_t$ instances",fc="#e9f5e6",ec="#59a14f")
box(67,15,14,10,"Decode pool\n$n^D_t$ instances",fc="#e9f5e6",ec="#59a14f")
box(85,22,14,10,"Realised demand\n$Y_t$ → violation\n$E_t$",fc="#f5f5f5",ec="#777")
arr(15,35,20,35); arr(35,35,42,31); arr(35,20,42,21); arr(51,27,51,25)
arr(62,24,67,34,"$U^P_t$"); arr(62,20,67,20,"$U^D_t$")
arr(81,35,85,29); arr(81,20,85,25)
ax.annotate("",(51,15.3),(92,22),arrowprops=dict(arrowstyle="-|>",lw=0.9,color="#c0392b",ls="--",connectionstyle="arc3,rad=-0.35"))
ax.text(76,6.5,"feedback after lead time h:  $\\alpha_{t+1}=\\alpha_t+\\gamma(\\alpha-E_t)$;  residuals → buffer",ha="center",fontsize=6.6,color="#c0392b")
ax.text(50,45.5,"Validity (Theorem 1) holds for any split $\\varphi_t$, forecaster and scale",ha="center",fontsize=7,style="italic")
import os
os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)),"figures"),exist_ok=True)
fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)),"figures","fig1_architecture.png"),bbox_inches="tight"); print("ok")
