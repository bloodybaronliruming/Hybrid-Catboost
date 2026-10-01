"""Render the author-approved Figure 1 geometry without private inputs."""
from pathlib import Path
import hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def render_pipeline(output):
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig=plt.figure(figsize=(180/25.4,109/25.4))
    ax=fig.add_axes([0,0,1,1]); ax.set(xlim=(0,180),ylim=(0,109)); ax.axis('off')
    colors={'neutral':'#647078','morgan':'#0072B2','descriptor':'#D55E00','hybrid':'#009E73'}
    texts=[]
    def box(x,y,w,h,lines,color):
        patch=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=1.3',linewidth=1,edgecolor=color,facecolor='white',zorder=3)
        ax.add_patch(patch)
        text=ax.text(x+w/2,y+h/2,lines,ha='center',va='center',linespacing=1.35,fontsize=8.5,color='#202428',zorder=4)
        texts.append((text,patch))
    def arrow(points):
        for a,b in zip(points[:-2],points[1:-1]):
            ax.plot([a[0],b[0]],[a[1],b[1]],color='#41484d',lw=1,zorder=1)
        ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>',mutation_scale=9,linewidth=1,color='#41484d',shrinkA=0,shrinkB=1,zorder=2))
    box(3,73,28,17,'Original SMILES\nRDKit parsing',colors['neutral'])
    box(40,87,39,17,'Morgan fingerprint\nRadius 2; 2,048 bits',colors['morgan'])
    box(40,52,39,17,'RDKit 2D descriptors\n210 initial descriptors',colors['descriptor'])
    box(87,47,43,27,'Median imputation\nZero-variance filtering\n206–207 descriptors',colors['descriptor'])
    box(137,76,40,20,'Feature concatenation\n2,254–2,255 features',colors['hybrid'])
    box(137,49,40,18,'Ipc transformation\nIpc → ln(1 + Ipc)',colors['hybrid'])
    box(137,22,40,18,'Hybrid-CatBoost\nFive independent fits',colors['hybrid'])
    box(110,3,67,13,'Five prediction sets\nSame 1,997 fixed-test records',colors['neutral'])
    for points in [[(31,83),(35,83),(35,95.5),(40,95.5)],[(31,80),(35,80),(35,60.5),(40,60.5)],[(79,95.5),(133,95.5),(133,89),(137,89)],[(79,60.5),(87,60.5)],[(130,60.5),(133,60.5),(133,82),(137,82)],[(157,76),(157,67)],[(157,49),(157,40)],[(157,22),(157,16)]]:
        arrow(points)
    ax.text(108.5,43.5,'Fitted on training\ndata only',ha='center',va='top',fontsize=8,color='#7a361b',linespacing=1.25)
    ax.text(4,25,'Official seeds 1–5',ha='left',va='center',fontsize=9,fontweight='bold',color='#202428')
    ax.text(4,18,'Seed-specific descriptor processing and model fitting.\nEach fitted pipeline is applied to the same fixed test set.',ha='left',va='top',fontsize=8.2,linespacing=1.5,color='#41484d')
    fig.canvas.draw(); renderer=fig.canvas.get_renderer()
    for text,patch in texts:
        tb=text.get_window_extent(renderer);pb=patch.get_window_extent(renderer)
        assert pb.contains(tb.x0,tb.y0) and pb.contains(tb.x1,tb.y1),text.get_text()
    outputs = {}
    for extension in ('pdf','svg','png'):
        target = output / f'figure_01_actual_pipeline.{extension}'
        fig.savefig(target,dpi=600,facecolor='white')
        outputs[extension] = {'path':target.name,'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
    plt.close(fig)
    return outputs
