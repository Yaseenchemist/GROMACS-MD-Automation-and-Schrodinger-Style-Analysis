#!/usr/bin/env python3
"""
pca_fel_pro.py - Professional PCA + FEL for GROMACS MD
Faithful reimplementation of reference pca_gromacs.py + plotly notebooks.

Pipeline: covar -mwa -ref -> anaeig -proj -> analyze -cc (official cosine)
          -> anaeig -2d/-3d/-extr -> sham (real Gibbs FEL)
Plots: interactive Plotly HTML (scree, cosine, 2D proj, FEL 2D+3D).
Group: uses Binding_Pocket (default Group 21) from index.ndx for all PCA/FEL coordinates.

Files: holo = MD.tpr / MD_center.xtc ; apo = apo_protein.tpr / apo_MD_center.xtc ; index = index.ndx
"""
from __future__ import annotations
import os, sys, argparse, subprocess
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import numpy as np
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    _HAS_PLOTLY = True
except Exception:
    _HAS_PLOTLY = False
COLORSCALE = "jet"

def run_gmx(gmx_bin, args, stdin_text="", cwd=None, label=""):
    cmd=[gmx_bin]+args
    print(f"[gmx] {label or args[0]}: {' '.join(cmd)}"+(f"  (stdin='{stdin_text.strip()}')" if stdin_text else ""))
    try:
        r=subprocess.run(cmd,input=stdin_text,capture_output=True,text=True,cwd=str(cwd) if cwd else None,timeout=7200)
        if r.returncode!=0:
            print(f"[gmx][error] {label} failed:\n{(r.stderr or '')[-500:]}"); return False
        return True
    except Exception as e:
        print(f"[gmx][error] {label} exception: {e}"); return False

def gmx_capture(gmx_bin,args,stdin_text="",cwd=None):
    try:
        r=subprocess.run([gmx_bin]+args,input=stdin_text,capture_output=True,text=True,cwd=str(cwd) if cwd else None,timeout=600)
        return (r.stdout or "")+(r.stderr or "")
    except Exception as e:
        return f"__ERROR__ {e}"

def detect_binding_pocket_group(gmx_bin, tpr, index_file, requested=21):
    """
    Verify and return the Binding_Pocket group from an existing index.ndx.

    The previous version auto-detected the built-in C-alpha group (usually Group 3).
    This version deliberately does NOT do that: PCA/FEL are performed on the
    user-defined Binding_Pocket group, normally Group 21 in index.ndx.
    """
    idx = Path(index_file).resolve()
    if not idx.exists():
        raise FileNotFoundError(f"Index file not found: {idx}")

    text = idx.read_text(errors="replace")
    groups = []
    current = None
    atoms = 0
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("[") and line.endswith("]"):
            if current is not None:
                groups.append((current, atoms))
            current = line[1:-1].strip()
            atoms = 0
        elif current is not None and line and not line.startswith(";"):
            for tok in line.split():
                try:
                    int(tok)
                    atoms += 1
                except ValueError:
                    pass
    if current is not None:
        groups.append((current, atoms))

    if requested >= len(groups):
        raise RuntimeError(
            f"Requested Group {requested}, but {idx.name} contains only "
            f"{len(groups)} groups."
        )
    name, natoms = groups[requested]
    if name != "Binding_Pocket":
        raise RuntimeError(
            f"Group {requested} in {idx.name} is '{name}', not 'Binding_Pocket'. "
            f"Please make sure Group 21 is named [ Binding_Pocket ]."
        )
    if natoms <= 0:
        raise RuntimeError(f"Binding_Pocket Group {requested} is empty in {idx.name}.")

    print(f"[info] Using Binding_Pocket = Group {requested} ({natoms} atoms) from {idx}")
    return requested, natoms

def _parse_xvg_block(text):
    x,y,xt,yt,title=[],[],None,None,None
    for l in text.splitlines():
        if "xaxis  label" in l: xt=l.split('"')[1] if '"' in l else None; continue
        if "yaxis  label" in l: yt=l.split('"')[1] if '"' in l else None; continue
        if l.strip().startswith("@    title"):
            if '"' in l: title=l.split('"')[1]
            continue
        e=l.strip().split()
        if len(e)==2:
            try: x.append(float(e[0])); y.append(float(e[1]))
            except ValueError: pass
    return title,x,y,xt,yt

def read_xvg_1set(path): return _parse_xvg_block(Path(path).read_text())

def read_xvg_Nsets(path):
    """Read a multi-set .xvg (sets separated by &). Returns (titles,xs,ys)."""
    titles,xs,ys=[],[],[]
    for blk in Path(path).read_text().split("&"):
        if len(blk)>10:
            t,x,y,_,_=_parse_xvg_block(blk)
            if x: titles.append(t); xs.append(x); ys.append(y)
    return titles,xs,ys

def read_pdb_coords(path):
    x,y,z=[],[],[]
    for l in Path(path).read_text().splitlines():
        if l.startswith("ATOM"):
            c=l[30:56]
            try: x.append(float(c[:8]));y.append(float(c[8:16]));z.append(float(c[16:]))
            except ValueError: pass
    return x,y,z

def read_xpm(path):
    """
    EXACT port of the reference notebook read_xpm (plot_pca_with_FEL.ipynb).
    Correctly extracts the REAL PC axis coordinates (nm) from the x-axis /
    y-axis comment lines — not grid indices.
    """
    letter_val={}
    matrix_vals=[]
    x_coordinates=[]
    y_coordinates=[]
    enter=0
    matrix=0
    with open(path) as f:
        for l in f:
            if l.startswith('"A '):
                enter=1
            if 'x-axis' in l:
                enter=0
                x_coordinates=list(map(float,l.split()[2:-1]))
            if enter==1:
                start=l.find('/*')
                end=l.find('*/')
                if start!=-1 and end!=-1:
                    val=l[start+2:end].strip()[1:-1]
                    letter=l[1]
                    try:
                        letter_val[letter]=float(val)
                    except ValueError:
                        pass
            if 'y-axis' in l:
                matrix=1
                y_coordinates=list(map(float,l.split()[2:-1]))
                continue
            if matrix==1:
                if ',' in l:
                    row=l.split(',')[0][1:-1]
                else:
                    row=l[1:-2]
                row_val=[]
                for letter in row:
                    if letter in letter_val:
                        row_val.append(letter_val[letter])
                if row_val:
                    matrix_vals.append(row_val)
    return matrix_vals,x_coordinates,y_coordinates

def cumulative_variance(vals):
    tot=sum(vals); return [sum(vals[:i+1])/tot*100 for i in range(len(vals))]

def prepare_binding_pocket(gmx_bin,tpr,xtc,pocket_group,index_file,outdir):
    pocket_xtc=outdir/"binding_pocket.xtc"; ref=outdir/"ref.pdb"
    if not run_gmx(gmx_bin,["trjconv","-s",str(tpr),"-f",str(xtc),"-n",str(index_file),"-o",str(pocket_xtc)],stdin_text=f"{pocket_group}\n",cwd=outdir,label="Binding_Pocket trajectory") or not pocket_xtc.exists():
        return None
    if not run_gmx(gmx_bin,["trjconv","-s",str(tpr),"-f",str(xtc),"-n",str(index_file),"-dump","0","-o",str(ref)],stdin_text=f"{pocket_group}\n",cwd=outdir,label="Binding_Pocket reference frame 0") or not ref.exists():
        return None
    natoms=sum(1 for l in ref.read_text().splitlines() if l.startswith(("ATOM","HETATM")))
    print(f"[info] Binding_Pocket atoms written = {natoms} (Group {pocket_group}) - PCA/cosine/FEL computed on these.")
    return pocket_xtc,ref

def covar(gmx_bin,ref,pocket_xtc,outdir,last=None,tag="all"):
    eigvec=outdir/f"eigenvectors_{tag}_ref.trr"; eigval=outdir/f"eigenvals_{tag}_ref.xvg"
    args=["covar","-mwa","-ref","-s",str(ref),"-f",str(pocket_xtc),"-v",str(eigvec),"-o",str(eigval)]
    if last: args+=["-last",str(last)]
    ok=run_gmx(gmx_bin,args,stdin_text="0\n0\n",cwd=outdir,label=f"covar ({tag})")
    return (eigval,eigvec) if ok and eigvec.exists() else (None,None)

def project(gmx_bin,ref,pocket_xtc,eigvec,eigval,outdir,first=1,last=10,out="proj.xvg"):
    proj=outdir/out
    ok=run_gmx(gmx_bin,["anaeig","-proj",str(proj),"-first",str(first),"-last",str(last),"-s",str(ref),"-f",str(pocket_xtc),"-v",str(eigvec),"-eig",str(eigval)],stdin_text="0\n0\n",cwd=outdir,label="anaeig -proj")
    return proj if ok and proj.exists() else None

def cosine_content(gmx_bin,proj,outdir,n=10,out="cc_all.xvg"):
    cc=outdir/out
    ok=run_gmx(gmx_bin,["analyze","-n",str(n),"-cc",str(cc),"-f",str(proj)],stdin_text="",cwd=outdir,label="analyze -cc")
    return cc if ok and cc.exists() else None

def two_d_proj(gmx_bin,ref,pocket_xtc,eigvec,eigval,outdir,a,b,out):
    o=outdir/out
    ok=run_gmx(gmx_bin,["anaeig","-2d",str(o),"-first",str(a),"-last",str(b),"-s",str(ref),"-f",str(pocket_xtc),"-v",str(eigvec),"-eig",str(eigval)],stdin_text="0\n0\n",cwd=outdir,label=f"anaeig -2d {a}{b}")
    return o if ok and o.exists() else None

def three_d_proj(gmx_bin,ref,pocket_xtc,eigvec,eigval,outdir,out="3dproj_123.pdb"):
    o=outdir/out
    ok=run_gmx(gmx_bin,["anaeig","-3d",str(o),"-first","1","-last","3","-s",str(ref),"-f",str(pocket_xtc),"-v",str(eigvec),"-eig",str(eigval)],stdin_text="0\n0\n",cwd=outdir,label="anaeig -3d")
    return o if ok and o.exists() else None

def extreme_structures(gmx_bin,ref,pocket_xtc,eigvec,eigval,outdir):
    o=outdir/"extreme.pdb"
    ok=run_gmx(gmx_bin,["anaeig","-extr",str(o),"-first","1","-last","3","-nframes","30","-s",str(ref),"-f",str(pocket_xtc),"-v",str(eigvec),"-eig",str(eigval)],stdin_text="0\n0\n",cwd=outdir,label="anaeig -extr")
    return o if ok and o.exists() else None

def sham_fel(gmx_bin,proj2d,outdir,temp,tag):
    # Reference line 9:  gmx sham -f 2d_proj.xvg -ls gibbs.xpm -notime
    # The -2d file has exactly two data columns (PC_a, PC_b) and NO time
    # column, so -notime makes sham treat them as the two landscape axes.
    gibbs=outdir/f"gibbs_{tag}.xpm"; prob=outdir/f"prob_{tag}.xpm"
    ok=run_gmx(gmx_bin,["sham","-f",str(proj2d),"-tsham",str(temp),
                        "-ls",str(gibbs),"-lp",str(prob),"-notime"],
               stdin_text="",cwd=outdir,label=f"sham FEL {tag}")
    return gibbs if ok and gibbs.exists() else None

def _write(fig,outdir,name):
    if not _HAS_PLOTLY: return
    fig.write_html(str(outdir/f"{name}.html"))
    try: fig.write_image(str(outdir/f"{name}.png"),scale=2)
    except Exception: pass
    print(f"[plot] OK {name}.html")

def plot_scree(eigval,outdir,dims=10):
    if not _HAS_PLOTLY: print("[plot][skip] scree: plotly not installed"); return
    _,xs,ys,_,_=read_xvg_1set(eigval)
    if not ys: print(f"[plot][skip] scree: no data parsed from {eigval}"); return
    var=cumulative_variance(ys)
    fig=make_subplots(specs=[[{"secondary_y":True}]])
    fig.add_trace(go.Scatter(x=xs[:dims],y=ys[:dims],mode="lines+markers",name="Individual Eigenvalues"),secondary_y=False)
    fig.add_trace(go.Scatter(x=xs[:dims],y=var[:dims],mode="lines+markers",name="Cumulative Variance (%)"),secondary_y=True)
    fig.update_xaxes(dtick=1,range=(1,dims),title_text="Eigenvector index")
    fig.update_yaxes(range=(0,100),title_text="Cumulative (%)",secondary_y=True,showgrid=False)
    fig.update_yaxes(range=(0,ys[0]*1.1),title_text="Eigenvalue (nm^2)",secondary_y=False,showgrid=False)
    fig.update_layout(title="Scree Plot - Eigenvalue Spectrum",font_family="Times New Roman",font_size=16)
    _write(fig,outdir,"scree_plot")

def plot_cosine(cc_holo,outdir,dims=10,cc_apo=None,holo_name="Holo",apo_name="Apo"):
    if not _HAS_PLOTLY: print("[plot][skip] cosine: plotly not installed"); return
    _,xh,yh,_,_=read_xvg_1set(cc_holo)
    fig=go.Figure(data=[go.Bar(x=xh,y=yh,name=holo_name)])
    if cc_apo and Path(cc_apo).exists():
        _,xa,ya,_,_=read_xvg_1set(cc_apo); fig.add_trace(go.Bar(x=xa,y=ya,name=apo_name))
    fig.update_xaxes(dtick=1,range=(0.5,dims+0.5),title_text="Eigenvector (PC) number")
    fig.update_yaxes(range=(0,1),title_text="Cosine content")
    fig.update_layout(title="Cosine Content per PC (gmx analyze -cc; Hess 2002)<br><sub>~1 drift, ~0 oscillatory</sub>",font_family="Times New Roman",font_size=16,barmode="group")
    _write(fig,outdir,"cosine_content")
    if yh: print(f"[cosine] PC1={yh[0]:.3f}  PC2={yh[1]:.3f}" if len(yh)>1 else f"[cosine] PC1={yh[0]:.3f}")

def plot_2d(proj_holo,outdir,a,b,proj_apo=None,holo_name="Holo",apo_name="Apo"):
    if not _HAS_PLOTLY: return
    _,xh,yh,_,_=read_xvg_1set(proj_holo)
    if not xh: print(f"[plot][skip] 2d: no data parsed from {proj_holo}"); return
    data=[go.Scatter(x=xh,y=yh,mode="markers",name=holo_name,marker=dict(color=list(range(len(xh))),colorscale="reds",showscale=True,colorbar=dict(title=f"{holo_name} frame")),text=[f"Frame {i}" for i in range(len(xh))])]
    if proj_apo and Path(proj_apo).exists():
        _,xa,ya,_,_=read_xvg_1set(proj_apo)
        if xa:
            data.append(go.Scatter(x=xa,y=ya,mode="markers",name=apo_name,marker=dict(color=list(range(len(xa))),colorscale="greys",showscale=True,colorbar=dict(title=f"{apo_name} frame",x=1.1)),text=[f"Frame {i}" for i in range(len(xa))]))
            data.append(go.Scatter(x=[np.mean(xa)],y=[np.mean(ya)],mode="markers",name=f"{apo_name} avg",marker=dict(size=14,color="yellow")))
    data.append(go.Scatter(x=[np.mean(xh)],y=[np.mean(yh)],mode="markers",name=f"{holo_name} avg",marker=dict(size=14,color="blue")))
    fig=go.Figure(data=data)
    fig.update_layout(title=f"2D projection along PC{a} and PC{b}",xaxis_title=f"PC{a} (nm)",yaxis_title=f"PC{b} (nm)",font_family="Times New Roman",font_size=16,legend=dict(orientation="h",y=-0.25,x=0.5,xanchor="center"))
    _write(fig,outdir,f"proj_{a}{b}")

def plot_fel(gibbs_xpm,proj2d,outdir,a,b,n_contours=7,step=2):
    if not _HAS_PLOTLY: print("[plot][skip] FEL: plotly not installed"); return
    matrix,xco,yco=read_xpm(gibbs_xpm)
    if not matrix: print(f"[plot][skip] FEL: no matrix parsed from {gibbs_xpm}"); return
    fig=make_subplots(rows=1,cols=2,specs=[[{"type":"heatmap"},{"type":"surface"}]])
    Z=matrix[::-1]
    fig.add_trace(go.Surface(z=Z,x=xco,y=yco,coloraxis="coloraxis"),row=1,col=2)
    fig.add_trace(go.Heatmap(z=Z,x=xco,y=yco,coloraxis="coloraxis",zsmooth="best"),row=1,col=1)
    # trajectory path overlay — the -2d file is a single set with two
    # columns (x=PC_a, y=PC_b), so read_xvg_1set gives exactly those.
    try:
        _,px,py,_,_=read_xvg_1set(proj2d)
        if px and py:
            fig.add_trace(go.Scatter(x=px[::step],y=py[::step],mode="lines+markers",line=dict(color="black",width=0.6),marker=dict(size=4,color=list(range(len(px[::step]))),colorscale="greys"),showlegend=False),row=1,col=1)
    except Exception as _e:
        print(f"[plot] FEL path overlay skipped: {_e}")
    fig.update_layout(title=f"2D + 3D Free Energy Landscape (PC{a}, PC{b})",coloraxis_colorbar_title="G (kJ/mol)",coloraxis_colorscale=COLORSCALE,font_family="Times New Roman",font_size=15,height=600,width=1200,scene=dict(xaxis_title=f"PC{a}",yaxis_title=f"PC{b}",zaxis_title="G (kJ/mol)"))
    _write(fig,outdir,f"FEL_{a}{b}")

def cosine_vs_time(gmx_bin,ref,pocket_xtc,outdir,dims=10,step_ns=10,max_ns=100):
    if not _HAS_PLOTLY: return
    times,cc_pc1,cc_pc2=[],[],[]
    for end in range(step_ns,max_ns+1,step_ns):
        seg=outdir/f"seg_0_{end}.xtc"
        if not run_gmx(gmx_bin,["trjconv","-s",str(ref),"-f",str(pocket_xtc),"-b","0","-e",str(end),"-tu","ns","-o",str(seg)],stdin_text="0\n",cwd=outdir,label=f"segment 0-{end}ns") or not seg.exists():
            continue
        ev,evec=covar(gmx_bin,ref,seg,outdir,tag=f"0_{end}")
        if ev is None: seg.unlink(missing_ok=True); continue
        pj=project(gmx_bin,ref,seg,evec,ev,outdir,first=1,last=dims,out=f"proj_0_{end}.xvg")
        if pj:
            cc=cosine_content(gmx_bin,pj,outdir,n=dims,out=f"cc_0_{end}.xvg")
            if cc:
                _,xx,yy,_,_=read_xvg_1set(cc)
                if yy:
                    times.append(end); cc_pc1.append(yy[0]); cc_pc2.append(yy[1] if len(yy)>1 else np.nan)
        for f in [seg,ev,evec,pj]:
            try: Path(f).unlink(missing_ok=True)
            except Exception: pass
    if times:
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=times,y=cc_pc1,mode="lines+markers",name="PC1"))
        fig.add_trace(go.Scatter(x=times,y=cc_pc2,mode="lines+markers",name="PC2"))
        fig.update_layout(title="Cosine content vs simulation length",xaxis_title="Window end (ns)",yaxis_title="Cosine content",yaxis_range=[0,1],font_family="Times New Roman",font_size=16)
        _write(fig,outdir,"cosine_vs_time")

def analyze_one(gmx_bin,tpr,xtc,pocket_group,index_file,outdir,temp,dims,do_time=False):
    outdir.mkdir(parents=True,exist_ok=True); res={}
    print("\n"+"="*70); print(f"  PCA + FEL  ->  {outdir}"); print("="*70)
    prep=prepare_binding_pocket(gmx_bin,tpr,xtc,pocket_group,index_file,outdir)
    if prep is None: print("[error] Binding_Pocket preparation failed; aborting"); return None
    pocket_xtc,ref=prep
    eigval,eigvec=covar(gmx_bin,ref,pocket_xtc,outdir,tag="all")
    if eigvec is None: print("[error] covar failed; aborting"); return None
    res["eigenvalues"]=str(eigval); plot_scree(eigval,outdir,dims=dims)
    proj=project(gmx_bin,ref,pocket_xtc,eigvec,eigval,outdir,first=1,last=dims,out="proj.xvg")
    if proj:
        cc=cosine_content(gmx_bin,proj,outdir,n=dims,out="cc_all.xvg")
        if cc: res["cosine"]=str(cc); plot_cosine(cc,outdir,dims=dims)
    for a,b in [(1,2),(1,3),(2,3)]:
        # -2d file → PC_a vs PC_b as two columns (NO time). This is what the
        # reference feeds to sham (line 9: sham -f 2d_proj.xvg -notime).
        p2d=two_d_proj(gmx_bin,ref,pocket_xtc,eigvec,eigval,outdir,a,b,f"2dproj_{a}{b}.xvg")
        if p2d:
            plot_2d(p2d,outdir,a,b)
            # FEL from the -2d file + -notime (reference line 9 exactly)
            gibbs=sham_fel(gmx_bin,p2d,outdir,temp,f"{a}{b}")
            if gibbs: plot_fel(gibbs,p2d,outdir,a,b)
    three_d_proj(gmx_bin,ref,pocket_xtc,eigvec,eigval,outdir)
    extr=extreme_structures(gmx_bin,ref,pocket_xtc,eigvec,eigval,outdir)
    if extr:
        res["porcupine"]=str(extr)
        print("[porcupine] extreme.pdb written. In PyMOL:")
        print("   load extreme.pdb ; run modevectors.py")
        print("   split_states extreme")
        print("   modevectors extreme_0001, extreme_0030, notail=1, head=0.4")
    if do_time: cosine_vs_time(gmx_bin,ref,pocket_xtc,outdir,dims=dims)
    print(f"[done] {outdir}\n")
    return res,pocket_xtc,ref,eigvec,eigval

def compare_holo_apo(gmx_bin,holo_pocket_xtc,holo_ref,apo_pocket_xtc,apo_ref,outdir,holo_name,apo_name):
    cmp=outdir/"compare"; cmp.mkdir(parents=True,exist_ok=True)
    print("\n"+"="*70); print(f"  Comparison  {holo_name} vs {apo_name}  ->  {cmp}"); print("="*70)
    concat=cmp/"concatenated.xtc"
    if not run_gmx(gmx_bin,["trjcat","-f",str(holo_pocket_xtc),str(apo_pocket_xtc),"-o",str(concat),"-cat"],stdin_text="",cwd=cmp,label="trjcat") or not concat.exists():
        print("[compare][error] trjcat failed; skipping"); return
    eigvec=cmp/"eig_concat.trr"; eigval=cmp/"eigval_concat.xvg"
    if not run_gmx(gmx_bin,["covar","-mwa","-ref","-s",str(apo_ref),"-f",str(concat),"-v",str(eigvec),"-o",str(eigval)],stdin_text="0\n0\n",cwd=cmp,label="covar (concat, apo ref)") or not eigvec.exists():
        print("[compare][error] covar failed; skipping"); return
    projs={}
    for name,xtc in [(holo_name,holo_pocket_xtc),(apo_name,apo_pocket_xtc)]:
        for a,b in [(1,2),(1,3),(2,3)]:
            o=cmp/f"2dproj_{a}{b}_{name}.xvg"
            run_gmx(gmx_bin,["anaeig","-2d",str(o),"-first",str(a),"-last",str(b),"-s",str(apo_ref),"-f",str(xtc),"-v",str(eigvec),"-eig",str(eigval)],stdin_text="0\n0\n",cwd=cmp,label=f"proj {name} {a}{b}")
            projs[(name,a,b)]=o
    for a,b in [(1,2),(1,3),(2,3)]:
        ph=projs.get((holo_name,a,b)); pa=projs.get((apo_name,a,b))
        if ph and Path(ph).exists():
            plot_2d(ph,cmp,a,b,proj_apo=pa,holo_name=holo_name,apo_name=apo_name)
    run_gmx(gmx_bin,["anaeig","-inpr",str(cmp/"inprod_1-10.xpm"),"-first","1","-last","10","-v",str(eigvec),"-v2",str(eigvec),"-eig",str(eigval),"-eig2",str(eigval),"-s",str(apo_ref)],stdin_text="0\n0\n",cwd=cmp,label="inner-product overlap")
    print(f"[compare] done -> {cmp}")

def main():
    ap=argparse.ArgumentParser(description="Professional GROMACS PCA + FEL")
    ap.add_argument("--tpr",required=True); ap.add_argument("--xtc",required=True)
    ap.add_argument("--outdir",default="PCA_out")
    ap.add_argument("--index",default="index.ndx",help="GROMACS index file containing [ Binding_Pocket ] as Group 21")
    ap.add_argument("--group",type=int,default=21,help="Binding_Pocket group number; default 21")
    ap.add_argument("--temp",type=float,default=310.0); ap.add_argument("--dims",type=int,default=10)
    ap.add_argument("--gmx-bin",default="gmx")
    ap.add_argument("--cosine-vs-time",action="store_true")
    ap.add_argument("--ref-tpr",default=None); ap.add_argument("--ref-xtc",default=None)
    ap.add_argument("--ref-index",default=None,help="Index file for apo/reference system; defaults to --index")
    ap.add_argument("--holo-name",default="Holo"); ap.add_argument("--apo-name",default="Apo")
    args=ap.parse_args()
    print("="*70)
    if _HAS_PLOTLY:
        import plotly
        print(f"[OK] plotly {plotly.__version__} found - interactive HTML plots WILL be generated.")
    else:
        print("[!!! WARNING !!!] plotly is NOT installed on this machine.")
        print("                 NO PLOTS will be generated (only gmx data files).")
        print("                 Fix with:  pip install plotly kaleido")
        print("                 Then re-run this exact command.")
    print("="*70)
    gmx_bin=args.gmx_bin; tpr=Path(args.tpr).resolve(); xtc=Path(args.xtc).resolve(); outdir=Path(args.outdir).resolve()
    index_file=Path(args.index).resolve()
    try:
        pocket_group,n_pocket=detect_binding_pocket_group(gmx_bin,tpr,index_file,args.group)
    except Exception as e:
        print(f"[error] Binding_Pocket index validation failed: {e}")
        sys.exit(2)
    holo=analyze_one(gmx_bin,tpr,xtc,pocket_group,index_file,outdir,args.temp,args.dims,do_time=args.cosine_vs_time)
    if not holo: print("[error] holo analysis failed"); sys.exit(1)
    _,holo_pocket_xtc,holo_ref,_,_=holo
    if args.ref_tpr and args.ref_xtc:
        rt,rx=Path(args.ref_tpr),Path(args.ref_xtc)
        if rt.exists() and rx.exists():
            apo_index=Path(args.ref_index).resolve() if args.ref_index else index_file
            try:
                apo_group,n=detect_binding_pocket_group(gmx_bin,rt,apo_index,args.group)
            except Exception as e:
                print(f"[WARNING] apo Binding_Pocket index validation failed: {e} - comparison skipped")
                apo_group=None
            apo_out=outdir.parent/(outdir.name+"_apo")
            apo=analyze_one(gmx_bin,rt,rx,apo_group,apo_index,apo_out,args.temp,args.dims) if apo_group is not None else None
            if apo:
                _,apo_pocket_xtc,apo_ref,_,_=apo
                compare_holo_apo(gmx_bin,holo_pocket_xtc,holo_ref,apo_pocket_xtc,apo_ref,outdir,args.holo_name,args.apo_name)
        else:
            print("[WARNING] reference files not found - comparison skipped")
    print("\n[ALL DONE]")

if __name__=="__main__":
    main()
