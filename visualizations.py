import py3Dmol
import numpy as np

def pdb_from_ca(sequence, coords):
    """
    Build a CA-only PDB string from:
      sequence: string of length L (single-letter AAs)
      coords: (L, 3) numpy array of CA positions
    """
    AA3 = {
        'A':'ALA','C':'CYS','D':'ASP','E':'GLU','F':'PHE','G':'GLY','H':'HIS','I':'ILE',
        'K':'LYS','L':'LEU','M':'MET','N':'ASN','P':'PRO','Q':'GLN','R':'ARG','S':'SER',
        'T':'THR','V':'VAL','W':'TRP','Y':'TYR'
    }
    lines = []
    for i,(aa,(x,y,z)) in enumerate(zip(sequence, coords), start=1):
        lines.append(
            f"ATOM  {i:5d}  CA  {AA3[aa]:>3} A{i:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C"
        )
    # Add CA–CA bonds for consecutive residues
    for i in range(1, len(sequence)):
        lines.append(f"CONECT{i:5d}{i+1:5d}")
    lines.append("END")
    return "\n".join(lines)
    

def color_by_error(true_ca, pred_ca, max_err=4.0):
    """
    Static scale:
        0 Å   = green (#00ff00)
        1 Å   = red   (#ff0000)
    Errors > 1 Å are clipped to red.
    """
    err = np.linalg.norm(true_ca - pred_ca, axis=-1)  # (L,)
    err_norm = np.clip(err / max_err, 0, 1)           # FIXED SCALE

    colors = []
    for v in err_norm:
        r = int(255 * v)
        g = int(255 * (1 - v))
        b = 0
        colors.append(f"#{r:02x}{g:02x}{b:02x}")
    return colors


def show_overlay_pdb_vs_pred(pdb_file=None, sequence=None, coords_true=None, coords_pred=None, rainbow=False):
    view = py3Dmol.view(width=800, height=600)

    if pdb_file is not None:
        # True structure ribbon
        view.addModel(open(pdb_file).read(), 'pdb')
        view.setStyle({'model':0}, {'cartoon': {'color':'spectrum' if rainbow else 'gray', 'opacity':0.89}})

    if coords_pred is not None:
        # Predicted CA trace
        pdb_pred = pdb_from_ca(sequence, coords_pred)
        view.addModel(pdb_pred, 'pdb')
    
        # Color by per-residue error ---
        err_colors = color_by_error(coords_true, coords_pred)
        for i, col in enumerate(err_colors, start=1):
            #view.setStyle({'model':1, 'serial':i}, {'sphere': {'color':col, 'radius':0.8}})
            view.addStyle({'model':1, 'serial':i}, {'stick': {'color':col, 'radius':0.25}})
            if i > 1:
                view.addLine({"start":{'model':1,'serial':i-1},
                              "end":  {'model':1,'serial':i}},
                             {'color':col, 'radius':0.4})

    view.zoomTo()
    return view.show()