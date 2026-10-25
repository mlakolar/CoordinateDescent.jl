import sys
from pathlib import Path
import numpy as np
from common import load_problem,prepare,save_array
for rep in range(1,6):
    root=Path(sys.argv[1])/f"medium-{rep}"
    d=load_problem(root);X,Y,_,C,_=prepare(d)
    weights=np.maximum(np.linalg.norm(d["B0"],axis=1),np.sqrt(np.finfo(float).eps))**-2
    maximum=np.max(np.linalg.norm(C@d["V0"],axis=1)/weights)
    save_array(root/"adaptive_weights.bin",weights)
    (root/"adaptive_max.txt").write_text(str(float(maximum))+"\n")
