import sys
from config import CASES,PRIMARY,CONTROLS
methods=PRIMARY if sys.argv[1]=='primary' else CONTROLS
for case in CASES:
 for rep in range(1,21):
  shift=rep%len(methods)
  for method in methods[shift:]+methods[:shift]:print(case,rep,method)
