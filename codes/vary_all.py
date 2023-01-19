#!/usr/bin/env python
from TS_fit import de_fit
import numpy as np
from itertools import product

filenames=['iPSC','iPSC_timeshifted','Partial','Partial_timeshifted']
x=['A','I','N']
combi=np.array(tuple(product(x,repeat=4)))
combi=np.apply_along_axis(''.join,1,combi)

for i in filenames:
    for f in combi:
        de_fit(f,i)
