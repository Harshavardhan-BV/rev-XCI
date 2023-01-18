#!/usr/bin/env python
from TS_fit import de_fit

filenames=['iPSC','iPSC_timeshifted','Partial','Partial_timeshifted']
combi=['AAAA','AIAA','IAAA','IIAA','AAII','AIII','IAII','IIII']

for i in filenames:
    for f in combi:
        de_fit(f,i)
