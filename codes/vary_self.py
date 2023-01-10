#!/usr/bin/env python
from TS_fit import de_fit

filenames=['iPSC','iPSC_timeshifted','Partial','Partial_timeshifted']
combi=['IIAI','IIIA','IINA','IINI','IIAN','IIIN','IINN']

for i in filenames:
    for f in combi:
        de_fit(f,i)
