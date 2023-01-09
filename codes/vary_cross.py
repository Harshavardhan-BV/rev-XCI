from TS_fit import de_fit

filenames=['iPSC','iPSC_timeshifted','Partial','Partial_timeshifted']
combi=['AIAA','IAAA','IIAA','AIII','IAII','IIII']

for i in filenames:
    for f in combi:
        de_fit(f,i)