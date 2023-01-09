from TS_fit import de_fit

filenames=['iPSC','iPSC_timeshifted','Partial','Partial_timeshifted']
combi=['AAAA','AAII','AAAI','AAIA','AANN','AANI','AAIN','AAAN','AANA']

for i in filenames:
    for f in combi:
        de_fit(f,i)