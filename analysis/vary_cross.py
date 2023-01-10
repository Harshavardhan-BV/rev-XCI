#%%
from common_fn import *
#%%
txt='vary_cross-AA'
filenames=['iPSC','iPSC_timeshifted','Partial','Partial_timeshifted']
combi=['AAAA','AIAA','IAAA','IIAA']
combi2d=np.array([['AAAA','AIAA'],['IAAA','IIAA']])
axs=['A','I']
for i in filenames:
        timeseries(combi,i,txt)
        hmap(combi2d,axs,i,txt)
# %%
#%%
txt='vary_cross-II'
combi=['AAII','AIII','IAII','IIII']
combi2d=np.array([['AAII','AIII'],['IAII','IIII']])
for i in filenames:
        timeseries(combi,i,txt)
        hmap(combi2d,axs,i,txt)