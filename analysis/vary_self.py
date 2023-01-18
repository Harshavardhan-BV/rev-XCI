#%%
from common_fn import *
#%%
txt='vary_self'
filenames=['iPSC','iPSC_timeshifted','Partial','Partial_timeshifted']
combi=['IIAA','IIAI','IIAN','IIIA','IIII','IIIN','IINA','IINI','IINN']
combi2d=np.array([['IIAA','IIAI','IIAN'],['IIIA','IIII','IIIN'],['IINA','IINI','IINN']])
axs=['A','I','N']
lbl=['Xa','Xi']
#%%
for i in filenames:
        timeseries(combi,i,txt)
        hmap(combi2d,axs,lbl,i,txt)
# %%
