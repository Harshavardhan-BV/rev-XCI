#%%
from common_fn import *
from itertools import product
#%%
txt='vary_all'
filenames=['iPSC','iPSC_timeshifted','Partial','Partial_timeshifted']
x=['A','I','N']
combi=np.array(tuple(product(x,repeat=4)))
combi=np.apply_along_axis(''.join,1,combi)
combi2d=np.reshape(combi,(9,9))
print(combi2d)
axs=np.array(tuple(product(x,repeat=2)))
axs=np.apply_along_axis(''.join,1,axs)
print(axs)
lbl=['Self','Cross']
for i in filenames:
        timeseries(combi,i,txt)
        hmap(combi2d,axs,lbl,i,txt)
