#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%%
#create a random dataframe 
df = pd.DataFrame(np.random.randn(10,4),columns=['a','b','c','d'])

# %%
df.a[0]

#%%
def testing():
    return 3

# %%
lsita = [testing()]
# %%
lsita