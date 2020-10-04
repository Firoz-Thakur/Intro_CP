import numpy as np
x=np.random.uniform(low=0, high=10, size=(50,2))
import pandas as pd
x=pd.DataFrame(x)
x.to_csv('train.csv')