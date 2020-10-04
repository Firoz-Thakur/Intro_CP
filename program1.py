
import pandas as pd
import numpy as np
df=pd.read_csv("train.csv")
df.describe()
print(df.head())



train_label=df.pop('tip')

test_label=test_data.pop('PE')
print(train_label.shape)
print(test_label.shape)

mf = [[['gaussmf',{'mean':np.mean(np.arange(0,4)),'sigma':np.std(np.arange(0,4))}],['gaussmf',{'mean':np.mean(np.arange(3,7)),'sigma':np.std(np.arange(3,7))}],['gaussmf',{'mean':np.mean(np.arange(6,10)),'sigma':np.std(np.arange(6,10))}]],
      [['gaussmf',{'mean':np.mean(np.arange(0,4)),'sigma':np.std(np.arange(0,4))}],['gaussmf',{'mean':np.mean(np.arange(3,7)),'sigma':np.std(np.arange(3,7))}],['gaussmf',{'mean':np.mean(np.arange(6,10)),'sigma':np.std(np.arange(6,10))}]]]
        
from membership import membershipfunction
mfc = membershipfunction.MemFuncs(mf)

import anfis
anf = anfis.ANFIS(train_data,train_label, mfc)

pred_train=anf.trainHybridJangOffLine(epochs=20)

train_label=np.reshape(train_label,[1,len(train_label)])
test_label=np.reshape(test_label,[1,len(test_label)])
print(train_label.shape)
print(test_label.shape)

error=np.mean((pred_train-train_label)**2)

print(error)
anf.plotErrors()
anf.plotResults()





