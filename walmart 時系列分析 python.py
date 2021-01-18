#!/usr/bin/env python
# coding: utf-8

# https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/data

# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from sklearn.metrics import confusion_matrix, accuracy_score,precision_score,recall_score, f1_score,classification_report
from random import randint, shuffle
import os
import seaborn as sns


# In[4]:


train_data = pd.read_csv('walmart/train.csv')


# In[5]:


test_data = pd.read_csv('walmart/test.csv')


# In[6]:


stores_data = pd.read_csv('walmart/stores.csv')


# In[7]:


sampleSubmission_data = pd.read_csv('walmart/sampleSubmission.csv')


# In[8]:


features_data = pd.read_csv('walmart/features.csv')


# # データ詳細

# In[9]:


train_data.head()


# In[89]:


train_data.tail()


# In[10]:


test_data.head()


# In[11]:


stores_data.head()


# In[12]:


features_data.head()


# In[13]:


print(train_data.shape)
print(test_data.shape)
print(stores_data.shape)
print(features_data.shape)


# # 欠損値の確認

# In[14]:


# 関数化
def isnull_check(data):
    col_names = data.columns
    for col_name in col_names:
        missing_num = sum(pd.isnull(data[col_name]))
    
        
        
    return print(col_names,missing_num)


# In[15]:


isnull_check(data = train_data)
isnull_check(data = test_data)
isnull_check(data = sampleSubmission_data)
isnull_check(data = stores_data)
isnull_check(data = features_data)


# # 欠損値はなし

# # データの可視化

# In[16]:


plt.xlim(0, 300000)                 # (1) x軸の表示範囲
# plt.ylim(0, 30)                 # (2) y軸の表示範囲
plt.title("Store-Weekly_Sales", fontsize=20)  # (3) タイトル
plt.xlabel("Weekly_Sales", fontsize=20)            # (4) x軸ラベル
plt.ylabel("Store", fontsize=20)      # (5) y軸ラベル
plt.grid(True)                            # (6) 目盛線の表示
plt.tick_params(labelsize = 12)    # (7) 目盛線のラベルサイズ 
 
# グラフの描画
plt.hist(train_data['Weekly_Sales'] , alpha=0.5, color= 'c') #(8) ヒストグラムの描画
plt.show()
 


# In[17]:


Weekly_Sales_by_IsHoliday =     train_data.groupby('IsHoliday').aggregate({'Weekly_Sales': np.mean}).reset_index()
Weekly_Sales_by_IsHoliday.plot.bar(x='IsHoliday')


# In[18]:


train_date_sales = pd.pivot_table(
    train_data, 
    columns=["Date"],
    index="Store",
    values="Weekly_Sales",
    aggfunc=np.mean).reset_index()


# In[19]:


train_date_sales.head()


# In[20]:


train_date_sales.T.plot(color='blue', alpha=1, legend=False)


# In[21]:


train_data2 = train_data.copy()


# In[22]:


train_data2['Date'] = pd.to_datetime(train_data2['Date'])


# In[23]:


train_data2.plot(x = 'Date', y = 'Weekly_Sales')


# StoreとDeptでデータを分けて、ランダムフォレストで予測するための前処理を行う

# In[24]:


store1_tra = train_data[train_data['Store'] == 1] 


# In[25]:


store1_1_tra = store1_tra[store1_tra['Dept'] == 1]


# In[25]:


store1_1_tra.head()


# In[26]:


store1_1_tra['Date'] = pd.to_datetime(store1_1_tra['Date'])


# In[27]:


store1_1_tra.plot(x = 'Date', y = 'Weekly_Sales') 


# ランダムフォレストで予測するために、週と月の特徴量を作成する

# In[13]:


def get_week(x):
    return x.week


# In[14]:


store1_1_tra.head()


# In[15]:


store1_1_tra['week'] = store1_1_tra['Date'].apply(get_week)


# In[16]:


store1_1_tra.head()


# In[17]:


def get_month(x):
    return x.month


# In[18]:


store1_1_tra['month'] = store1_1_tra['Date'].apply(get_month)


# In[19]:


store1_1_tra.head()


# In[20]:


store1_1_tra_a = pd.get_dummies(store1_1_tra, columns = ['IsHoliday','week','month'])


# In[21]:


store1_1_tra_a.head()


# 目的関数と特徴量を定義する

# In[22]:


target_col = 'Weekly_Sales'


# In[24]:


exclude_cols = ['Store','Dept','Date','Weekly_Sales']


# In[28]:


feature_cols = []
for col in store1_1_tra_a.columns:
    if col not in exclude_cols:
        feature_cols.append(col)


# In[29]:


X = store1_1_tra_a[feature_cols]
y = store1_1_tra_a[target_col]


# In[31]:


X.head()


# In[32]:


y.head()


# データを学習とテストデータに分ける

# In[34]:


X.shape


# データ数は143なので、
# 143 * 0.7 ≒100

# In[35]:


train_rows = 100
X_train = X[:train_rows]
X_test = X[train_rows:]
y_train = y[:train_rows]
y_test = y[train_rows:]


# In[38]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# ランダムフォレストで学習

# In[39]:


rf = RandomForestRegressor(random_state=1234)
# train_dataで学習
rf.fit(X_train, y_train)
# 予測する
y_pred = rf.predict(X_test)


# RMSEで精度算出

# In[40]:


mse = mean_squared_error(y_pred, y_test)
print('RSME: ', np.sqrt(mse))


# 予測値と実際の値をグラフ化する

# In[63]:


y_graph = pd.DataFrame({'y_test': y_test, 'y_pred':y_pred})


# In[65]:


y_graph.plot()


# # Storeごとにデータを分けてランダムフォレストで予測する

# In[72]:


max(train_data["Dept"])


# 最初にstore1のDept1のデータを用意する

# In[78]:


store1_tra_sales = store1_tra[store1_tra["Dept"]==1]["Weekly_Sales"]


# In[79]:


store1_tra_sales.reset_index(drop=True)


# 次にstore1のDept2以降を連結していく

# In[80]:


for Dept_i in range(2, 100):
    store1_tra_temp = store1_tra[store1_tra['Dept'] == Dept_i]['Weekly_Sales']
    store1_tra_tepm_series = store1_tra_temp.reset_index(drop=True)
    store1_tra_sales = pd.concat([store1_tra_sales, store1_tra_tepm_series], axis = 1)    


# In[82]:


store1_tra_sales.head()


# In[91]:


# 欠損の確認
store1_tra_sales.isnull().sum()


# In[92]:


# 欠損を0で穴埋め
store1_tra_sales = store1_tra_sales.fillna(0)


# データを学習とテストデータに分ける

# In[93]:


store1_tra_sales.shape


# In[94]:


# target変数が変わるのでYと定義する
Y = store1_tra_sales


# データ数は143なので、
# 143 * 0.7 ≒100

# In[95]:


train_rows = 100
X_train = X[:train_rows]
X_test = X[train_rows:]
Y_train = Y[:train_rows]
Y_test = Y[train_rows:]


# In[96]:


print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# ランダムフォレストで学習

# In[105]:


rf2 = RandomForestRegressor(random_state=1234)
# train_dataで学習
rf2.fit(X_train, Y_train)
# 予測する
Y_pred = rf2.predict(X_test)


# RMSEで精度算出

# In[114]:


mse2 = mean_squared_error(Y_pred, Y_test)
print('RSME: ', np.sqrt(mse2))


# 各DeptごとのRSMEを算出

# In[108]:


# Y_testをnpに変換
Y_test_ar = np.array(Y_test)


# In[112]:


for i in range(Y_pred.shape[1]):
    mse3 = mean_squared_error(Y_pred[:,i], Y_test_ar[:,i])
    print('RMSE:' + 'Dept' + str(i) + ':',np.sqrt(mse3))


# 予測値と実際の値をグラフ化する

# In[120]:


i = 1
Y_graph = pd.DataFrame({'Y_test_ar': Y_test_ar[:,i], 'Y_pred':Y_pred[:,i]})


# In[123]:


for i in range(2,98):
    Y_graph = pd.DataFrame({'Y_test_ar': Y_test_ar[:,i], 'Y_pred':Y_pred[:,i]})
    Y_graph.plot()


# # 時系列分析

# In[26]:


# 基本のライブラリを読み込む
import numpy as np
import pandas as pd
from scipy import stats

# グラフ描画
from matplotlib import pylab as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()

# グラフを横長にする
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6

# 統計モデル
import statsmodels.api as sm


# In[27]:


store1_1_tra_b = store1_1_tra[['Date','Weekly_Sales']]


# In[28]:


store1_1_tra_b.head()


# In[74]:


ts = store1_1_tra_b['Weekly_Sales']


# In[41]:


ts.shape


# In[30]:


plt.plot(ts)


# In[75]:


# 自己相関を求める
ts_acf = sm.tsa.stattools.acf(ts, nlags=40)
ts_acf


# In[37]:


# 偏自己相関
ts_pacf = sm.tsa.stattools.pacf(ts, method='ols')
ts_pacf


# In[63]:


#  自己相関と偏自己相関のグラフ
fig = plt.figure(figsize=(14,10))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(ts, lags=70, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(ts, lags=70, ax=ax2)


# データの分割

# In[65]:


train_rows = 100
ts_train = ts[:train_rows]
ts_test = ts[train_rows:]


# In[66]:


print(ts_train.shape)
print(ts_test.shape)


# ARIMA関数を適用してみる

# In[76]:


# 次数は適当
from statsmodels.tsa.arima_model import ARIMA
ARIMA_3_1_2 = ARIMA(ts_train, order=(3, 1, 2)).fit(dist=False)
ARIMA_3_1_2.params


# In[84]:


ts_train.head()


# In[83]:


ts_test.tail()


# In[91]:


# 予測する
pred = ARIMA_3_1_2.predict('2010-02-05','2012-10-26')


# # Prophetによる時系列予測

# In[25]:


from fbprophet import Prophet

