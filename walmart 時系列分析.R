##### Rでデータを読み込む
#データの読み込み
train_data<-read.csv("train.csv")
head(data)

#データの全体像をつかむ
summary(train_data)

#列名の一覧
colnames(train_data)

#行数・列数
dim(train_data)

#データの可視化
library(forecast)
library(ggplot2)

plot(train_data$Weekly_Sales,type="l")
                     
# ts関数を用いて、データが時系列データである事を R に認識させる。
train_data_ts <- ts(train_data$Weekly_Sales, start=c(2010,2), frequency=12)
ts.plot(train_data_ts,gpars=list(lt=c(1,2,3,3),col=c(1,2,4,4)))

# decompose関数を使いトレンド+周期変動、季節変動、不規則変動をtrend、seasonal、randomとして分解する
plot(decompose(train_data_ts))

# ADF検定を行い、時系列データが定常過程かどうか確認する
library(tseries)
adf.test(train_data_ts)
install.packages("sem", dependencies=TRUE)

# 自己相関・偏自己相関を確認する

# 自己相関
acf(train_data_ts,lag.max=96,ylim=c(-.2,1.0)) 

# 偏自己相関
pacf(train_data_ts,lag.max=96,ylim=c(-.2,1.0)) 


#　データの分割
train_store1 <- subset(train_data, Store=='1')
train_store1_1 <- subset(train_store1, Dept=='1')
train_store1_1_ts <- ts(train_store1_1$Weekly_Sales, start=min(train_store1_1$Date),end = max((train_store1_1$Date))
ts.plot(train_store1_1_ts,gpars=list(lt=c(1,1,3),col=c(2,1,4)))

train_store1_1_a <- train_store1_1[1:100,]
test_store1_1_a <- train_store1_1[100:143,]

#分割したデータが時系列データである事を R に認識させる
train_store1_1_a_ts <- ts(train_store1_1_a$Weekly_Sales,frequency=52,start=c(2010,2),end = c(2011,12))
test_store1_1_a_ts <- ts(test_store1_1_a$Weekly_Sales,frequency=52,start=c(2011,12),end = c(2012,10))

ts.plot(train_store1_1_a_ts,gpars=list(lt=c(1,1,3),col=c(2,1,4)))                         
      
ts.plot(test_store1_1_a_ts,gpars=list(lt=c(1,1,3),col=c(2,1,4)))

                 
library(forecast)

# ARモデル
AR <-ar(train_store1_1_a_ts)
AR

#predict関数を使って、43週先まで予測
pred <- predict(AR,n.ahead=43)
ts.plot(test_store1_1_a_ts,train_store1_1_a_ts,pred$pred,gpars=list(lt=c(1,1,3),col=c(2,1,4)))


# RMSEを計算
#RMSE = function(pred$pred, test_store1_1_a$Weekly_Sales){
#  sqrt(mean((pred$pred - test_store1_1_a$Weekly_Sales)^2))
#}

#MAモデル
MA <-ma(train_store1_1_a_ts,3)
MA
#predict関数を使って、43週先まで予測
pred <- predict(MA,n.ahead=43)
ts.plot(test_store1_1_a_ts,train_store1_1_a_ts,pred$pred,gpars=list(lt=c(1,2,3,3),col=c(1,2,4,4)))

# ARMAモデル
#パラメーターの推定
ARIMA_para <- auto.arima(train_store1_1_a_ts, ic="aic", stepwise=T, trace=T) 
ARIMA_para


ARIMA <-arima(train_store1_1_a_ts,order = c(0,0,1))
ARIMA

#predict関数を使って、43週先まで予測
pred_arima <- predict(ARIMA,n.ahead=43)
ts.plot(test_store1_1_a_ts,train_store1_1_a_ts,pred_arima$pred,gpars=list(lt=c(1,1,3),col=c(2,1,4)))


# SARIMAモデル
#パラメーターの推定
SARIMA_para <- auto.arima(train_store1_1_a_ts, ic="aic",stepwise=T, trace=T,seasonal = T) 
SARIMA_para
, # FALSEでARIMA（デフォルト）、TRUEでSARIMA