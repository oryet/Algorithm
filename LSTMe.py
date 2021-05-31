import numpy
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
import  pandas as pd
import  os
from tensorflow.keras.models import Sequential, load_model
from sklearn.preprocessing import MinMaxScaler


# 对数据进行处理
def create_dataset(dataset, look_back):
#这里的look_back与timestep相同
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return numpy.array(dataX),numpy.array(dataY)


def reshapeData(csvDath):
    # csvpath = r'F:\Work\科研项目\2020.3\河北\江苏曲线\01\cuv.csv'
    dataframe = pd.read_csv(csvDath, usecols=[1])  # 'airline-passengers.csv'
    # train_seq = readDataCsv(csvpath, 1)
    dataset = dataframe.values
    # dataset = train_seq
    # 将整型变为float
    dataset = dataset.astype('float32')
    #归一化 在下一步会讲解
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    train_size = len(dataset) - 96*3
    trainlist = dataset[:train_size]
    testlist = dataset[train_size:]

    #训练数据太少 look_back并不能过大
    look_back = 96
    trainX,trainY  = create_dataset(trainlist,look_back)
    testX,testY = create_dataset(testlist,look_back)

    trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1] ,1 ))

    return scaler, trainX, testX, trainY, testY



# create and fit the LSTM network
# LSTM的输入为 [samples, timesteps, features]
# 这里的timesteps为步数，features为维度 这里我们的数据是1维的
def LSTMfit(trainX, trainY):
    model = Sequential()
    model.add(LSTM(4, input_shape=(None,1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
    model.save(os.path.join("DATA","JS01Test" + ".h5"))

# make predictions
def makePredic(trainX, testX):
    model = load_model(os.path.join("DATA","JS01Test" + ".h5"))
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    return trainPredict, testPredict

#反归一化
def normalizationData(scaler, trainY, trainPredict, testY, testPredict):
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform(trainY)
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform(testY)
    return trainY, trainPredict, testY, testPredict


def pltData(trainY, trainPredict, testY, testPredict):
    plt.figure()

    plt.subplot(211)
    plt.plot(trainY)
    plt.plot(trainPredict[1:])

    plt.subplot(212)
    plt.plot(testY)
    plt.plot(testPredict[1:])
    plt.show()

if __name__ == '__main__':
    # csvData = 'airline-passengers.csv'
    csvData = 'F:\Work\科研项目\\2020.3\河北\江苏曲线\\01\cuv.csv'
    scaler, trainX, testX, trainY, testY = reshapeData(csvData)

    #LSTMfit(trainX, trainY)

    trainPredict, testPredict = makePredic(trainX, testX)
    trainY, trainPredict, testY, testPredict = normalizationData(scaler, trainY, trainPredict, testY, testPredict)
    pltData(trainY, trainPredict, testY, testPredict)