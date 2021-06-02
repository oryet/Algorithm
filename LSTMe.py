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
def createTrainDataset(dataset, look_back):
#这里的look_back与timestep相同
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return numpy.array(dataX),numpy.array(dataY)


def readDataset(csvDath):
    dataframe = pd.read_csv(csvDath, usecols=[1])  # 'airline-passengers.csv'
    # train_seq = readDataCsv(csvpath, 1)
    dataset = dataframe.values
    # dataset = train_seq
    # 将整型变为float
    dataset = dataset.astype('float32')
    # 归一化 在下一步会讲解
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    return dataset


# 对数据进行处理
def createPredictDataset(dataset, look_back):
#这里的look_back与timestep相同
    dataX= []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back)]
        dataX.append(a)
    return numpy.array(dataX)


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
    trainX,trainY  = createTrainDataset(trainlist,look_back)
    testX,testY = createTrainDataset(testlist,look_back)

    trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1] ,1 ))

    return scaler, trainX, testX, trainY, testY



# create and fit the LSTM network
# LSTM的输入为 [samples, timesteps, features]
# 这里的timesteps为步数，features为维度 这里我们的数据是1维的
def LSTMfit(trainX, trainY, saveName):
    model = Sequential()
    model.add(LSTM(4, input_shape=(None,1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
    model.save(os.path.join("DATA", saveName + ".h5"))


#反归一化
def normalizationData(scaler, trainY, trainPredict, testY, testPredict):
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform(trainY)
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform(testY)
    return trainY, trainPredict, testY, testPredict


def plt2Data(trainY, trainPredict, testY, testPredict):
    plt.figure()

    plt.subplot(211)
    plt.plot(trainY)
    plt.plot(trainPredict[1:])

    plt.subplot(212)
    plt.plot(testY)
    plt.plot(testPredict[1:])
    plt.show()


def pltData(dataY, dataPredict):
    plt.figure()
    plt.plot(dataY, color="blue")
    plt.plot(dataPredict[1:], color="red")
    plt.legend(('original', 'predict'), loc='upper right')
    plt.xticks(range(0, len(dataY) ,96))
    plt.show()


def dataTrainMain(csvData, saveName):
    scaler, trainX, testX, trainY, testY = reshapeData(csvData)
    LSTMfit(trainX, trainY, saveName)


def dataPredictMain(modelName, csvData):
    model = load_model(os.path.join("DATA", modelName + ".h5"))
    dataset = readDataset(csvData)
    dataX, dataY = createTrainDataset(dataset, 96)
    dataPredict = model.predict(dataX)
    return dataPredict, dataY


if __name__ == '__main__':
    # csvData = 'airline-passengers.csv'
    csvData = r'F:\Work\科研项目\\2020.3\河北\江苏曲线\\cuv.csv'
    # saveName = r'LSTM_JS01_10_96'
    saveName = r'LSTM_JS01_96'
    # dataTrainMain(csvData, saveName)

    csvPredictData = r'F:\Work\科研项目\\2020.3\河北\江苏曲线\\03\cuv.csv'
    dataPredict, dataY = dataPredictMain(saveName, csvPredictData)
    pltData(dataY, dataPredict)