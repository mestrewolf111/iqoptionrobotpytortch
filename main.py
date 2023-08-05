from iqoptionapi.stable_api import IQ_Option
import time
import csv
import datetime
import warnings
import datetime
warnings.filterwarnings('ignore')
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot
import matplotlib as plt
par = "EURGBP"
time_frame = 5
nrtentativas = 0
iq = IQ_Option(f"email", "senha")
iq.connect()  # connect to iqoption
def getdataiq(par, iq, time_frame):
    velas = iq.get_candles(par, 60, 1000, time.time())
    data = pd.DataFrame(velas)
    X = data[["open", "close", "min", "max", "volume"]]
    return X

data = getdataiq(par, iq, time_frame)

# Função que calcula a média móvel
def media_movel(data, periodo):
    mms = data.rolling(periodo).mean()
    return mms


def checktempo():
    while True:
        time.sleep(1)
        if  datetime.datetime.now().second == 2:
            break

def rsi_signal(data):
    ma_period = 14
    velas = data
    data['delta'] = data['close'] - data['open']
    data['up'] = np.where(data['delta'] > 0, data['delta'], 0)
    data['down'] = np.where(data['delta'] < 0, abs(data['delta']), 0)
    avg_up_period = data['up'].rolling(window=ma_period).mean()
    avg_down_period = data['down'].rolling(window=ma_period).mean()
    rs = avg_up_period / avg_down_period
    rsi = 100 - (100 / (1 + rs))
    return rsi

def candlestypes(data):
    velas = data
    X = data
    X['entradas'] = 0

    # Power Bear
    for i in range(len(X)):
        if X['close'][i] >= X['open'][i] and (X['close'][i] - X['open'][i]) >= (X['max'][i] - X['min'][i]):
            X['entradas'][i] = 1
        elif X['close'][i] < X['open'][i] and (X['open'][i] - X['close'][i]) >= (X['max'][i] - X['min'][i]):
            X['entradas'][i] = 2
        elif X['close'][i] >= X['open'][i] and (X['close'][i] - X['open'][i]) < (X['max'][i] - X['min'][i]):
            X['entradas'][i] = 3
        elif X['close'][i] < X['open'][i] and (X['open'][i] - X['close'][i]) < (X['max'][i] - X['min'][i]):
            X['entradas'][i] = 4
        else:
            X['entradas'][i] = 0
    return X['entradas']

def getcoresLstm(par,iq,time_frame):
    velas = iq.get_candles(par, time_frame, 1000, time.time())
    velas_df = pd.DataFrame(velas)
    velas_df['cores'] = np.where(velas_df['open'] < velas_df['close'], 2,
                             np.where(velas_df['open'] > velas_df['close'], 1,
                                      0))
    return velas_df['cores']



def adx(par, iq, time_frame, time, n=14):
    velas = iq.get_candles(par, time_frame, 1000, time.time())
    data = pd.DataFrame(velas)
    X = data[["open", "close", "min", "max", "volume"]]
    data['high_to_low'] = data['max'].sub(data['min'], axis=0)
    data['high_to_close'] = abs(data['max'].sub(data['close'].shift(1), axis=0))
    data['low_to_close'] = abs(data['min'].sub(data['close'].shift(1), axis=0))
    data['TR'] = data[['high_to_low', 'high_to_close', 'low_to_close']].max(axis=1)
    # Calculate the Average Directional Index (ADX)
    data['ADX'] = data['TR'] * 10000
    data['Signal'] = np.where(data['ADX'] < 1, 2, np.where(data['ADX'] > 1, 1, 0))
    return data[['Signal']]

def rsi_signal(data):
    ma_period = 100
    velas = data
    data['delta'] = data['close'] - data['open']
    data['up'] = np.where(data['delta'] > 0, data['delta'], 0)
    data['down'] = np.where(data['delta'] < 0, abs(data['delta']), 0)
    avg_up_period = data['up'].rolling(window=ma_period).mean()
    avg_down_period = data['down'].rolling(window=ma_period).mean()
    rs = avg_up_period / avg_down_period
    rsi = 100 - (100 / (1 + rs))
    rsi.dropna(inplace=True)
    rsi = np.array(rsi, dtype=np.float)
    signal = []
    for i in rsi:
        if i > 90:
            signal.append(1)
        elif i < 10:
            signal.append(2)
        else:
            signal.append(0)
    df = pd.DataFrame(signal)
    return df

def bollingerbands(par,iq, time_frame):
    ma_period = 20
    velas = iq.get_candles(par, time_frame, 1000, time.time())
    data = pd.DataFrame(velas)
    X = data[["open", "close", "min", "max", "volume"]]
    data['MA'] = data['close'].rolling(window=ma_period).mean()
    data['BB_up'] = data['MA'] + 2*data['close'].rolling(window=ma_period).std()
    data['BB_low'] = data['MA'] - 2*data['close'].rolling(window=ma_period).std()
    data['BB_range'] = data['BB_up'] - data['BB_low']
    data['BB_percent'] = (data['close'] - data['BB_low']) / data['BB_range']
    data['BB_percent'] = data['BB_percent'].apply(lambda x: 100 if x > 1 else (0 if x < 0 else x*100))
    bandas = []

    return data['BB_percent']


def stocastico(par, iq, time_frame):
    velas = iq.get_candles(par, time_frame, 1000, time.time())
    data = pd.DataFrame(velas)
    X = data[["open", "close", "min", "max", "volume"]]

    # Calculate Stochastic
    data['sto_k'] = 100 * (X['close'] - X['min']) / (X['max'] - X['min'])
    return data['sto_k']


def getcoresMorobozu(data):
    velas_df = data
    velas_df['cores'] = np.where(velas_df['open'] < velas_df['close'], 2,
                                 np.where(velas_df['open'] == velas_df['close'], 1,
                                 np.where(velas_df['open'] > velas_df['max'], 3,
                                 np.where(velas_df['open'] < velas_df['min'], 4, 0))))
    return velas_df['cores']



def trendlinaspercent(par, iq, time_frame):
    velas = iq.get_candles(par, time_frame, 1000, time.time())
    data = pd.DataFrame(velas)
    X = data[["open", "close", "min", "max", "volume"]]

    LTA = X.rolling(window=9).mean()
    LTB = X.rolling(window=21).mean()

    lta_pct_change = LTA.pct_change().fillna(0)
    ltb_pct_change = LTB.pct_change().fillna(0)

    return lta_pct_change, ltb_pct_change


def supresist(data):
    data['supresist'] = 0
    data.loc[(data['max'] > data['max'].shift(1)) & (data['max'] > data['max'].shift(-1)) & (
                data['max'].shift(-1) > data['max'].shift(-2)), 'supresist'] = 0
    data.loc[(data['min'] < data['min'].shift(1)) & (data['min'] < data['min'].shift(-1)) & (
                data['min'].shift(-1) < data['min'].shift(-2)), 'supresist'] = 0
    data.loc[(data['max'] > data['max'].shift(1)) & (data['max'] < data['max'].shift(-1)) & (
                data['max'].shift(-1) < data['max'].shift(-2)), 'supresist'] = 2
    data.loc[(data['min'] < data['min'].shift(1)) & (data['min'] > data['min'].shift(-1)) & (
                data['min'].shift(-1) > data['min'].shift(-2)), 'supresist'] = 1
    return data['supresist']
trade = True
reward = 0
while True:
    par = "EURGBP"
    time_frame = 60
    bet_money = 100
    X = getdataiq(par, iq, time_frame)
    mms = media_movel(X['close'], 21)
    # Criação de dataframe de saída
    output = pd.DataFrame(
        columns=['open', 'close', 'min', 'max','volume','LTA','LTB','bollinger','stocastico','supresist', 'mms', 'rsi', 'adx', 'cores', 'candles',
                 'reward'])
    output['candles'] = candlestypes(data)
    output['supresist'] = supresist(data)
    output['cores'] = getcoresLstm(par, iq, time_frame)
    output['reward'] = reward
    output['adx'] = adx(data, n=20)
    output['open'] = X['open']
    output['LTA'] = trendlinaspercent(par, iq, time_frame)[0]
    output['LTB'] = trendlinaspercent(par, iq, time_frame)[1]
    output['close'] = X['close']
    output['stocastico'] = stocastico(par, iq, time_frame)
    output['min'] = X['min']
    output['max'] = X['max']
    output['bollinger'] =  bollingerbands(par,iq, time_frame)
    output['mms'] = mms
    output['rsi'] = rsi_signal(data)

    # Criação da coluna de saída (labels)
    output['prediction'] = getcoresMorobozu(data)

    # Impressão dos resultados
    output.isna().sum()
    output = output.fillna(0)
    output.fillna(method="ffill", inplace=True)
    output = output.loc[~output.index.duplicated(keep='first')]
    features = output.drop('prediction', axis=1)
    labels = output['prediction']
    features_tensor = torch.from_numpy(features.values).float()
    labels_tensor = torch.from_numpy(labels.values).long()
    class PyTorchNet(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super(PyTorchNet, self).__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    # Definindo os parâmetros do modelo
    input_dim = len(features.columns)
    hidden_dim = 12
    output_dim = 3

    # Inicializando o modelo
    model = PyTorchNet(input_dim, hidden_dim, output_dim)

    # Definindo o otimizador
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Definindo a função de custo
    criterion = nn.CrossEntropyLoss()

    # Realizando o treinamento
    blablabla = iq.get_balance()
    num_epochs = 500
    for epoch in range(num_epochs):
        inputs = features_tensor
        labels = labels_tensor
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))
    # Realizando previsões

    torch.save(model, 'testemodelotc222.pth')
    model = torch.load('testemodelotc222.pth')
    predicts = model(inputs).detach().numpy()

    print(predicts[0])
    yhat = int(predicts[0][1])
    print(yhat)
    if yhat== 2:
        while datetime.datetime.now().second != 1:
            xd = "xd"
        trade = True
        saldin = iq.get_balance()
        check, id = iq.buy(bet_money, par, "call", 1)
        time.sleep(15)
        print("call")
        checktempo()
        betsies = iq.get_balance()
        vaisefude = betsies - blablabla
        betsies = iq.get_balance()
        vaisefude = betsies - blablabla
        if vaisefude > 0:
            reward += 1
            print("WIN")
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            inputs = features_tensor
            labels = labels_tensor
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            trade = False
            time.sleep(20)
            if (epoch + 1) % 10 == 0:
                print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))
        else:
            reward = 0
            print("LOSS")
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            inputs = features_tensor
            labels = labels_tensor
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            trade = False
            if (epoch + 1) % 10 == 0:
                print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))
    elif yhat== 1:
        saldin = iq.get_balance()
        while datetime.datetime.now().second != 1:
            xd = "xd"
        check, id = iq.buy(bet_money, par, "put", 1)
        time.sleep(15)
        print("put")
        trade = True
        checktempo()
        betsies = iq.get_balance()
        vaisefude = betsies - blablabla
        if vaisefude > 0:
            reward = 1
            print("WIN")
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            inputs = features_tensor
            labels = labels_tensor
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            trade = False
            time.sleep(20)
            if (epoch + 1) % 10 == 0:
                print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))
        else:
            reward = 1
            print("LOSS")
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            inputs = features_tensor
            labels = labels_tensor
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            trade = False
            if (epoch + 1) % 10 == 0:
                print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))
   # torch.save(model, 'testemodelotc222.pth')
    time.sleep(1)
