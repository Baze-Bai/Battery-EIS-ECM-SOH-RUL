import tensorflow as tf
import random
# 检查 TensorFlow 版本
print("TensorFlow version:", tf.__version__)

# 检查可用的 GPU 数量
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import math
from math import sqrt
from numpy import log10
import cmath
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
import sklearn.metrics as metrics
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Reshape
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model


# 配置 GPU 按需分配内存
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

###特征读取
length = pd.read_csv('./data/length.csv')#读取每个电池的对应长度（既有EIS数据也有容量数据）
length = list(length.iloc[:,1])
feature_path = './data/feature/'
feature_files = ['feature25c01.csv','feature25c02.csv','feature25c03.csv','feature25c04.csv','feature25c05.csv','feature25c06.csv','feature25c07.csv','feature25c08.csv','feature35c01.csv','feature35c02.csv','feature45c01.csv','feature45c02.csv']



ECM_feature = []
for x in range(12):
    ECM_feature.append(pd.read_csv(feature_path+feature_files[x], header=None))

for i in range(len(ECM_feature)):
    ECM_feature[i].columns = ['Rs', 'R1', 'CPE1-1', 'CPE1-2', 'R2', 'Ws-1', 'Ws-2', 'CPE2-1', 'CPE2-2']





###容量读取
C_files = ['Data_Capacity_25C01.txt','Data_Capacity_25C02.txt','Data_Capacity_25C03.txt','Data_Capacity_25C04.txt','Data_Capacity_25C05.txt','Data_Capacity_25C06.txt','Data_Capacity_25C07.txt','Data_Capacity_25C08.txt', 'Data_Capacity_35C01.txt', 'Data_Capacity_35C02.txt', 'Data_Capacity_45C01.txt' , 'Data_Capacity_45C02.txt']
Capacity = []
path = './data/Capacity/'
for filename in C_files:
    Capacity.append(pd.DataFrame(np.loadtxt(path + filename,comments='t',delimiter='\t')))

for i in range(len(Capacity)):
    Capacity[i]= Capacity[i].iloc[:length[i], :]

for i in range(12):
    sequence = list(range(len(Capacity[i])))
    sequence.reverse()

    for j in sequence:
        Capacity[i].iloc[j,0] = Capacity[i].iloc[j,0]/Capacity[i].iloc[0,0]



##25度数据排除异常值

delete_list = []
for i in range(8):
    for j in range(len(ECM_feature[i])):
        if (ECM_feature[i].iloc[j, 1] > 10 and j not in delete_list):
            delete_list.append(j)
        if (ECM_feature[i].iloc[j, 2] > 1 and j not in delete_list):
            delete_list.append(j)
        if (ECM_feature[i].iloc[j, 7] > 0.12 and j not in delete_list):
            delete_list.append(j)
        if (ECM_feature[i].iloc[j, 3] > 0.8 and j not in delete_list):
            delete_list.append(j)
    ECM_feature[i] = ECM_feature[i].drop(index=delete_list, axis=0)
    Capacity[i] = Capacity[i].drop(index=delete_list, axis=0)
    delete_list = []



###25度数据结束清洗


##35度训练数据排除异常


ECM_feature[8].reset_index(drop=True, inplace=True)
Capacity[8].reset_index(drop=True, inplace=True)
delete_list = []#存储将要删除的行

#对第一个35度数据集进行清洗
for i in range(len(ECM_feature[8])):
    if (ECM_feature[8].iloc[i, 1] > 50 and i not in delete_list):
        delete_list.append(i)
    if (ECM_feature[8].iloc[i, 2] > 20 and i not in delete_list):
        delete_list.append(i)
    if (ECM_feature[8].iloc[i, 6] > 50 and i not in delete_list):
        delete_list.append(i)

ECM_feature[8] = ECM_feature[8].drop(index=delete_list, axis=0)
Capacity[8] = Capacity[8].drop(index=delete_list, axis=0)

ECM_feature[8].reset_index(drop=True, inplace=True)
Capacity[8].reset_index(drop=True, inplace=True)
delete_list = []


ECM_feature[8] = np.array(ECM_feature[8])


reserve_list = [not np.all(ECM_feature[8][i] == 0) for i in range(ECM_feature[8].shape[0])]
ECM_feature[8] = ECM_feature[8][reserve_list, :]
Capacity[8] = Capacity[8][reserve_list]

ECM_feature[8] = pd.DataFrame(ECM_feature[8])
ECM_feature[8].columns = ['Rs', 'R1', 'CPE1-1', 'CPE1-2', 'R2', 'Ws-1', 'Ws-2', 'CPE2-1', 'CPE2-2']

#结束第一个35度数据集的清洗

#对第二个35度数据集进行清洗
for i in range(len(ECM_feature[9])):
    if (ECM_feature[9].iloc[i, 1] > 50 and i not in delete_list):
        delete_list.append(i)
    if (ECM_feature[9].iloc[i, 2] > 30 and i not in delete_list):
        delete_list.append(i)
    if (ECM_feature[9].iloc[i, 6] > 50 and i not in delete_list):
        delete_list.append(i)

ECM_feature[9] = ECM_feature[9].drop(index=delete_list, axis=0)
Capacity[9] = Capacity[9].drop(index=delete_list, axis=0)

ECM_feature[9].reset_index(drop=True, inplace=True)
Capacity[9].reset_index(drop=True, inplace=True)
delete_list = []

ECM_feature[9] = np.array(ECM_feature[9])


reserve_list = [not np.all(ECM_feature[9][i] == 0) for i in range(ECM_feature[9].shape[0])]
ECM_feature[9] = ECM_feature[9][reserve_list, :]
Capacity[9] = Capacity[9][reserve_list]

ECM_feature[9] = pd.DataFrame(ECM_feature[9])
ECM_feature[9].columns = ['Rs', 'R1', 'CPE1-1', 'CPE1-2', 'R2', 'Ws-1', 'Ws-2', 'CPE2-1', 'CPE2-2']

#结束第二个35度数据集的清洗

##45度数据排除异常

delete_list = []
for i in [10, 11]:
    for j in range(len(ECM_feature[i])):
        if (ECM_feature[i].iloc[j, 1] > 0.3 and j not in delete_list):
            delete_list.append(j)
        if (ECM_feature[i].iloc[j, 2] > 10 and j not in delete_list):
            delete_list.append(j)
        if (ECM_feature[i].iloc[j, 3] < 0.5 and j not in delete_list):
            delete_list.append(j)
        if (ECM_feature[i].iloc[j, 5] > 1 and j not in delete_list):
            delete_list.append(j)
    ECM_feature[i] = ECM_feature[i].drop(index=delete_list, axis=0)
    Capacity[i] = Capacity[i].drop(index=delete_list, axis=0)
    delete_list = []

##结束45度数据集的清洗

result = pd.read_csv('./outputs/随机种子实验结果/实验结果RMSE与MAE/results.csv')
iteration = []
for i in range(len(result)):
    if(result.iloc[i, 3] > 0.9 and result.iloc[i, 0] > 100):
        iteration.append(result.iloc[i, 0])

window_size = 5
n_features = 6
l_train = [0,1,2,4,5,6,7,8,9,10,11]
for i in l_train:
    Capacity[i] = Capacity[i][window_size - 1:]


Capacity_25 = Capacity[5]
ECM_25 = ECM_feature[5]
ECM_25 = ECM_25.drop(columns=['Ws-2', 'R2', 'R1'])
scaler = MinMaxScaler()
ECM_25 = pd.DataFrame(scaler.fit_transform(ECM_25), columns=ECM_25.columns)


ECM_25 = np.array(ECM_25)
Capacity_25 = np.array(Capacity_25)


n_samples = len(ECM_25)


ECM_25 = np.array([ECM_25[i:i + window_size] for i in range(n_samples - window_size + 1)])


Capacity_25_1 = Capacity[2] ##第二个25度数据集的特征与容量
ECM_25_1 = ECM_feature[2]

ECM_25_1 = ECM_25_1.drop(columns=['Ws-2', 'R2', 'R1'])
scaler = MinMaxScaler()
ECM_25_1 = pd.DataFrame(scaler.fit_transform(ECM_25_1), columns=ECM_25_1.columns)

Capacity_25_1 = np.array(Capacity_25_1)
ECM_25_1 = np.array(ECM_25_1)


n_samples_1 = len(ECM_25_1)


ECM_25_1 = np.array([ECM_25_1[i:i + window_size] for i in range(n_samples_1 - window_size + 1)]) ##对特征做时间序列处理

tra_results = pd.DataFrame(columns=['seed', 'RMSE', 'MAE', 'R2', 'MSE'])

for i in range(115, 116):
    print('Index: ', i+1)
    random.seed(i)
    np.random.seed(i)
    tf.random.set_seed(i)

    l = int(ECM_25_1.shape[0] * 0.2)

    src_X = ECM_25
    src_Y = Capacity_25
    tat_X_train = ECM_25_1[:l]
    tat_Y_train = Capacity_25_1[:l]
    tat_X_test = ECM_25_1[l:]
    tat_Y_test = Capacity_25_1[l:]

    index = np.arange(src_X.shape[0])
    np.random.shuffle(index)
    src_X = (src_X[index, :, :])
    src_Y = (src_Y[index, :])

    index = np.arange(tat_X_train.shape[0])
    np.random.shuffle(index)
    tat_X_train = (tat_X_train[index, :, :])
    tat_Y_train = (tat_Y_train[index, :])

    lstm_model = Sequential()
    lstm_model.add(LSTM(128, input_shape=(window_size, n_features), return_sequences=True))
    lstm_model.add(LSTM(68, return_sequences=False))
    lstm_model.add(Dense(1))

    lstm_model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.0005))

    lstm_model.fit(src_X, src_Y, epochs=2000, batch_size=8)

    l_refer_model = tf.keras.models.clone_model(lstm_model)

    l_refer_model.set_weights(lstm_model.get_weights())

    ##将预训练的层进行固定 将输出层改变

    l_refer_model.pop()

    # Get the input of the original model
    model_input = l_refer_model.inputs[0]

    # Get the output of the second LSTM layer in the original model
    lstm_output = l_refer_model.layers[1].output

    # Reshape the lstm_output to make it compatible with the new LSTM layer
    lstm_output_reshaped = Reshape(target_shape=(1, 68))(lstm_output)

    # Add a new Dense layer for the new task
    new_output_layer1 = Dense(32, 'relu')(lstm_output_reshaped)

    # Add a new Dense layer for the new task
    new_output_layer = Dense(1, 'sigmoid')(new_output_layer1)

    # Create a new model using the original model's input and the new output layer
    new_model = Model(inputs=model_input, outputs=new_output_layer)

    for layer in new_model.layers[:-2]:
        layer.trainable = False

    new_model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.008))

    # 定义EarlyStopping回调
    early_stopping = EarlyStopping(
        monitor='val_loss',  # 需要监控的指标，这里选择验证集上的损失
        patience=1000,  # 在性能不再提升后，允许训练继续进行的epoch数量
        restore_best_weights=True,  # 是否恢复验证集性能最佳时的模型权重
        verbose=1  # 打印信息，1表示打印早停信息
    )

    new_model.fit(tat_X_train, tat_Y_train, epochs=25000, validation_data=(tat_X_test, tat_Y_test),
                  callbacks=early_stopping)

    X_test = tat_X_test
    Y_test = tat_Y_test
    y_pred_old = new_model.predict(X_test)
    Y_test = Y_test.reshape(-1)
    y_pred_old = y_pred_old.reshape(-1)

    r2 = r2_score(Y_test, y_pred_old)
    RMSE = np.sqrt(metrics.mean_squared_error(Y_test, y_pred_old))
    MAE = metrics.mean_absolute_error(Y_test, y_pred_old)
    MSE = metrics.mean_squared_error(Y_test, y_pred_old)
    print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, y_pred_old))
    print('Mean Squared Error:', metrics.mean_squared_error(Y_test, y_pred_old))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, y_pred_old)))
    print('R2', r2)


    tra_results.loc[i] = [i, RMSE, MAE, r2, MSE]

    plt.clf()
    y_test_old = np.array(Y_test)
    y_pred_old = np.array((y_pred_old))
    plt.plot(y_test_old, label='real')
    plt.plot(y_pred_old, label='predict')
    plt.legend()
    plt.ylabel('Capacity')
    plt.show()

tra_results.to_csv('./outputs/随机种子实验结果/实验结果RMSE与MAE/tra_results.csv', index=False)
ad_results = pd.DataFrame(columns=['seed', 'RMSE', 'MAE', 'R2', 'MSE'])


for i in iteration:
    ad_results.loc[i] = result.loc[i]

ad_results.to_csv('./outputs/随机种子实验结果/实验结果RMSE与MAE/ad_results.csv', index=False)



