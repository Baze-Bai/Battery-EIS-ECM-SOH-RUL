import tensorflow as tf

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
import random


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


sample = ECM_feature[1]

#Capacity[4] = Capacity[4].iloc[:-9,:]
#ECM_feature[4] = ECM_feature[4].iloc[:-9, :]

#画特征相关性图
id = 1

fig, axes = plt.subplots(3, 3, figsize=(16, 10))
for i in range(3):
    for j in range(3):
        ax = axes[i, j]
        no = i*2+j
        ax.plot(ECM_feature[id].iloc[:, no], Capacity[id].iloc[:, 0], 'ro')
        ax.set_xlabel('feature')
        ax.set_ylabel('Capacity')
        ax.set_title(ECM_feature[id].columns[no], fontsize=15)
# 调整子图之间的间距
plt.subplots_adjust(wspace=0.25, hspace=0.5)  # 调整宽度和高度的间距
plt.show()


###画容量随时间衰减图
plt.clf()
id = 2
C_sample = Capacity[id]

plt.plot(range(C_sample.shape[0]), Capacity[id].iloc[:])
plt.title('NO.'+str((id+1))+' battery capacity')
plt.show()




# 计算25°数据集中特征与容量的线性相关性
correlations = ECM_feature[id].corrwith(Capacity[id].iloc[:, 0])
print(correlations)


# 利用源域数据训练lstm模型 并利用目标域数据进行对抗性训练（同时进行）
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

l_tat = int(n_samples_1*0.2)
tat_X = ECM_25_1[:l_tat]
tat_Y = Capacity_25_1[:l_tat]

X_test = ECM_25_1[l_tat:]
Y_test = Capacity_25_1[l_tat:]

src_X = ECM_25
src_Y = Capacity_25
###定义损失函数
@tf.function
def negative_binary_crossentropy(y_true, y_pred):
    # 计算中间变量
    mean_log = tf.reduce_mean(tf.math.log(y_pred / 0.5))
    # 使用 tf.cond 来替代 if-else 语句
    a = tf.cond(mean_log >= 0,
                true_fn=lambda: mean_log,
                false_fn=lambda: -mean_log)
    return a


batch_size = 8

lstm_1 = LSTM(128, return_sequences=True)
lstm_2 = LSTM(64, return_sequences=False)

input1 = Input(shape=(window_size, n_features)) #设置输入层格式
x1 = lstm_1(input1)
x2 = lstm_2(x1)

output1 = Dense(1, 'sigmoid')(x2)
src_model = tf.keras.Model(inputs=input1, outputs=output1)

src_model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse')

input2 = Input(shape=(window_size, n_features))  #设置输入层格式
x1 = lstm_1(input2)
x2 = lstm_2(x1)

output2 = Dense(1, 'sigmoid')(x2)
adversarial_model = tf.keras.Model(inputs=input2, outputs=output2)

adversarial_model.layers[3].trainable = False

adversarial_model.compile(optimizer=Adam(learning_rate=0.0005), loss=negative_binary_crossentropy, metrics=['accuracy'])

l_src = src_X.shape[0]



epochs = 2000

index = np.arange(src_X.shape[0])
np.random.shuffle(index)
src_X = (src_X[index, :, :])
src_Y = (src_Y[index, :])

index = np.arange(tat_X.shape[0])
np.random.shuffle(index)
tat_X = (tat_X[index, :, :])
tat_Y = (tat_Y[index, :])



for epoch in range(epochs):
    print(epoch+1 ,'/' ,epochs)
    # 随机选择源域和目标域的数据
    source_indices = np.random.randint(0, l_src, size=batch_size)
    target_indices = np.random.randint(0, l_tat, size=batch_size)

    source_batch = src_X[source_indices, :, :]
    source_labels = src_Y[source_indices]
    target_batch = tat_X[target_indices, :, :]
    target_labels = tat_Y[target_indices]

    # 训练数值预测器
    src_model.train_on_batch(source_batch, source_labels)

    #输出当前数值预测器的评价指标
    src_prediction = src_model.predict(source_batch)

    print('src_model Mean Absolute Error:', metrics.mean_absolute_error(source_labels, src_prediction))


    # 训练对抗性模型（鼓励特征提取器生成相似的特征）
    combined_data = np.concatenate([source_batch, target_batch])
    adversarial_labels = np.concatenate([np.ones(batch_size), np.zeros(batch_size)])   # 在这个示例中，对抗性标签设为1
    adversarial_model.train_on_batch(combined_data, adversarial_labels)


    src_prediction = src_model.predict(source_batch)
    print('Adversarial src_model Mean Absolute Error:', metrics.mean_absolute_error(source_labels, src_prediction))

pre_E = adversarial_model.predict(np.concatenate([src_X, tat_X]))


y_pred = src_model.predict(X_test)
r2 = r2_score(Y_test, y_pred)
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, y_pred)))
print('R2', r2)


plt.clf()
y_test = np.array(Y_test)
y_pred = np.array((y_pred))
plt.plot(y_test, label='real')
plt.plot(y_pred, label='predict')
plt.legend()
plt.ylabel('Capacity')
plt.xlabel('Cycle')
plt.show()


refer_model = tf.keras.models.clone_model(src_model)

refer_model.set_weights(src_model.get_weights())


# Get the input of the original model
model_input = refer_model.inputs[0]

# Get the output of the second LSTM layer in the original model
lstm_output = refer_model.layers[2].output

# Reshape the lstm_output to make it compatible with the new LSTM layer
lstm_output_reshaped = Reshape(target_shape=(1, 64))(lstm_output)

# Add a new Dense layer for the new task
new_output_layer1 = Dense(32, 'relu')(lstm_output_reshaped)


new_output_layer = Dense(1, 'sigmoid')(new_output_layer1)

# Create a new model using the original model's input and the new output layer
final_model = Model(inputs=model_input, outputs=new_output_layer)


for layer in final_model.layers[:-2]:
    layer.trainable = False


# 定义EarlyStopping回调
early_stopping = EarlyStopping(
    monitor='val_loss',  # 需要监控的指标，这里选择验证集上的损失
    patience=3000,  # 在性能不再提升后，允许训练继续进行的epoch数量
    restore_best_weights=True,  # 是否恢复验证集性能最佳时的模型权重
    verbose=1  # 打印信息，1表示打印早停信息
)

final_model.compile(optimizer=Adam(learning_rate=0.008), loss='mse')

final_model.fit(tat_X, tat_Y, epochs=20000, validation_data=(X_test, Y_test), batch_size=8, callbacks=early_stopping)

#A = pd.read_csv('./outputs/Battery_transfer_data_NO.3.csv')
y_pred = final_model.predict(X_test)
y_pred = y_pred.reshape(y_pred.shape[0],1)
#y_pred = A.iloc[:, 1]
#Y_test = A.iloc[:, 0]
r2 = r2_score(Y_test, y_pred)
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, y_pred)))
print('R2', r2)


plt.clf()
y_test = np.array(Y_test)
y_pred = np.array((y_pred))
plt.plot(y_test, label='real')
plt.plot(y_pred, label='predict')
plt.legend()
plt.ylabel('Capacity')
plt.xlabel('Cycle')
plt.show()



Battery_transfer_data = np.concatenate((y_test, y_pred), axis=1)

Battery_transfer_data = pd.DataFrame(Battery_transfer_data)
Battery_transfer_data.columns = ['Test', 'Pred']

# 将 DataFrame 保存为 CSV 文件
Battery_transfer_data.to_csv('./outputs/Battery_transfer_data_NO.3.csv', index=False, header=True)


final_model.save('./data/different_battery_final_model_NO.3.keras')
final_model.save('./data/different_battery_src_model_NO.2_wipe_Ws-2_R2_R1.keras')


















###传统lstm迁移学习
###相同温度下不同电池数据集的迁移

###对第一个25数据集进行迁移学习

##将被迁移的电池数据（源数据集6号电池的数据集）的全部数据进行训练

window_size = 5
n_features = 6

l_train = [0,1,2,4,5,6,7,8,9,10,11]
for i in l_train:
    Capacity[i] = Capacity[i][window_size - 1:]



random.seed(115)
np.random.seed(115)
tf.random.set_seed(115)

Capacity_25 = Capacity[5]
ECM_25 = ECM_feature[5]
ECM_25 = ECM_25.drop(columns=['Ws-2', 'R2', 'R1'])
scaler = MinMaxScaler()
ECM_25 = pd.DataFrame(scaler.fit_transform(ECM_25), columns=ECM_25.columns)


ECM_25 = np.array(ECM_25)
Capacity_25 = np.array(Capacity_25)

n_samples = len(ECM_25)

ECM_25 = np.array([ECM_25[i:i + window_size] for i in range(n_samples - window_size + 1)])





##利用3号电池的数据集训练迁移层，采用20%的数据
ECM_25_1 = ECM_feature[2]
ECM_25_1 = ECM_25_1.drop(columns=['Ws-2', 'R2', 'R1'])

scaler = MinMaxScaler()
ECM_25_1 = pd.DataFrame(scaler.fit_transform(ECM_25_1), columns=ECM_25_1.columns)

ECM_25_1 = np.array(ECM_25_1)
n_samples = len(ECM_25_1)
ECM_25_1 = np.array([ECM_25_1[i:i + window_size] for i in range(n_samples - window_size + 1)])


Capacity_25_1 = Capacity[2]
Capacity_25_1 = np.array(Capacity_25_1)


l = int(ECM_25_1.shape[0]*0.2)

train_target = ECM_25_1[:l]
train_target_C = Capacity_25_1[:l]

test_target = ECM_25_1[l:]
test_target_C = Capacity_25_1[l:]

index = np.arange(ECM_25.shape[0])
np.random.shuffle(index)
ECM_25 = (ECM_25[index, :, :])
Capacity_25 = (Capacity_25[index, :])

index = np.arange(train_target.shape[0])
np.random.shuffle(index)
train_target = (train_target[index, :, :])
train_target_C = (train_target_C[index, :])



lstm_model = Sequential()
lstm_model.add(LSTM(128, input_shape=(window_size, n_features), return_sequences=True))
lstm_model.add(LSTM(68, return_sequences=False))
lstm_model.add(Dense(1))

lstm_model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.0005))

lstm_model.fit(ECM_25, Capacity_25, epochs=2000, batch_size=8)



l_refer_model =  tf.keras.models.clone_model(lstm_model)

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


new_model.fit(train_target, train_target_C, epochs=25000, validation_data=(test_target, test_target_C), callbacks=early_stopping)

X_test = test_target
Y_test = test_target_C
y_pred_old = new_model.predict(X_test)
Y_test = Y_test.reshape(-1)
y_pred_old = y_pred_old.reshape(-1)
r2 = r2_score(Y_test, y_pred_old)
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, y_pred_old))
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, y_pred_old))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, y_pred_old)))
print('R2', r2)

plt.clf()
y_test_old = np.array(Y_test)
y_pred_old = np.array((y_pred_old))
plt.plot(y_test_old, label='real')
plt.plot(y_pred_old, label='predict')
plt.legend()
plt.ylabel('Capacity')
plt.show()





Tra_Battery_transfer_data = np.column_stack((y_test_old, y_pred_old))

Tra_Battery_transfer_data = pd.DataFrame(Tra_Battery_transfer_data)
Tra_Battery_transfer_data.columns = ['Test', 'Pred']

# 将 DataFrame 保存为 CSV 文件
Tra_Battery_transfer_data.to_csv('./outputs/Tra_Battery_different_transfer_data.csv', index=False, header=True)























###第一个25度数据集单独训练
window_size = 10

l_train = [0,1,2,4,5,6,7,8,9,10,11]
for i in l_train:
    Capacity[i] = Capacity[i]

    Capacity[i] = Capacity[i][window_size - 1:]

np.random.seed(11)
tf.random.set_seed(2023)

Capacity_25_1 = Capacity[0] ##第一个25度数据集的特征与容量
ECM_25_1 = ECM_feature[0]

ECM_25_1 = ECM_25_1.drop(columns=['Ws-2'])

Capacity_25_1 = np.array(Capacity_25_1)
ECM_25_1 = np.array(ECM_25_1)


n_samples = len(ECM_25_1)
n_features = 8

ECM_25_1 = np.array([ECM_25_1[i:i + window_size] for i in range(n_samples - window_size + 1)]) ##对特征做时间序列处理

l = ECM_25_1.shape[0]
l_train = int(l*0.2)
X_train = ECM_25_1[:l_train] ##第一个25度数据集训练集划分
X_test = ECM_25_1[l_train:]


Y_train = Capacity_25_1[:l_train]   ##测试集划分
Y_test = Capacity_25_1[l_train:]

Y_test = np.array(Y_test)
Y_train = np.array(Y_train)



##打乱训练集
index = np.arange(X_train.shape[0])
np.random.shuffle(index)
X_train = (X_train[index, :, :])
Y_train = (Y_train[index, :])


lstm_model = Sequential()
lstm_model.add(LSTM(128, input_shape=(window_size, n_features), return_sequences=True))
lstm_model.add(LSTM(68, return_sequences=False))
lstm_model.add(Dense(1))
optimizer = Adam(learning_rate=0.0001)  ###lstm模型预训练层

lstm_model.compile(loss='mean_squared_error', optimizer=optimizer)


# 定义EarlyStopping回调
early_stopping = EarlyStopping(
    monitor='val_loss',  # 需要监控的指标，这里选择验证集上的损失
    patience=2000,  # 在性能不再提升后，允许训练继续进行的epoch数量
    restore_best_weights=True,  # 是否恢复验证集性能最佳时的模型权重
    verbose=1  # 打印信息，1表示打印早停信息
)


lstm_model.fit(X_train, Y_train, epochs=2500, batch_size=32,validation_data=(X_test,Y_test), callbacks=[early_stopping])


# 提取LSTM特征
lstm_train_features = lstm_model.predict(X_train)
lstm_test_features = lstm_model.predict(X_test)


y_pred = lstm_test_features

r2 = r2_score(Y_test, y_pred)
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, y_pred)))
print('R2', r2)

plt.clf()
y_test = np.array(Y_test)
y_pred = np.array((y_pred))
plt.plot(y_test, label='real')
plt.plot(y_pred, label='predict')
plt.legend()
plt.ylabel('Capacity')
plt.show()



In_Battery_transfer_data = np.concatenate((y_test, y_pred), axis=1)

In_Battery_transfer_data = pd.DataFrame(In_Battery_transfer_data)
In_Battery_transfer_data.columns = ['Test', 'Pred']

# 将 DataFrame 保存为 CSV 文件
In_Battery_transfer_data.to_csv('./outputs/In_Battery_transfer_data.csv', index=False, header=True)




# 预测效果对比图
tra = pd.read_csv('./outputs/Tra_Battery_transfer_data.csv')
new = pd.read_csv('./outputs/Battery_transfer_data.csv')
index = range(len(new))

# 创建一个4x1的子图布局
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 绘制第一个子图
axes[0].scatter(index, tra.iloc[:len(new), 0], color='blue', s=50, marker='o',label='Real')
axes[0].plot(index, tra.iloc[:len(new), 1], color='red', label='Prediction')
axes[0].set_title('Tranditional LSTM transfer learning', fontsize=18)
axes[0].set_xlabel('Cycle', fontsize=13)
axes[0].set_ylabel('RUL', fontsize=13)
r2 = r2_score(tra.iloc[:, 0], tra.iloc[:, 1])
axes[0].legend()
axes[0].annotate('R2 = {}'.format(round(r2, 4)), xy=(110, 0.69), xytext=(36, 0.55),
                    arrowprops=dict(arrowstyle='->', lw=2, color='black'),
                    fontsize=15, color='black')



axes[1].scatter(index, new.iloc[:, 0], color='blue', s=50, marker='o',label='Real')
axes[1].plot(index, new.iloc[:, 1], color='red', label='Prediction')
axes[1].set_title('Improved LSTM transfer learning', fontsize=18)
axes[1].set_xlabel('Cycle', fontsize=13)
axes[1].set_ylabel('RUL', fontsize=13)
axes[1].legend()
r2_new = r2_score(new.iloc[:, 0], new.iloc[:, 1])
axes[1].annotate('R2 = {}'.format(round(r2_new, 4)), xy=(110, 0.69), xytext=(36, 0.55),
                    arrowprops=dict(arrowstyle='->', lw=2, color='black'),
                    fontsize=15, color='black')


plt.tight_layout()
plt.show()



# 设置DPI以提高分辨率
dpi = 1200  # 设置DPI值

# 保存图形时指定DPI
plt.savefig("./outputs/25°_传统方法与改进后的对比.png", dpi=dpi)


r2 = r2_score(new.iloc[:, 0], new.iloc[:,1])
print('Mean Absolute Error:', metrics.mean_absolute_error(new.iloc[:, 0], new.iloc[:,1]))
print('Mean Squared Error:', metrics.mean_squared_error(new.iloc[:, 0], new.iloc[:,1]))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(new.iloc[:, 0], new.iloc[:,1])))
print('R2', r2)


r2 = r2_score(tra.iloc[:, 0], tra.iloc[:,1])
print('Mean Absolute Error:', metrics.mean_absolute_error(tra.iloc[:, 0], tra.iloc[:,1]))
print('Mean Squared Error:', metrics.mean_squared_error(tra.iloc[:, 0], tra.iloc[:,1]))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(tra.iloc[:, 0], tra.iloc[:,1])))
print('R2', r2)
