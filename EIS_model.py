import  os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import math
from math import sqrt
from kneed import KneeLocator
from numpy import  log10
from scipy import optimize
import scipy.signal as signal
import  cmath
from impedance import preprocessing
from impedance.models.circuits import CustomCircuit
from impedance.visualization import plot_nyquist
import impedance

path = './data/EIS data/EIS_state_I_25C02.txt'
cp35path = './data/capacity35C02.csv'
eis35path = './data/EIS_data_35C02.csv'

###实验数据整理
extract = pd.DataFrame(np.loadtxt(path, comments='t', delimiter='\t'))
extract = extract.iloc[:,2:5]
extract = list(extract.iloc[:60, 0])

capacity35 = pd.read_csv(cp35path)
EIS35 = pd.read_csv(eis35path)
EIS = pd.DataFrame(columns=['freq', 're', 'im'])
EIS = np.zeros((299*60,3))

for i in range(len(EIS35)):
    for j in range(120):
        if(j < 60):
            EIS[60*i + j, 1] = EIS35.iloc[i,j]
            EIS[60*i + j, 0] = extract[j]
        else:
            EIS[60*i + (j-60), 2] = EIS35.iloc[i,j]

EIS = pd.DataFrame(EIS,columns=['freq', 're', 'im'])
EIS_ = []
EIS_.append(EIS)
EIS = EIS_




##读取数据进EIS
files = os.listdir(path)
EIS = []
EIS_path = path + '/'
for filename in files:
    EIS.append(pd.DataFrame(np.loadtxt(EIS_path + filename,comments='t',delimiter='\t')))

for i in range(len(EIS)):
    EIS[i] = EIS[i].iloc[:,2:5]
    EIS[i].columns = ['freq','re','im']


##画EIS曲线








##画波特图
plt.clf()
for i in range(int(EIS_[5].shape[0]/60)):
    data_re = []
    data_re.append(EIS_[5].iloc[60*i,1])
    data_re.append(EIS_[5].iloc[60*i+1,1])
    data_re.append(EIS_[5].iloc[60 * i + 2, 1])
    data_im = []
    data_im.append(EIS_[5].iloc[60*i,2])
    data_im.append(EIS_[5].iloc[60*i+1,2])
    data_im.append(EIS_[5].iloc[60 * i + 2, 2])
    filter_data_re = []
    filter_data_re.append(sum(data_re)/3)
    filter_data_im = []
    filter_data_im.append(sum(data_im)/3)


    for j in range(3,60):
        data_re.pop(0)
        data_re.append(EIS_[5].iloc[60*i+j,1])
        filter_data_re.append(sum(data_re)/3)
        data_im.pop(0)
        data_im.append(EIS_[5].iloc[60*i+j,2])
        filter_data_im.append(sum(data_im)/3)
    n = int(math.sqrt(EIS[5].shape[0] / 60)) + 1
    plt.subplot(n, n, i + 1)
    x = log10(list(EIS_[5].iloc[60*i+2:60*i+60,0]))
    plt.plot(x, filter_data_re, 'k--')
    plt.xlabel('log(freq)')
    plt.ylabel('Z(re)')

plt.show()

###均值滤波改
plt.clf()
for i in range(int(EIS_[5].shape[0]/60)):
    data_re = []
    for n in range(6):
        data_re.append(EIS_[5].iloc[60 * i+n, 1])


    data_im = []
    for n in range(6):
        data_im.append(EIS_[5].iloc[60 * i + n, 1])

    filter_data_re = []
    filter_data_re.append(sum(data_re)/6)
    filter_data_im = []
    filter_data_im.append(sum(data_im)/6)


    for j in range(6,60):
        if(j<17):
            data_re.pop(0)
            data_re.append(EIS_[5].iloc[60*i+j,1])
            filter_data_re.append(sum(data_re)/6)
            data_im.pop(0)
            data_im.append(EIS_[5].iloc[60*i+j,2])
            filter_data_im.append(sum(data_im)/6)
        else:
            if len(data_re) == 5:
                data_re = data_re[2:]
                data_im = data_im[2:]
            data_re.pop(0)
            data_re.append(EIS_[5].iloc[60*i+j,1])
            filter_data_re.append(sum(data_re)/3)
            data_im.pop(0)
            data_im.append(EIS_[5].iloc[60*i+j,2])
            filter_data_im.append(sum(data_im)/3)

    n = int(math.sqrt(EIS[5].shape[0] / 60)) + 1
    plt.subplot(n, n, i + 1)
    x = log10(list(EIS_[5].iloc[60*i+5:60*i+60,0]))
    plt.plot(x, filter_data_im, 'k--')
    plt.xlabel('log(freq)')
    plt.ylabel('Z(re)')

plt.show()
q = EIS[5]


##均值滤波
for i in range(int(EIS[5].shape[0]/60)):
    data_re = []
    data_re.append(EIS[5].iloc[60*i,0])
    data_re.append(EIS[5].iloc[60*i+1,0])
    data_re.append(EIS[5].iloc[60 * i + 2, 0])
    data_im = []
    data_im.append(EIS[5].iloc[60*i,1])
    data_im.append(EIS[5].iloc[60*i+1,1])
    data_im.append(EIS[5].iloc[60 * i + 2, 1])
    filter_data_re = []
    filter_data_re.append(sum(data_re)/3)
    filter_data_im = []
    filter_data_im.append(sum(data_im)/3)


    for j in range(3,60):
        data_re.pop(0)
        data_re.append(EIS[5].iloc[60*i+j,0])
        filter_data_re.append(sum(data_re)/3)
        data_im.pop(0)
        data_im.append(EIS[5].iloc[60*i+j,1])
        filter_data_im.append(sum(data_im)/3)


    n = int(math.sqrt(EIS[5].shape[0] / 60)) + 1
    plt.subplot(n, n, i + 1)
    plt.plot(filter_data_re, filter_data_im, 'k--')
    x = filter_data_re
    y = filter_data_im
    kneedle_con_inc = KneeLocator(x,
                                      y,
                                      curve='concave',
                                      direction='increasing',
                                      online=True)##计算拐点
    plt.plot(kneedle_con_inc.knee, kneedle_con_inc.knee_y, 'ro')
plt.show()

data = []
data.append(0)
data.append(1)
print(data[0])

a = EIS[x]
im = a.iloc[0:59,2]
im = np.array(im)
print(im[signal.argrelextrema(-im, np.greater)])









##画图以及标出拐点(实际使用)
plt.clf()
x = files.index('EIS_state_V_25C02.txt')
fig = plt.figure()
fig.suptitle('EIS_state_V_25C02.txt')
n = int(math.sqrt(EIS[x].shape[0] / 60)) + 1

for i in range(int(EIS[x].shape[0]/60)):

    plt.subplot(n,n,i+1)
    plt.plot(EIS[x].iloc[60*i:60*(i+1),1] ,EIS[x].iloc[60*i:60*(i+1),2] ,'k--')

    x1 = EIS[x].iloc[60*i:60*(i+1),1]
    y = EIS[x].iloc[60*i:60*(i+1),2]

    kneedle_con_inc = KneeLocator(x1,
                                      y,
                                      curve='concave',
                                      direction='increasing',
                                      online=True)
    plt.plot(list(kneedle_con_inc.all_knees)[0], list(kneedle_con_inc.all_knees_y)[0], 'ro')

plt.show()

plt.clf()
plt.plot(EIS[x].iloc[60*298:60*(298+1),1],EIS[x].iloc[60*298:60*(298+1),2],'*')
plt.show()

plt.clf()
plt.plot(EIS[x].iloc[60 * 3:60 * (3 + 1), 1], EIS[x].iloc[60 * 3:60 * (3 + 1), 2], '*')
plt.show()
l = EIS[x].iloc[60 * 3:60 * (3 + 1), :]

plt.clf()
x = files.index('EIS_state_V_25C02.txt')
fig = plt.figure()
fig.suptitle('EIS_state_V_25C02.txt')
n = int(math.sqrt(EIS[x].shape[0] / 60)) + 1

for i in range(int(EIS[x].shape[0]/60)):

    plt.subplot(n,n,i+1)
    plt.plot(EIS[x].iloc[60*i:60*(i+1),1] ,EIS[x].iloc[60*i:60*(i+1),2] ,'k--')

    x1 = np.array(EIS[x].iloc[60*i:60*(i+1),1])
    y = np.array(EIS[x].iloc[60*i:60*(i+1),2])

    plt.plot(x1[signal.argrelextrema(y, np.greater)[0]], y[signal.argrelextrema(y, np.greater)], 'o')
    plt.plot(x1[signal.argrelextrema(-y, np.greater)[0]], y[signal.argrelextrema(-y, np.greater)], '+')


plt.show()
print(signal.argrelextrema(y, np.greater))


x = 0
###确定R欧姆

R_sum = []
for i in range(int(EIS[x].shape[0]/60)):
    print(i)
    if(EIS[x].iloc[60*i,2] > 0):
        R = EIS[x].iloc[60*i,2]
    else:
        for j in range(60):
            if(EIS[x].iloc[60*i+j,2] > 0):
                id_1 = 60*i+j-1
                id_2 = 60*i+j
                break
        print(id_1)
        print(((-EIS[x].iloc[id_1,2])/(EIS[x].iloc[id_2,2]-EIS[x].iloc[id_1,2]))*(EIS[x].iloc[id_2,1]-EIS[x].iloc[id_1,1]))
        R = EIS[x].iloc[id_1,1] + ((-EIS[x].iloc[id_1,2])/(EIS[x].iloc[id_2,2]-EIS[x].iloc[id_1,2]))*(EIS[x].iloc[id_2,1]-EIS[x].iloc[id_1,1])
    R_sum.append(R)

###确定极小值点与第一个圆的顶点
n = int(math.sqrt(EIS[x].shape[0] / 60)) + 1
fig = plt.figure()
fig.suptitle('EIS_state_V_35C02.txt')
mini = []
circlep = []
for i in range(int(EIS[x].shape[0]/60)):
    print(i)
    p = np.array(EIS[x].iloc[60*i:60*(i+1),2])

    print(signal.argrelextrema(-p,np.greater)[0])
    print(signal.argrelextrema(p,np.greater)[0])

    if bool(list(signal.argrelextrema(-p,np.greater)[0])):
        m = signal.argrelextrema(-p,np.greater)[0]
    if bool(list(signal.argrelextrema(p,np.greater)[0])):
        c = signal.argrelextrema(p,np.greater)[0]
    mini.append(m[-1])
    circlep.append(c[-1])
    plt.subplot(n, n, i + 1)
    plt.plot(EIS[x].iloc[60 * i:60 * (i + 1), 1], EIS[x].iloc[60 * i:60 * (i + 1), 2], 'k--')
    plt.plot(EIS[x].iloc[60 * i + m[-1], 1],EIS[x].iloc[60 * i + m[-1], 2],'ro')
    plt.plot(EIS[x].iloc[60 * i + c[-1], 1], EIS[x].iloc[60 * i + c[-1], 2], 'bo')

###确定圆的拐点
infp = []

fig = plt.figure()
fig.suptitle('EIS_state_V_35C02.txt')
n = int(math.sqrt(EIS[x].shape[0] / 60)) + 1

for i in range(int(EIS[x].shape[0]/60)):

    plt.subplot(n,n,i+1)
    plt.plot(EIS[x].iloc[60*i:60*(i+1),1] ,EIS[x].iloc[60*i:60*(i+1),2] ,'k--')

    x1 = list(EIS[x].iloc[60*i:60*(i+1),1])
    y = list(EIS[x].iloc[60*i:60*(i+1),2])

    kneedle_con_inc = KneeLocator(x1,
                                      y,
                                      curve='concave',
                                      direction='increasing',
                                      online=True)
    if(list(kneedle_con_inc.all_knees)[-1] >= EIS[x].iloc[60*i+(mini[i]-5),1]):
        idex = 0
    else:
        idex = -1
    plt.plot(list(kneedle_con_inc.all_knees)[idex], list(kneedle_con_inc.all_knees_y)[idex], 'ro')
    infp.append(x1.index(list(kneedle_con_inc.all_knees)[idex]))


###对圆进行拟合

help(curve_fit)
x = files.index('EIS_state_V_25C02.txt')
def circle(x,o,b,c):
    return np.sqrt(np.power(c,2)-np.power(x-o-c, 2)) + b
a_estimate = []
b_estimate = []
r_estimate = []
stan_dev = []

plt.clf()
for i in range(int(EIS[x].shape[0]/60)):
    for j in range(60):
        if(EIS[x].iloc[60 * i + j, 2] > 0):
            break

    end = mini[i]
    try:
        popt, pcov = curve_fit(circle, EIS[x].iloc[60 * i + j:60 * i+end, 1], EIS[x].iloc[60 * i + j:60 * i+end, 2], maxfev=10000)
    except:
        popt = [0,0,0]

    y_pred = [circle(i, popt[0], popt[1],popt[2]) for i in EIS[x].iloc[60 * i + j:60 * i+end, 1]]
    plt.subplot(n, n, i + 1)
    plt.plot(EIS[x].iloc[60 * i + j:60 * i+end, 1], EIS[x].iloc[60 * i + j:60 * i+end, 2],'*',label='origin')
    plt.plot(EIS[x].iloc[60 * i + j:60 * i+end, 1], y_pred,'r',label='fit')

    print(i)
    print(popt[0])
    print(popt[1])
    print(popt[2])
    a_estimate.append(popt[0])
    b_estimate.append(popt[1])
    r_estimate.append(popt[2])
    stan_dev.append(np.sqrt(np.diag(pcov)))

stan_dev = pd.DataFrame(stan_dev)
plt.show()

###对复数公式直接进行拟合
F = EIS[x]
F['complex'] = ''
x = files.index('EIS_state_V_25C02.txt')
R_set = []
Q_set = []
k_set = []
O_set = []
def cir(w,R,Q,k,O):
    return R/(1+R*Q*np.power(complex(0,w),k)) + O

for i in range(F.shape[0]):
    a = float(F.iloc[i,1])
    b = float(F.iloc[i,2])
    F.iloc[i,3] = complex(a, b)


for i in range(int(F.shape[0] / 60)):
    for j in range(60):
        if(F.iloc[60 * i + j, 2] > 0):
            break

    end = mini[i]


    popt, pcov = curve_fit(cir, F.iloc[60 * i + j:60 * i+end, 0], F.iloc[60 * i + j:60 * i+end, 3], maxfev=10000)


    y_pred = [cir(i, popt[0], popt[1],popt[2],popt[3]) for i in F.iloc[60 * i + j:60 * i+end, 1]]
    R_set.append(popt[0])
    Q_set.append(popt[1])
    k_set.append(popt[2])
    O_set.append(popt[3])



B = EIS[0]
fig = plt.figure()
fig.suptitle('EIS_state_V_35C02.txt')
n = int(math.sqrt(EIS.shape[0] / 60)) + 1
feature = np.zeros((299,9))
x = 0
outputpath = './data/'
for i in  range(int(EIS[x].shape[0] / 60)):

    A = EIS[x].iloc[60*i:60*(i+1) ,:]
    A= pd.DataFrame(A)
    A.iloc[:, 2] = A['im'].map(lambda x: -x)

    A.to_csv(outputpath+'simulate_data.csv',index=False,header=False)

    ###利用impedance包进行拟合
    frequencies, Z = preprocessing.readCSV('./data/simulate_data.csv')
    frequencies, Z = preprocessing.ignoreBelowX(frequencies, Z)

    ###拟合第一个半圆
    circuit1 = 'R0-p(R1,CPE1)'
    Rs=R_sum[-1]
    R1 = A.iloc[infp[-1],1]-R_sum[-1]
    w1 = A.iloc[infp[-1],0]
    C1 = 1/(R1*w1)
    initial_guess1 = [Rs, R1, C1, 0.5]
    circuit1 = CustomCircuit(circuit1, initial_guess=initial_guess1)
    start = 0
    for j in range(60):
        if(A.iloc[j, 2]>0):
            continue
        else:
            start = j
            break

    frequencies1 = frequencies[:infp[0]+1-start]
    Z1 = Z[:infp[0]+1-start]

    circuit1.fit(frequencies1, Z1)
    Z_fit1 = circuit1.predict(frequencies1)
    print(circuit1)


    parameters_1 =  circuit1.parameters_

    ###第二个半圆拟合
    circuit2 = 'R0-p(R2,CPE2)'
    Rs = A.iloc[infp[-1],1]
    R2 = A.iloc[mini[-1],1]-A.iloc[infp[-1],1]
    w2 = A.iloc[circlep[-1],0]
    C2 = 1/(R2*w2)
    initial_guess2 = [Rs, R2, C2, 0.5]
    circuit2 = CustomCircuit(circuit2, initial_guess=initial_guess2)

    frequencies2 = frequencies[infp[0]+1-start:mini[-1]+1-start]
    Z2 = Z[infp[0]+1-start:mini[-1]+1-start]

    circuit2.fit(frequencies2, Z2)
    Z_fit2 = circuit2.predict(frequencies2)
    print(circuit2)


    parameters_2 =  circuit2.parameters_

    ###warburg 阻抗拟合
    Rs = A.iloc[mini[-1], 1]
    circuit3 = 'R0-Ws1'
    initial_guess3 = [Rs, 1.5,40]
    circuit3 = CustomCircuit(circuit3, initial_guess=initial_guess3)

    frequencies3 = frequencies[mini[-1]+1-start:]
    Z3 = Z[mini[-1]+1-start:]

    circuit3.fit(frequencies3, Z3)
    Z_fit3 = circuit3.predict(frequencies3)
    print(circuit3)


    parameters_3 = circuit3.parameters_

    ###最终拟合
    circuit = 'R0-p(R1,CPE1)-p(R2-Ws1,CPE2)'
    Rs=R_sum[-1]

    initial_guess = [parameters_1[0], parameters_1[1], parameters_1[2], parameters_1[3], parameters_2[1], parameters_3[1], parameters_3[2], parameters_2[2],parameters_2[3]]
    circuit = CustomCircuit(circuit, initial_guess=initial_guess)
    circuit.fit(frequencies, Z)
    Z_fit = circuit.predict(frequencies)
    print(circuit)
    parameters = list(circuit.parameters_)
    for j in range(9):
        feature[i,j] = parameters[j]
    print(i)

print(type(Z[1]))
z1 = Z_fit
z2 = Z_fit




#等效电路模型
circuit = 'R0-p(R1,C1)-p(R2-Wo1,C2)'
#根据等效电路，从左到右对每个电路元件给出起始的拟合值
initial_guess = [.01, .01, 100, .01, .05, 100, 1]
#设计circuit为我们选用的电化学模型电路，并传入起始值
circuit = CustomCircuit(circuit, initial_guess=initial_guess)



###直接曲线拟合代码
def funcre(w,o,b,c,d):
    return o + (c+b*np.power(w,-1/2))/(np.power((d*b*np.power(w,1/2) + 1),2) + np.power(w,2)*np.power(d,2)*np.power((c+b*np.power(w,-1/2)),2))

def funcim(w,o,b,c,d):
    return (w*d*np.power((c+b*np.power(w,-1/2)),2) + b*np.power(w,-1/2)*(np.power(w,1/2)*d*b+1))/(np.power((d*b*np.power(w,1/2)),2) + np.power(w,2)*np.power(d,2)*np.power((c+b*np.power(w,-1/2)),2))

EIS_ = []

for filename in files:
    EIS_.append(pd.DataFrame(np.loadtxt(EIS_path + filename,comments='t',delimiter='\t')))

for i in range(len(EIS_)):
    EIS_[i] = EIS_[i].iloc[:,2:5]
    EIS_[i].columns = ['freq','re','im']

a = EIS_[0]

popt,pcov = curve_fit(funcre,a.iloc[:60,0],a.iloc[:60,1],maxfev = 10000)

y_pred = [funcre(i,popt[0],popt[1],popt[2],popt[3]) for i in a.iloc[:60,0]]

plt.clf()
plt.plot(a.iloc[:60,0],a.iloc[:60,2],'*',label='origin')
plt.plot(a.iloc[:60,0],y_pred,'r',label='fit')
plt.legend()
plt.show()

###文件导出

F.to_excel(outputpath+'simulate_data.xlsx',index=False,header=True)
mini = pd.DataFrame(mini)
mini.to_excel(outputpath + 'mini.xlsx',index=False,header=True)
circlep = pd.DataFrame(circlep)
circlep.to_excel(outputpath + 'circlep.xlsx',index=False,header=True)
R_sum = pd.DataFrame(R_sum)
R_sum.to_excel(outputpath + 'R_sum.xlsx',index=False,header=True)

help(impedance)

###xgboost模型初步代入
import xgboost as xgb
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(feature,capacity35,test_size=0.2,random_state=0)
model = xgb.XGBRegressor(max_depth=6,learning_rate=0.05,n_estimators=100,randam_state=42)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:',
      np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

y_test = np.array(y_test)
y_pred = np.array((y_pred))
plt.plot(y_test, label= 'real')
plt.plot(y_pred, label= 'prediction')
plt.legend()
plt.ylabel('Capacity')
plt.show()
plt.clf()

if (metrics.mean_absolute_error(Y_test, y_pred) < mae):
    mae = metrics.mean_absolute_error(Y_test, y_pred)
    op = i
###TCA
import numpy as np
import scipy.io
import scipy.linalg
import os
import sklearn.metrics
from sklearn.neighbors import KNeighborsClassifier

def kernel(ker, X1, X2, gamma):
    K = None
    if not ker or ker == 'primal':
        K = X1
    elif ker == 'linear':
        if X2 is not None:
            K = sklearn.metrics.pairwise.linear_kernel(
                np.asarray(X1).T, np.asarray(X2).T)
        else:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T)
    elif ker == 'rbf':
        if X2 is not None:
            K = sklearn.metrics.pairwise.rbf_kernel(
                np.asarray(X1).T, np.asarray(X2).T, gamma)
        else:
            K = sklearn.metrics.pairwise.rbf_kernel(
                np.asarray(X1).T, None, gamma)
    return K



class TCA:
    def __init__(self, kernel_type='primal', dim=30, lamb=1, gamma=1):
        '''
        Init func
        :param kernel_type: kernel, values: 'primal' | 'linear' | 'rbf'
        :param dim: dimension after transfer
        :param lamb: lambda value in equation
        :param gamma: kernel bandwidth for rbf kernel
        '''
        self.kernel_type = kernel_type
        self.dim = dim
        self.lamb = lamb
        self.gamma = gamma

    def fit(self, Xs, Xt):
        '''
        Transform Xs and Xt
        :param Xs: ns * n_feature, source feature
        :param Xt: nt * n_feature, target feature
        :return: Xs_new and Xt_new after TCA
        '''
        X = np.hstack((Xs.T, Xt.T))
        X /= np.linalg.norm(X, axis=0)
        m, n = X.shape
        ns, nt = len(Xs), len(Xt)
        e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))
        M = e * e.T
        M = M / np.linalg.norm(M, 'fro')
        H = np.eye(n) - 1 / n * np.ones((n, n))
        K = kernel(self.kernel_type, X, None, gamma=self.gamma)
        n_eye = m if self.kernel_type == 'primal' else n
        a, b = K @ M @ K.T + self.lamb * np.eye(n_eye), K @ H @ K.T
        w, V = scipy.linalg.eig(a, b)
        ind = np.argsort(w)
        A = V[:, ind[:self.dim]]
        Z = A.T @ K
        Z /= np.linalg.norm(Z, axis=0)
        Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T
        return Xs_new, Xt_new

###DBSCAN
from sklearn.cluster import DBSCAN # 引入DBSCAN模块
from sklearn import metrics # 调用评价指标

EIS = pd.DataFrame(np.loadtxt(path, comments='t', delimiter='\t'))
EIS = EIS.iloc[:,2:5]
EIS.columns = ['freq','re','im']



print(EIS.shape[0])

n = int(EIS.shape[0]/60)


EIS = np.array(EIS)
a = EIS[0]
b = a - EIS
d = b**2
c = (b**2).sum(axis=1)
g = c.sort()
c.sort(reversed=True)
t=np.sum(b,axis=1)
X = EIS

k = 9*2-1
def k_distance(X, k):
    X = np.array(X)
    k_dist = []
    for i in range(len(X)):
        dist = (((X[i] - X)**2).sum(axis=1)**0.5)
        dist.sort()
        k_dist.append(dist[k])

    return k_dist

k_dist = k_distance(feature,k)

x = np.arange(len(k_dist))
y = k_dist
kneedle_con_inc = KneeLocator(x,
                                      y,
                                      curve='concave',
                                      direction='increasing',
                                      online=True)



Capacity_release = []
for i in range(len(Capacity_)):
    C = []
    for j in range(len(Capacity_[i])):
        if(j == (len(Capacity_[i])-1)):
            break
        elif( Capacity_[i].iloc[j,2]==1 and Capacity_[i].iloc[j+1,2]==0 and Capacity_[i].iloc[j,1]  == Capacity_[i].iloc[j+1,1]):
            if(Capacity_[i].shape[1]==4):
                C.append(Capacity_[i].iloc[j,3])
            else:
                C.append(Capacity_[i].iloc[j, 5])
    Capacity_release.append(C)

Capacity = []
for i in range(len(Capacity_)):
    C = []
    for j in range(len(Capacity_[i])):
        if(j == (len(Capacity_[i])-1)):
            if(Capacity_[i].shape[1]==4):
                C.append(Capacity_[i].iloc[j,3])
            else:
                C.append(Capacity_[i].iloc[j, 5])
        elif( Capacity_[i].iloc[j,2]==0 and Capacity_[i].iloc[j+1,2]==1 and (Capacity_[i].iloc[j,1] + 1) == Capacity_[i].iloc[j+1,1]):
            if(Capacity_[i].shape[1]==4):
                C.append(Capacity_[i].iloc[j,3])
            else:
                C.append(Capacity_[i].iloc[j, 5])
    Capacity.append(C)














