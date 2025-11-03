import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import *

normal_data = pd.read_csv('normal.csv', encoding='utf-8')
leakage_data = pd.read_csv('leakage.csv', encoding='utf-8')

data = pd.concat([normal_data, leakage_data], ignore_index=True)

X = data[['value']]
y = data[['target']]


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)
# X_test = X_test + 0.05                        # Offset = 0.05, 0.1, 0.5, 1.0

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)  
X_test_scaled = scaler.transform(X_test)

from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
y_train_fixed = y_train['target'].values.reshape(-1, 1)
y_train_one_hot = encoder.fit_transform(y_train_fixed).toarray()

y_test_fixed = y_test['target'].values.reshape(-1, 1)
y_test_one_hot = encoder.fit_transform(y_test_fixed).toarray()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))    

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 

    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))

def cross_entropy_error(y_pred, y_100):
    delta = 1e-7    
    result = 0
    for i in range(batch_size):
        x = -np.sum(y_100[i:i+1] * np.log(y_pred[i:i+1] + delta))
        result += x
         
    return result / 100

class ThreeLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def forward(self, X_100):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
    
        A1 = np.dot(X_100, W1) + b1
        Z1 = sigmoid(A1)
        A2 = np.dot(Z1, W2) + b2
        y_pred = softmax(A2)
    
        return y_pred, Z1
    
    def backward(self, X_100, y_100, y_pred, Z1):
        grads = {}
    
        dy = (y_pred - y_100) / batch_size
        grads['W2'] = np.dot(Z1.T, dy)
        grads['b2'] = np.sum(dy, axis = 0)
    
        dz1 = np.dot(dy, self.params['W2'].T) * (Z1 * (1 - Z1))
        grads['W1'] = np.dot(X_100.T, dz1)
        grads['b1'] = np.sum(dz1, axis = 0)
    
        return grads
    
    def loss(self, X_100, y_100):
        y_pred, Z1 = self.forward(X_100)
        return cross_entropy_error(y_pred, y_100)
        
    def gradient_descent(self, grads, learning_rate):
        for key in self.params.keys():
            self.params[key] -= learning_rate * grads[key]
            
    def accuracy(self, Value, target):
        y_pred, Z1 = self.forward(Value)
        y_pred =  np.argmax(y_pred, axis=1)
        y_target = np.argmax(target, axis=1)
        
        class_0_0 = class_0_1 = class_0_2 = 0
        class_1_0 = class_1_1 = class_1_2 = 0
        class_2_0 = class_2_1 = class_2_2 = 0
        
        for i in range(len(y_target)):
            if y_target[i] == 0 and y_pred[i] == 0: 
                class_0_0 += 1
            elif y_target[i] == 0 and y_pred[i] == 1:
                class_0_1 += 1
            elif y_target[i] == 0 and y_pred[i] == 2:
                class_0_2 += 1
            elif y_target[i] == 1 and y_pred[i] == 0:
                class_1_0 += 1
            elif y_target[i] == 1 and y_pred[i] == 1:
                class_1_1 += 1
            elif y_target[i] == 1 and y_pred[i] == 2:
                class_1_2 += 1
            elif y_target[i] == 2 and y_pred[i] == 0:
                class_2_0 += 1
            elif y_target[i] == 2 and y_pred[i] == 1:
                class_2_1 += 1
            elif y_target[i] == 2 and y_pred[i] == 2:
                class_2_2 += 1
                
        confusion_matrix = np.array([[class_0_0, class_0_1, class_0_2],
                                     [class_1_0, class_1_1, class_1_2],
                                     [class_2_0, class_2_1, class_2_2]])


        accuracy = class_0_0 + class_1_1 + class_2_2

        return accuracy, confusion_matrix
    
    
input_size = 100
hidden_size = 64                # 은닉층 개수 조절
output_size = 3

MNN = ThreeLayerNet(input_size, hidden_size, output_size)

batch_size = 100
train_epochs = int(len(X_train) / batch_size)

learning_rate_graph = {}
total_accuracy_list = []
total_accuracy_cnt = []
class_accuracy_list = []

def train_function():
    learning_rate = []
    learning_rate.append(float(lr_12.get()))
    learning_rate.append(float(lr_22.get()))
    learning_rate.append(float(lr_32.get()))
    learning_rate.append(float(lr_42.get()))
    learning_rate.append(float(lr_52.get()))
    learning_rate.append(float(lr_62.get()))

    
    for j in learning_rate:
        lst = []
        Train_total_accuracy = 0
        Train_total_confusion_matrix = np.zeros((3,3), dtype = int)
    
        for i in range(1, train_epochs+1):  # 학습 부분
 
            X_batch = X_train_scaled[(i-1)*batch_size : i*batch_size]
            X_train_100 = X_batch + np.zeros((100, 100))
            y_train_pred, Z1 = MNN.forward(X_train_100)
            y_train_100 = y_train_one_hot[(i-1)*batch_size : i*batch_size]
        
            loss = cross_entropy_error(y_train_pred, y_train_100)
            grads = MNN.backward(X_train_100, y_train_100, y_train_pred, Z1)
            MNN.gradient_descent(grads, j)
        
            lst.append(loss)
    
        for i in range(1, train_epochs+1):  # 정확도 계산 부분
 
            X_batch = X_train_scaled[(i-1)*batch_size : i*batch_size]
            X_train_100 = X_batch + np.zeros((100, 100))
            y_train_100 = y_train_one_hot[(i-1)*batch_size : i*batch_size]
                
            Train_epoch_accuracy, Train_epoch_confusion_matrix = MNN.accuracy(X_train_100, y_train_100)
            Train_total_accuracy += Train_epoch_accuracy
            Train_total_confusion_matrix += Train_epoch_confusion_matrix
        
        learning_rate_graph[j] = lst
        total_accuracy_list.append(Train_total_accuracy / (train_epochs * batch_size))
        total_accuracy_cnt.append(Train_total_accuracy)
        class_accuracy_list.append(Train_total_confusion_matrix)
    
        MNN.params['W1'] = 0.01 * np.random.randn(input_size, hidden_size)
        MNN.params['b1'] = np.zeros(hidden_size)
        MNN.params['W2'] = 0.01 * np.random.randn(hidden_size, output_size)
        MNN.params['b2'] = np.zeros(output_size)
        
    get_graph()
    
def get_graph():
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    X_learning_rate = range(1,211)
    
    for i, (key, value) in enumerate(learning_rate_graph.items()):
        row = i // 3  
        col = i % 3 
        axes[row, col].plot(X_learning_rate, value)
        axes[row, col].set_title("Learning rate : {}".format(key))
        axes[row, col].set_xlabel("Epochs")
        axes[row, col].set_ylabel("Loss")
    
        accuracy = total_accuracy_list[i]
        cnt = total_accuracy_cnt[i]
        axes[row, col].text(0.95, 0.95, "Accuracy : {:.4f}%\nCount : {}/{}".format(accuracy*100,cnt,len(X_train)), verticalalignment='top', horizontalalignment='right', 
                        fontsize=10, color='blue', bbox=dict(facecolor='white', alpha=0.7), transform=axes[row, col].transAxes)
    plt.show()

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    learning_rate_key = list(learning_rate_graph.keys())
    
    for i, ax in enumerate(axes.flat):  
        if i < len(class_accuracy_list):
            confusion_matrix = class_accuracy_list[i]
            accuracy = total_accuracy_list[i]
            cax = ax.matshow(confusion_matrix, cmap='Blues')
            fig.colorbar(cax, ax=ax, shrink=0.8)
            ax.set_title('Learning rate : {}\nAccuracy : {:.4f}%'.format(learning_rate_key[i], accuracy*100))
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')
            ax.xaxis.set_ticks_position('bottom')
            ax.set_xticks(np.arange(confusion_matrix.shape[1]))
            ax.set_yticks(np.arange(confusion_matrix.shape[0]))
            ax.set_xticklabels(np.arange(confusion_matrix.shape[1]))
            ax.set_yticklabels(np.arange(confusion_matrix.shape[0]))
        
            for j in range(confusion_matrix.shape[0]):
                for k in range(confusion_matrix.shape[1]):
                    value = confusion_matrix[j, k]
                    ax.text(k, j, str(value), ha='center', va='center', color='black', fontsize=10, bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
    plt.show()       
        



def test_function():
    select_learning_rate = float(lr_entry.get())
    test_epochs = int(len(X_test) / batch_size)
    Test_total_accuracy = 0
    Test_total_confusion_matrix = np.zeros((3,3), dtype = int)
    
    for i in range(1, train_epochs+1):
 
        X_batch = X_train_scaled[(i-1)*batch_size : i*batch_size]
        X_train_100 = X_batch + np.zeros((100, 100))
        y_train_pred, Z1 = MNN.forward(X_train_100)
        y_train_100 = y_train_one_hot[(i-1)*batch_size : i*batch_size]
        grads = MNN.backward(X_train_100, y_train_100, y_train_pred, Z1)
        MNN.gradient_descent(grads, select_learning_rate)

        
    for i in range(1, test_epochs+1):  # 정확도 계산 부분
 
        X_batch = X_test_scaled[(i-1)*batch_size : i*batch_size]
        X_test_100 = X_batch + np.zeros((100, 100))
        y_test_100 = y_test_one_hot[(i-1)*batch_size : i*batch_size]
        
        Test_epoch_accuracy, Test_epoch_confusion_matrix = MNN.accuracy(X_test_100, y_test_100)
        Test_total_accuracy += Test_epoch_accuracy
        Test_total_confusion_matrix += Test_epoch_confusion_matrix
        

    MNN.params['W1'] = 0.01 * np.random.randn(input_size, hidden_size)
    MNN.params['b1'] = np.zeros(hidden_size)
    MNN.params['W2'] = 0.01 * np.random.randn(hidden_size, output_size)
    MNN.params['b2'] = np.zeros(output_size)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    cax = ax.matshow(Test_total_confusion_matrix, cmap='Blues')
    fig.colorbar(cax, ax=ax, shrink=0.8)
    ax.set_title('Confusion Matrix_learnig rate : {}\nTotal Accuracy: {:.4f}%'.format(select_learning_rate, Test_total_accuracy*100/(test_epochs*batch_size)), fontsize=15)
    ax.text(2.95, 2.95, "Accuracy Count : {}/{}".format(Test_total_accuracy, test_epochs*batch_size), ha='right', va='top', color='white', fontsize=10, bbox=dict(facecolor='black', edgecolor='none'))

    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.xaxis.set_ticks_position('bottom')

    ax.set_xticks(np.arange(Test_total_confusion_matrix.shape[1]))
    ax.set_yticks(np.arange(Test_total_confusion_matrix.shape[0]))
    ax.set_xticklabels(np.arange(Test_total_confusion_matrix.shape[1]))
    ax.set_yticklabels(np.arange(Test_total_confusion_matrix.shape[0]))
    
    for j in range(Test_total_confusion_matrix.shape[0]):
        for k in range(Test_total_confusion_matrix.shape[1]):
            value = Test_total_confusion_matrix[j, k]
            ax.text(k, j, str(value), ha='center', va='center', color='black', fontsize=10, bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
    
    plt.show()
 

def clear_lr_window():
    Learing_rate_window.destroy()

def open_lr_window():
    global Learing_rate_window, lr_12, lr_22, lr_32, lr_42, lr_52, lr_62

    Learing_rate_window = Tk()
    Learing_rate_window.title('Select learning rate')

    title = Label(Learing_rate_window, text='Enter learning rate.', font='Arial 30 bold', fg='black')
    title.grid(row=0, column=0, columnspan=4, ipady=10)
    lr_11 = Label(Learing_rate_window, text='learning rate 1', font='Arial 15')
    lr_11.grid(row=1, column=0) 
    lr_12 = Entry(Learing_rate_window, width=20, bg='white')
    lr_12.grid(row=1, column=1, columnspan=2, sticky=W+E, ipady=5)
    lr_21 = Label(Learing_rate_window, text='learning rate 2', font='Arial 15')
    lr_21.grid(row=2, column=0) 
    lr_22 = Entry(Learing_rate_window, width=20, bg='white')
    lr_22.grid(row=2, column=1, columnspan=2, sticky=W+E, ipady=5)
    lr_31 = Label(Learing_rate_window, text='learning rate 3', font='Arial 15')
    lr_31.grid(row=3, column=0) 
    lr_32 = Entry(Learing_rate_window, width=20, bg='white')
    lr_32.grid(row=3, column=1, columnspan=2, sticky=W+E, ipady=5)
    lr_41 = Label(Learing_rate_window, text='learning rate 4', font='Arial 15')
    lr_41.grid(row=4, column=0) 
    lr_42 = Entry(Learing_rate_window, width=20, bg='white')
    lr_42.grid(row=4, column=1, columnspan=2, sticky=W+E, ipady=5)
    lr_51 = Label(Learing_rate_window, text='learning rate 5', font='Arial 15')
    lr_51.grid(row=5, column=0) 
    lr_52 = Entry(Learing_rate_window, width=20, bg='white')
    lr_52.grid(row=5, column=1, columnspan=2, sticky=W+E, ipady=5)
    lr_61 = Label(Learing_rate_window, text='learning rate 6', font='Arial 15')
    lr_61.grid(row=6, column=0) 
    lr_62 = Entry(Learing_rate_window, width=20, bg='white')
    lr_62.grid(row=6, column=1, columnspan=2, sticky=W+E, ipady=5)
    check = Button(Learing_rate_window, text='Check!', font='Arial 12 bold', bg='black', fg='white', command=train_function)
    check.grid(row=4, column=3, pady=5)
    close = Button(Learing_rate_window, text='Close', font='Arial 12 bold', bg='black', fg='white', command=clear_lr_window)
    close.grid(row=5, column=3, pady=5)

    Learing_rate_window.mainloop()

def open_test_window():
    global lr_entry
    
    testwindow = Tk()
    testwindow.title('Test window')
    title = Label(testwindow, text='Please enter the desired learning rate.', font='Arial 20', fg='black')
    title.grid(row=0, column=0, columnspan=4, ipady=10)
    Blank1 = Label(Mainwindow)
    Blank1.grid(row=1, column=0, columnspan=4, pady=10)
    lr_label = Label(testwindow, text='Learning rate', font='Arial 15')
    lr_label.grid(row=2, column=0) 
    lr_entry = Entry(testwindow, width=20, bg='white')
    lr_entry.grid(row=2, column=1, columnspan=2, sticky=W+E, ipady=5)
    Blank2 = Label(Mainwindow)
    Blank2.grid(row=3, column=0, columnspan=4, pady=10)
    test = Button(testwindow, text='Test', font='Arial 12 bold', bg='black', fg='white', command=test_function)
    test.grid(row=4, column=1)
    testwindow.mainloop()
    
    
Mainwindow = Tk()
Mainwindow.title('Motor Trouble shooting')

title = Label(Mainwindow, text='Motor trouble shooting', font='Arial 30 bold', fg='black')
title.grid(row=0, column=0, columnspan=4, ipady=10)
Blank1 = Label(Mainwindow)
Blank1.grid(row=1, column=0, columnspan=4, pady=10)
select_lr = Button(Mainwindow, text='Select learning rate', font='Arial 12 bold', bg='black', fg='white', command=open_lr_window)
select_lr.grid(row=4, column=0, columnspan=2, pady=5)
test_model = Button(Mainwindow, text='Test model', font='Arial 12 bold', bg='black', fg='white', command=open_test_window)
test_model.grid(row=4, column=2, columnspan=2, pady=5)
Mainwindow.mainloop()





