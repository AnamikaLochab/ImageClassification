import numpy as np
import matplotlib.pyplot as plt
import time
import random
from statistics import mean, stdev

def get_label(Labels):
    file_lines = open(Labels).readlines()
    file_lines=  [int(line.strip()) for line in file_lines]
    for i in range(len(file_lines)):
        if file_lines[i]<=0:
            file_lines[i]=0
    return file_lines,len(file_lines)

def load_file_images(filename,l,pool):
    file_lines=open(filename).readlines()
    file_len = int(len(file_lines))
    w= int(len(file_lines[0]))
    length= int(file_len/l)
    Images = []
    for i in range(l):
        Img = np.zeros((length,w))
        c=0
        for j in range (length*i,length*(i+1)):
            Line=file_lines[j]
            for k in range(len(Line)):
                if(Line[k]=="+" or Line[k]=="#"):
                    Img[c,k]=1
            c=c+1
        Images.append(Img)
    N_r=int(length/pool)
    N_c=int(w/pool)
    N_Images = np.zeros((l,N_r,N_c))
    for i in range(l):
        for j in range(N_r):
            for k in range(N_c):
                pix=0
                for r in range(pool*j,pool*(j+1)):
                    for c in range(pool*k,pool*(k+1)):
                        pix =pix + Images[i][r,c]
                N_Images[i,j,k]=pix
    return N_Images

def proccessingData(FileData,FileLabel,pool):
    file_lines,lenLabel= get_label(FileLabel)
    image_data=load_file_images(FileData,lenLabel,pool)
    FlattenData=[]
    for i in range(len(image_data)):
        FlattenData.append(image_data[i].flatten())
    S_data=np.random.shuffle(np.arange(int(len(FlattenData))))
    return np.squeeze(np.array(FlattenData)[S_data]),np.squeeze(np.array(file_lines)[S_data])

def probability(x_train,y_train,f):
    train_len = x_train.shape[0]
    length=y_train.shape[0]
    Num_of_Labels=np.unique(y_train).shape[0]
    Num_of_Feature=x_train.shape[1]
    Features = f*f +1
    count_fL=np.zeros((Num_of_Labels,Num_of_Feature,Features))
    count_L=[0]*Num_of_Labels

    for i in range(train_len):
        label_name = int(y_train[i])
        count_L[label_name]+=1
        for j in range(Num_of_Feature):
            f_val=int(x_train[i,j])
            count_fL[label_name,j,f_val] = count_fL[label_name,j,f_val]+1
    prob_fl=np.zeros_like(count_fL)
    prior = np.zeros(Num_of_Labels)
    for i in range(Num_of_Labels):
        prob_fl[i,:,:]=count_fL[i,:,:]/count_L[i]
        prior[i]=count_L[i]/length
    return prob_fl,prior
def buildModel(data,probabilityFL,prior):
    NumofLabels=probabilityFL.shape[0]
    length=data.shape[0]
    NumOfFeatures=data.shape[1]
    p=np.ones((NumofLabels,length))
    pred_y=np.zeros(length)
    for i in range(NumofLabels):
        for j in range(length):
            for k in range(NumOfFeatures):
                val = int(data[j,k])
                if(probabilityFL[i,k,val]<=0.01):
                    probabilityFL[i,k,val]=0.01
                p[i,j]=p[i,j]*probabilityFL[i,k,val]
            p[i,j] = p[i,j]*prior[i]
    for i in range(length):
        pred_y[i]=np.argmax(p[:,i])
    return pred_y

def Accuracy(pred_y,true_y):
    l=pred_y.shape[0]
    c=0
    for i in range(l):
        if(pred_y[i] == true_y[i]):
            c=c+1
    Accuracy=c/l
    return Accuracy

def main():
    trainData="data/face/facedatatrain"
    DataLabels="data/face/facedatatrainlabels"
    testData ="data/face/facedatatest"
    TestLabels="data/face/facedatatestlabels"
    x_train,y_train=proccessingData(trainData,DataLabels,3)
    train = []
    for i in range(x_train.shape[0]):
        train.append((x_train[i], y_train[i]))
    x_test, y_test=proccessingData(testData,TestLabels,3)
    DataPercent=int(x_train.shape[0]/10)
    MeanTimeTaken = []
    MeanTestAccuracy = []
    StdDeviation = []

    for i in range(10):
        timeTaken=[]
        TestAccuracy=[]
        for k in range(5):
            input = random.sample(train, DataPercent * (i + 1))
            x_data = []
            y_data = []
            for j in range(len(input)):
                x_data.append(input[j][0])
                y_data.append(input[j][1])
            s=time.time()
            probabilityFL,prior=probability(np.array(x_data),np.array(y_data),3)
            pred_y = buildModel(x_test,probabilityFL,prior)
            end = time.time()
            timeTaken.append(end - s)
            TestAccuracy.append(Accuracy(pred_y,y_test))
        MeanTimeTaken.append(mean(timeTaken))
        MeanTestAccuracy.append(mean(TestAccuracy))
        StdDeviation.append(stdev(TestAccuracy))
        print("Training Data percent = ", (i + 1)*10, " Average Time taken = ", mean(timeTaken), " Average Accuracy = ", mean(TestAccuracy), "Standard Deviation of Accuracy = ", stdev(TestAccuracy))
    x = np.arange(10, 101, 10)
    plt.plot(x, MeanTimeTaken, label='time', color="red")
    plt.xlabel('Percentage of Training Data')
    plt.title("Average Time for training and testing in NB for Face data")
    plt.ylabel("Average Time Taken")
    plt.tight_layout()

    plt.show()
    x = np.arange(10, 101, 10)
    plt.plot(x, MeanTestAccuracy, label='time', color="red")
    plt.xlabel('Percentage of Training Data')
    plt.title("Average Accuracy in NB for Face data")
    plt.ylabel("Average Test Accuracy")
    plt.tight_layout()
    plt.show()

    x = np.arange(10, 101, 10)
    plt.plot(x, StdDeviation, label='time', color="red")
    plt.xlabel('Percentage of Training Data')
    plt.title("Standard Deviation of Accuracy in NB for Face data")
    plt.ylabel("Standard Deviation")
    plt.tight_layout()
    plt.show()


main()
