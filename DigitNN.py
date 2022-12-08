import numpy as np
import matplotlib.pyplot as plt
import time
import random
#encoding a digit number feature which will have 9 values (0,1....9)
def onehotencoding(lines):
    for i in range(len(lines)):
        Array = []
        for j in range (10):
            Array.append(0)
        Array[int(lines[i])]=1
        lines[i] = np.array(Array)
    lines = np.array(lines)
    return lines


def get_label(Labels):
    file_lines = open(Labels).readlines()
    file_lines=  [int(line.strip()) for line in file_lines]
    for i in range(len(file_lines)):
        if file_lines[i]<=0:
            file_lines[i]=0
    return file_lines,len(file_lines)

def load_file_images(filename,l):
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
    return Images

def proccessingData(FileData,FileLabel):
    file_lines,lenLabel= get_label(FileLabel)
    image_data=load_file_images(FileData,lenLabel)
    file_lines = onehotencoding(file_lines)
    FlattenData=[]
    for i in range(len(image_data)):
        FlattenData.append(image_data[i].flatten())
    S_data=np.random.shuffle(np.arange(int(len(FlattenData))))
    return np.squeeze(np.array(FlattenData)[S_data]),np.squeeze(np.array(file_lines)[S_data])


# def probability(x_train,y_train,f):
#     train_len = x_train.shape[0]
#     length=y_train.shape[0]
#     Num_of_Labels=np.unique(y_train).shape[0]
#     Num_of_Feature=x_train.shape[1]
#     Features = f*f +1
#     count_fL=np.zeros((Num_of_Labels,Num_of_Feature,Features))
#     count_L=[0]*Num_of_Labels
#
#     for i in range(train_len):
#         label_name = int(y_train[i])
#         count_L[label_name]+=1
#         for j in range(Num_of_Feature):
#             f_val=int(x_train[i,j])
#             count_fL[label_name,j,f_val] = count_fL[label_name,j,f_val]+1
#     prob_fl=np.zeros_like(count_fL)
#     prior = np.zeros(Num_of_Labels)
#     for i in range(Num_of_Labels):
#         prob_fl[i,:,:]=count_fL[i,:,:]/count_L[i]
#         prior[i]=count_L[i]/length
#     return prob_fl,prior
def buildModel(x_train,y_train,iteration,r):
    w = np.zeros((x_train.shape[1],10))
    b=[]
    for j in range (10):
            b.append(0)
    for i in range(iteration):
        l = x_train.shape[0]
        Z = np.squeeze(1/(1+np.exp(-(np.dot(x_train,w)+b)))) #sigmoid
        dw = (1/l)*np.dot(x_train.T,(Z-y_train)).reshape(w.shape[0],10)
        w = w -r*dw
        db = (1/l)*np.sum(Z-y_train)
        b = b = r*db
        cost = (1/l)*np.sum(y_train*np.log(Z) + (1-y_train)*np.log(1-Z))
    return w,b


def Predicting(w,b,x):
    w = w.reshape(x.shape[1],10)
    predY = 1/(1+np.exp(-(np.dot(x,w)+b)))
    for i in range(predY.shape[0]):
        Array = [0,0,0,0,0,0,0,0,0,0]
        Array[np.argmax(predY[i])] = 1
        predY[i] = Array
    return predY



def Accuracy(pred_y,true_y):
    c=0
    difference = pred_y - true_y
    for i in range(pred_y.shape[0]):
        if(difference[i] == 0).all():
            c=c+1
            #print(pred_y[i], true_y[i])
    Accuracy=c/pred_y.shape[0]
    return Accuracy

def main():
    trainData="data/digitdata/trainingimages"
    DataLabels="data/digitdata/traininglabels"
    testData ="data/digitdata/testimages"
    TestLabels="data/digitdata/testlabels"
    x_train,y_train=proccessingData(trainData,DataLabels)
    x_test, y_test=proccessingData(testData,TestLabels)
    DataPercent=int(x_train.shape[0]/10)
    timeTaken=[]
    TestAccuracy=[]

    for i in range(10):
        s=time.time()
        w,b = buildModel(x_train[0:DataPercent*(i+1)],y_train[0:DataPercent*(i+1)],2000,0.6)
        pred_y = Predicting(w,b,x_test)
        end = time.time()
        timeTaken.append(end - s)
        TestAccuracy.append(Accuracy(pred_y,y_test))
        print("Training Data percent = ", (i + 1)*10, " Time taken = ", timeTaken[i], " Accuracy = ", TestAccuracy[i])
    x = np.arange(10, 101, 10)
    plt.plot(x, timeTaken, label='time', color="red")
    plt.xlabel('Percentage of Training Data')
    plt.title("Time for training and testing in NB for Digit data")
    plt.ylabel("Time taken")
    #plt.tight_layout()
    plt.show()
    x = np.arange(10, 101, 10)
    plt.plot(x, TestAccuracy, label='time', color="red")
    plt.xlabel('Percentage of Training Data')
    plt.title("Accuracy in NB for Digit data")
    plt.ylabel("Test Accuracy")
    plt.tight_layout()
    plt.show()
    # for i in range(10):
    #     print("Data percent = ", (i+1)*10, " Time taken = ",timeTaken[i]," Accuracy = ",TestAccuracy[i])

main()
