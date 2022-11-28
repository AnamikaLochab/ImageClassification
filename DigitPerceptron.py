import numpy as np
import matplotlib as plt
import time

def sigmoid_func(s):
    return 1/(1+np.exp(-s))

def get_label(Labels):
    file_lines = open(Labels).readlines()
    file_lines=  [int(line.strip()) for line in file_lines]
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

# def buildModel(data,probabilityFL,prior):
#     NumofLabels=probabilityFL.shape[0]
#     length=data.shape[0]
#     NumOfFeatures=data.shape[1]
#     p=np.ones((NumofLabels,length))
#     pred_y=np.zeros(length)
#     for i in range(NumofLabels):
#         for j in range(length):
#             for k in range(NumOfFeatures):
#                 val = int(data[j,k])
#                 if(probabilityFL[i,k,val]<=0.01):
#                     probabilityFL[i,k,val]=0.01
#                 p[i,j]=p[i,j]*probabilityFL[i,k,val]
#             p[i,j] = p[i,j]*prior[i]
#     for i in range(length):
#         pred_y[i]=np.argmax(p[:,i])
#     return pred_y

def trainmodel(x_train,y_train, lr,iteration):
    x = np.random.rand(x_train.shape[1],10)
    print("Y_train",y_train.shape)
    for it in range(iteration):
        e=0
        for i in range(y_train.shape[0]):
            temp = np.squeeze(np.dot(x_train[i],x))
            P_temp=np.argmax(temp)
            if(P_temp!=y_train[i]):
                x[:,y_train[i]] = x[:,y_train[i]] + lr*x_train[i,y_train[i]]
                e = e+1
        if(e==0):
            break
    return x
def activationf(f):
    return np.where(f>0,1,0)

# def Model(x_train,y_train,lr,iteration):
#     length,n_features=x_train.shape
#     weights=np.zeros(n_features)
#     bias=0
#     y = np.where(y_train>0,1,0)
#     for i in range(iteration):
#         for index,x in enumerate(x_train):
#             f=np.dot(x,weights) + bias
#             pred_y=activationf(f)
#             update = lr*(y[index]-pred_y)
#             weights = weights+update*x
#             bias= bias+ update
#     return weights,bias

# def Prediction(weights,bias,x_test):
#     l=x_test.shape[0]
#     # temp = np.dot(x_test,x)
#     # y_pred = np.zeros(temp.shape[0])
#     # x= x.reshape(x_test.shape[1],10)
#     for i in range(l):
#         f = np.dot(x_test, weights) + bias
#         pred_y = activationf(f)
#     return pred_y

def Prediction(x,x_test):
    l=x_test.shape[0]
    temp = np.dot(x_test,x)
    y_pred = np.zeros(temp.shape[0])
    x= x.reshape(x_test.shape[1],10)
    for i in range(l):
        P_max = np.argmax(temp[i])
        y_pred[i]=P_max
    return y_pred

def Accuracy(pred_y,true_y):
    predY=np.squeeze(pred_y)
    l=predY.shape[0]
    c=0
    #print(l,true_y.shape[0])
    for i in range(l):
        if(predY[i] != true_y[i]):
            c=c+1
        print(predY[i],true_y[i])
    Accuracy=c/l
    print(c," ",l)
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
        #probabilityFL,prior=probability(x_train[0:DataPercent*(i+1)],y_train[0:DataPercent*(i+1)],1)
        x = trainmodel(x_train[0:DataPercent*(i+1)],y_train[0:DataPercent*(i+1)],0.09,100)
        end=time.time()
        timeTaken.append(end-s)
        pred_y = Prediction(x,x_test)
        TestAccuracy.append(Accuracy(pred_y,y_test))
        print("Training Data percent = ", (i + 1)*10, " Time taken = ", timeTaken[i], " Accuracy = ", TestAccuracy[i])
    # for i in range(10):
    #     print("Data percent = ", (i+1)*10, " Time taken = ",timeTaken[i]," Accuracy = ",TestAccuracy[i])

main()
