import numpy as np
import matplotlib.pyplot as plt
import time

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


def buildModel(x_train,y_train,iteration,r):
    w = np.zeros((x_train.shape[1],1))
    b=0
    for i in range(iteration):
        l = x_train.shape[0]
        Z = np.squeeze(1/(1+np.exp(-(np.dot(x_train,w)+b)))) #sigmoid
        dw = (1/l)*np.dot(x_train.T,(Z-y_train)).reshape(w.shape[0],1)
        w = w -r*dw
        db = (1/l)*np.sum(Z-y_train)
        b = b = r*db
    return w,b

def Predicting(w,b,x):
    w = w.reshape(x.shape[1],1)
    predY = 1/(1+np.exp(-(np.dot(x,w)+b)))
    for i in range(predY.shape[0]):
        if(predY[i]> 0.5):
            predY[i] =1
        else:
            predY[i] =0
    return predY

def Accuracy(pred_y,true_y):
    l=pred_y.shape[0]
    c=0
    for i in range(l):
        if(pred_y[i] == true_y[i]):
            c=c+1
    Accuracy=c/l
    return Accuracy

def main():
    trainData="data/facedata/facedatatrain"
    DataLabels="data/facedata/facedatatrainlabels"
    testData ="data/facedata/facedatatest"
    TestLabels="data/facedata/facedatatestlabels"
    x_train,y_train=proccessingData(trainData,DataLabels,3)
    x_test, y_test=proccessingData(testData,TestLabels,3)
    DataPercent=int(x_train.shape[0]/10)
    timeTaken=[]
    TestAccuracy=[]

    for i in range(10):
        s=time.time()
        w, b = buildModel(x_train[0:DataPercent * (i + 1)], y_train[0:DataPercent * (i + 1)], 2000, 0.6)
        pred_y = Predicting(w, b, x_test)
        end = time.time()
        timeTaken.append(end - s)
        TestAccuracy.append(Accuracy(pred_y,y_test))
        print("Training Data percent = ", (i + 1)*10, " Time taken = ", timeTaken[i], " Accuracy = ", TestAccuracy[i])
    x = np.arange(10, 101, 10)
    plt.plot(x, timeTaken, label='time', color="red")
    plt.xlabel('Percentage of Training Data')
    plt.title("Time for training and testing in NB for Face data")
    plt.ylabel("Time Taken")
    plt.tight_layout()

    plt.show()
    x = np.arange(10, 101, 10)
    plt.plot(x, TestAccuracy, label='time', color="red")
    plt.xlabel('Percentage of Training Data')
    plt.title("Accuracy in NB for Face data")
    plt.ylabel("Test Accuracy")
    plt.tight_layout()
    plt.show()
    # for i in range(10):
    #     print("Data percent = ", (i+1)*10, " Time taken = ",timeTaken[i]," Accuracy = ",TestAccuracy[i])

main()
