import numpy as np
from matplotlib import pyplot as plt 
 
def createDataSet():  
    global  group,labels
    group = np.array([[1,0,1], [0,1,1], [1,1,0],[0,1,0]])  #设矩阵
    labels = [-1, -1, -1,1] #贴标签  

def perceptronClassify(trainGroup,trainLabels):
    global w, b

    numSamples = trainGroup.shape[0]
    mLenth = trainGroup.shape[1]

    a=[1 for x in range(numSamples)]      #增广矩阵
    trainGroup = np.column_stack((trainGroup, a))

    w=[1 for x in range(mLenth+1)]     #初始化W
    b=1                                #初始化B
    count=0                            #计数值，当count>向量个数numSamples，即完成训练
    cycle=100                            #循环次数，当超过循环次数，结束

    while(1):
        for i in range(numSamples):
            cycle-=1
            print(w,trainGroup[i])      #每运行一步就打印一次W与对应的X
            if cal(trainGroup[i],trainLabels[i]) <= 0:               
                count=0
                w=w-direction*b*trainGroup[i]        #W=W-BX
            else:
                count+=1               
        if count >=numSamples:           #判断是否取到适合的W
            print('Acomplish！')
            break
        elif cycle<=0:                   #失败
            print('Failed!')
            break


def cal(row,trainLabel):            #方向函数，用来判断X与W的方向
    global w, b,direction

    if np.matmul(row,w)==0:
        if trainLabel>0:
            direction=-1
        else:
            direction=1
        return 0
    elif np.matmul(row,w)>0:
        judge = 1
        if judge != trainLabel:
            direction=1
            return 0
    else:
        judge = -1
        if judge != trainLabel:
            direction=-1
            return 0
    return 1

createDataSet()
perceptronClassify(group,labels)