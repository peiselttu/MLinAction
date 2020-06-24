import numpy as np
import operator
import os

def createDataSet():
    group=np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels=['A','A','B','B']
    return group,labels

def classify0(inX,dataSet,labels,k):
    datasetSize=dataSet.shape[0]    # get the number of samples
    X=np.tile(inX,(datasetSize,1))  # tile the sample and make the size equal to that of the  training dataset
    diffMat=X-dataSet
    diffMat=diffMat**2 #to square each element in the array
    sqDistance=diffMat.sum(axis=1)
    distances=np.sqrt(sqDistance)

    sortedDistanceIndices=distances.argsort() # sort the value of distances array in an ascending manner and return the corresponding index

    classCount={}
    for i in range(k):
        voteIlabel=labels[sortedDistanceIndices[i]] #get the ith element in the sortedDistances which indicate the row number of original dataset
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1

    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True) # return a list of key-value tuple

    return sortedClassCount[0][0] #return the first item in the first tuple

def file2matrix(filename):
    with open(filename) as f:
        fileData=[data.strip().split('\t') for data in f.readlines()]
        returnMat=np.array(fileData)[:,:3].astype(float)
        classLabelVector=np.array(fileData)[:,3].astype(float).flatten()
        return returnMat,classLabelVector

def autoNorm(dataset):
    minVal=dataset.min(0)
    maxVal=dataset.max(0)
    rangeVal=maxVal-minVal
    normVal=(dataset-minVal)/rangeVal
    return minVal,rangeVal,normVal

def datingClassTest():
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    hoRatio = 0.1
    dataNum = len(datingDataMat)
    trainingNum = dataNum - int(dataNum * 0.1)
    testingNum = int(dataNum * 0.1)

    indx = list(range(datingDataMat.shape[0]))
    np.random.shuffle(indx)
    datingDataMat = datingDataMat[indx]
    datingLabels = datingLabels[indx]
    minVal, maxVal, normVal = autoNorm(datingDataMat)

    X_train, y_train, X_test, y_test = normVal[:trainingNum], datingLabels[:trainingNum], + \
        normVal[trainingNum:], datingLabels[trainingNum:]
    correctNum = 0
    for indx, testData in enumerate(X_test):
        y_pred = classify0(testData, X_train, y_train, 3)
        y_true = y_test[indx]
        if y_pred == y_true:
            correctNum += 1

    acc = correctNum / testingNum
    return X_train, y_train, X_test, y_test, acc


def classifyPerson():
    resultList = ['not at all', 'in small dose', 'in large dose']
    percentTats = float(input('percentage of time spent playing video games?'))
    ffMiles = float(input('frequent flier miles earned per year'))
    iceCream = float(input('liters of ice cream consumed per year'))

    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    minVal, rangeVal, normVal = autoNorm(datingDataMat)
    intArr = np.array([[ffMiles, percentTats, iceCream]])
    y_pred = classify0(intArr - minVal / rangeVal, normVal, datingLabels, 3)

    print('You will probabily like this person:', resultList[int(y_pred) - 1])


# convert the image file into row vector
def img2vector(filename):
    with open(filename) as f:
        data=f.readlines()
        dataMat=np.zeros((1,1024))
        i=0
        for dr in data:
            for d in dr[:32]:
                dataMat[0,i]=int(d)
                i+=1
    return dataMat



def handWritingClassTest():
    currPath = os.path.abspath(os.path.dirname('__file__'))
    dataPath = os.path.join(currPath, 'digits')
    trainingData = os.path.join(dataPath, 'trainingDigits')
    testingData = os.path.join(dataPath, 'testDigits')
    trainFiles = os.listdir(trainingData)
    numTrain = len(trainFiles)
    trainingMat = np.zeros((numTrain, 1024))
    hwLabels = []
    for i in range(numTrain):
        train_data = img2vector(os.path.join(trainingData, trainFiles[i]))

        hwLabels.append(int(trainFiles[i].split('_')[0]))
        trainingMat[i, :] = train_data[0]

    testFiles = os.listdir(testingData)
    correctNum = 0
    numTest = len(testFiles)

    for t in range(numTest):
        test_data = img2vector(os.path.join(testingData, testFiles[t]))
        test_Label = int(testFiles[t].split('_')[0])
        y_pred = classify0(test_data, trainingMat, hwLabels, 3)
        print('the classifier came back with: %d, the real answer is: %d' % (y_pred, test_Label))
        if y_pred == test_Label:
            correctNum += 1
    print('the total testing samples that were predicted correctly is:', correctNum)
    print('the prediction accuracy is ', np.round(correctNum / numTest, 3))


if __name__=="__main__":

    group,labels=createDataSet()
    print(group)
    print(labels)

    inX=[0.2,0.3]

    pred_label=classify0(inX,group,labels,2)
    # print(pred_label)

    returnMat,classLabelVector=file2matrix('datingTestSet.txt')
    print(returnMat.shape)
    print(classLabelVector)

    minVal,maxVal,normVal=autoNorm(returnMat)
    print('minVal',minVal)
    print('maxVal',maxVal)
    print('normVal',normVal)