import kNN
import numpy as np
import matplotlib.pyplot as plt


group,labels=kNN.createDataSet()
returnMat,classLabelVector=kNN.file2matrix('datingTestSet2.txt')
# print(group)
# print(labels)
# print(returnMat)
# print(classLabelVector.astype(int))
fig=plt.figure()
ax=fig.add_subplot(121)
ax.scatter(returnMat[:,1],returnMat[:,2],15.0*classLabelVector.astype(int),
           15.0*classLabelVector.astype(int),facecolor='white')
plt.xlabel('The rate of time consumed by playing games (%)')
plt.ylabel('The consumed icecream weekly (liter)')
ax1=fig.add_subplot(122)
ax1.scatter(returnMat[:,0][classLabelVector==1.0],returnMat[:,1][classLabelVector==1.0],color='r',
           s=15.0*classLabelVector[classLabelVector==1.0].astype(int),label='DndLike')
ax1.scatter(returnMat[:,0][classLabelVector==2.0],returnMat[:,1][classLabelVector==2.0],color='b',
           s=15.0*classLabelVector[classLabelVector==2.0].astype(int),label='SmallDoses')
ax1.scatter(returnMat[:,0][classLabelVector==3.0],returnMat[:,1][classLabelVector==3.0],color='g',
           s=15.0*classLabelVector[classLabelVector==3.0].astype(int),label='LargeDoses')
ax1.legend(loc='best')

plt.ylabel('The rate of time consumed by playing games (%)')
plt.xlabel('The miles flied each year')

plt.show()
