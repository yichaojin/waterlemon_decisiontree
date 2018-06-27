# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 13:30:43 2018

@author: hasee
"""
from sklearn import datasets
iris= datasets.load_iris()
from sklearn.feature_extraction import DictVectorizer
dv = DictVectorizer()
import pandas
import numpy as np


#==============================================================================
# 读取数据
#==============================================================================
data = pandas.read_excel(r"E:/研究生/算法书籍/西瓜DT.xlsx",sep=",")   
#data = data.values [13,8]
#data(2,1)
shape = data.shape
dataSet = []   
for i in range(shape[0]):  
    dataSet.append([])  
    for j in range(1,9):  
        dataSet[i].append(data.values[i][j])
data_array = np.array(dataSet, dtype = str)
#labelsk = np.array(['色泽:乌黑','色泽:浅白','色泽:青绿','根蒂:硬挺','根蒂:稍蜷','根蒂:蜷缩','敲声:沉闷','敲声:浊响','敲声:清脆','纹理:模糊','纹理:清晰','纹理:稍糊','脐部:凹陷','脐部:平坦','jibu:t','触感:硬滑','touch:soft','denty','sugarlv'] ) 

#data_01=dv.fit_transform(data_array).toarray()
#classlabelxg=np.array(['No','Yes'])

#==============================================================================
# 字典外list
#==============================================================================
dictnowon=[]
for i in range(shape[0]):
    key = 'color','root','sound','patent','jibu','touch','denty','sugar'
    value = dataSet[i]
    dictnowon.append( dict(zip(key, value)))
    




#==============================================================================
# dictnowon转换01
#==============================================================================


begin=dv.fit_transform(dictnowon).toarray()
labelbegin=dv.get_feature_names()
target2=np.array([1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0])
#target2=np.array(['Yes','Yes','Yes','Yes','Yes','Yes','Yes','Yes','No','No','No','No','No','No','No','No','No'])
targetw3=np.array(['0','1'])


from sklearn.tree import DecisionTreeClassifier
dt= DecisionTreeClassifier()
dt.fit(begin,target2)

from io import StringIO
from sklearn import tree
import pydotplus
#from IPython.display import Image
str_buffer = StringIO()
tree.export_graphviz(dt,out_file=str_buffer,rounded=True,feature_names=labelbegin,class_names=targetw3,special_characters=True,filled=True)
graph=pydotplus.graph_from_dot_data(str_buffer.getvalue())
graph.write_pdf("x.pdf")

#graph = pydotplus.graph_from_dot_data(str_buffer.getvalue())  
#Image(graph.create_png())


