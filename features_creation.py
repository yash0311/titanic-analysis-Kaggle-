# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 23:31:55 2017

@author: yash
"""
import matplotlib.pyplot
import csv

def return_train_test(string):
    string = string + '.csv'
    
    train_open           =open(string)
    train_reader         =csv.reader(train_open) 
    list_train_data      =list(train_reader)
    
    
    length= len(list_train_data[0])
    
    
    titanic_data={}
    
    for i in range(1,len(list_train_data)):
        dict_train_data={}
        for j in range(length):
            if j==3 and string=='train.csv':
                continue
            elif j==2 and string=='test.csv':
                continue
            dict_train_data[list_train_data[0][j]]=list_train_data[i][j]
        if string=='train.csv':
            titanic_data[list_train_data[i][3]]=dict_train_data
        else:
            titanic_data[list_train_data[i][2]]=dict_train_data
    
    for key in titanic_data:
        if titanic_data[key]['Age']=='':
            titanic_data[key]['Age']=0.
    
    return titanic_data

titanic_data={}
titanic_data=return_train_test('test')

for key in titanic_data:
    if titanic_data[key]['Age']=='':
        print 'yes'

count=0






'''
for key in titanic_data:
    if titanic_data[key]['Survived']=='1':
        Age = float(titanic_data[key]['Age'])
        Fare = float(titanic_data[key]['Fare'])
        matplotlib.pyplot.scatter( Fare, Age )
    
matplotlib.pyplot.xlabel("Fare")
matplotlib.pyplot.ylabel("Age")
matplotlib.pyplot.show()
'''

'''
s_count=0
q_count=0
c_count=0
total_count=0
total_male=0
total_female=0
pclass_1=0
pclass_2=0
pclass_3=0
both_zero=0

for key in titanic_data:
    if titanic_data[key]['Survived']=='1':
        total_count+=1
        if titanic_data[key]['Parch']=='0' and titanic_data[key]['SibSp']=='0':
            both_zero+=1
        if titanic_data[key]['Pclass']=='1':
            pclass_1+=1
        elif titanic_data[key]['Pclass']=='2':
            pclass_2+=1
        else:
            pclass_3+=1
            
        if titanic_data[key]['Sex']=='female':
            total_female+=1
        else:
            total_male+=1
        
        if titanic_data[key]['Embarked']=='S':
            s_count+=1
        elif  titanic_data[key]['Embarked']=='Q':
            q_count+=1
        else:
            c_count+=1
            
print 'total count=',total_count
print "s=",s_count
print "q=",q_count
print "c=",c_count
print 'total male=',total_male
print 'total female=',total_female
print 'pclass_1=',pclass_1
print 'pclass_2=',pclass_2           
print 'pclass_3=',pclass_3           
print 'both zero',both_zero

'''






