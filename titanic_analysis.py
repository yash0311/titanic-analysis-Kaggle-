from tools import return_train_test,targetFeatureSplit,featureFormat
from sklearn.feature_selection import SelectKBest
import matplotlib.pyplot
from sklearn import tree
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.ensemble import RandomForestClassifier
import random

titanic_data_dict=return_train_test('train')
titanic_test_dict=return_train_test('test')

'''
for key in titanic_data_dict:
    if ', Master.' in key or ', Miss.' in key:
        titanic_data_dict[key]['Age']=random.randint(0,20)

for key in titanic_test_dict:
    if ', Master.' in key or ', Miss.' in key:
        titanic_test_dict[key]['Age']=random.randint(0,20)
'''

###### OUTLIERS REMOVAL START #######

outliers=[]

for key in titanic_data_dict:
    age=float(titanic_data_dict[key]['Age'])
    if age>70:
        outliers.append((key,titanic_data_dict[key]['Age']))
    else:
        Age = float(titanic_data_dict[key]['Age'])
        Fare = float(titanic_data_dict[key]['Fare'])
        matplotlib.pyplot.scatter( Age, Fare )
    
    
matplotlib.pyplot.xlabel("Age")
matplotlib.pyplot.ylabel("Fare")
matplotlib.pyplot.show()

for x in outliers:
    print(x)
    titanic_data_dict.pop(x[0],0)

########  OUTLIERS REMOVAL END #######

#######  TITANIC NEW FEATURE CREATION START ##########

def create_feature1(SibSp,Parch):
    add=float(SibSp)+float(Parch)
    return str(add)
    
def create_feature2(sex,Pclass):
    s=''
    if sex=='male':
        s+= '1'
        if Pclass=='1':
            s+='1'
        elif Pclass=='2':
            s+='2'
        else:
            s+='3'
    else:
        s+= '0'
        if Pclass=='1':
            s+='4'
        elif Pclass=='2':
            s+='5'
        else:
            s+='6'
    return s

def create_feature3(name,sex,Pclass):
    if ', Mr.' in name and sex=='male' and Pclass=='3':
        return '1'
    elif ', Mr.' in name and sex=='male' and Pclass=='2':
        return '2'
    elif ', Mr.' in name and sex=='male' and Pclass=='1':
        return '3'
    elif ', Mrs.' in name and sex=='female' and Pclass=='3':
        return '4'
    elif ', Mrs.' in name and sex=='female' and Pclass=='2':
        return '5'
    elif ', Mrs.' in name and sex=='female' and Pclass=='1':
        return '6'
    elif ', Master.' and Pclass=='3':
        return '7'
    elif ', Master.' and Pclass=='2':
        return '8'
    elif ', Master.' and Pclass=='1':
        return '9'
    elif ', Miss.' and Pclass=='3':
        return '10'
    elif ', Miss.' and Pclass=='2':
        return '11'
    elif ', Miss.' and Pclass=='1':
        return '12'



for key in titanic_data_dict:
    titanic_data_dict[key]['add_SibSp_Parch']=create_feature1(titanic_data_dict[key]['SibSp'],titanic_data_dict[key]['Parch'])
    
for key in titanic_test_dict:
    titanic_test_dict[key]['add_SibSp_Parch']=create_feature1(titanic_test_dict[key]['SibSp'],titanic_test_dict[key]['Parch'])

for key in titanic_data_dict:
    titanic_data_dict[key]['Pclass_sex']=create_feature2(titanic_data_dict[key]['Sex'],titanic_data_dict[key]['Pclass'])

for key in titanic_test_dict:
    titanic_test_dict[key]['Pclass_sex']=create_feature2(titanic_test_dict[key]['Sex'],titanic_test_dict[key]['Pclass'])

for key in titanic_data_dict:
    titanic_data_dict[key]['Pclass_mfc']=create_feature3(key,titanic_data_dict[key]['Sex'],titanic_data_dict[key]['Pclass'])

for key in titanic_test_dict:
    titanic_test_dict[key]['Pclass_mfc']=create_feature3(key,titanic_test_dict[key]['Sex'],titanic_test_dict[key]['Pclass'])

#######  TITANIC NEW FEATURE CREATION END ###########

######  SELECTION OF BEST FEATURES FOR ANALYSIS #######

my_dataset = titanic_data_dict
features_list=['Survived','add_SibSp_Parch','Parch','SibSp','Embarked','Fare','Age','Sex','Pclass','Pclass_sex','Pclass_mfc']

data = featureFormat(my_dataset, features_list)
labels, features = targetFeatureSplit(data)

k=5
k_best = SelectKBest(k=k)
k_best.fit(features, labels)
scores = k_best.scores_
print(scores)

feature_list=['Survived','Pclass_sex','age','Pclass_mfc','Fare','Sex']


######## FEATURE SCALING START ###########
data = featureFormat(my_dataset, features_list)
labels, features = targetFeatureSplit(data)


for i in range(len(features_list)-1):
    tmp =[]
    k=0
    for x in features:
        tmp.append(float(x[i]))
    tmp = MinMaxScaler().fit_transform(tmp)
    for x in features:
        x[i]=tmp[k]
        k = k + 1

####### FEATURE SCALING END #######

####### TRAINING & TESTING USING CLASSIFIER ########
clf=RandomForestClassifier()

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)


clf.fit(features_train,labels_train)
pred=clf.predict(features_test)

print accuracy_score(labels_test,pred)
print(recall_score(labels_test, pred))
####### TRAINING & TESTING USING CLASSIFIER END ########


