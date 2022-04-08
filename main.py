""":cvar
    veri kümemizi bölüyoruz ilk aşamada. modelin doğruluğunu kontrol etmek için
    karışılık matrisi ve çapraz doğrulama kullanılmıştır. Sonuçta %94.62 lik bir doğruluk
    elde etmiş olduk.
    işlenmiş verimize de aynı işlemleri uyguladık. sonuçta %91.68 doğruluk elde etmiş
    olduk.
    arada yaklaşık olarak %3 lük bir doğruluk kaybı yaşanmıştır.


"""


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"# load dataset into Pandas DataFrame
df = pd.read_csv(url, names=['sepal length','sepal width','petal length','petal width','target'])


features = ['sepal length', 'sepal width', 'petal length', 'petal width']# Separating out the features
x = df.loc[:, features].values# Separating out the target
y = df.loc[:,['target']].values# Standardizing the features
x = StandardScaler().fit_transform(x)



X_train, X_test, y_train, y_test = train_test_split(x, y.ravel(), test_size = 0.25, random_state = 0)

#Create the SVM model

classifier = SVC(kernel = 'linear', random_state = 0)
#Fit the model for the data

classifier.fit(X_train, y_train)

#Make the prediction
y_pred = classifier.predict(X_test)

accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("df Accuracy: {:.2f} %".format(accuracies.mean()*100))


#----------------------------------------------------------------------------------------------------------------------------------

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

finalDf = pd.concat([principalDf, df[['target']]], axis = 1)

featuresFinal = ['principal component 1', 'principal component 2']# Separating out the features
X = finalDf.loc[:, featuresFinal].values# Separating out the target
Y = finalDf.loc[:,['target']].values# Standardizing the features
x = StandardScaler().fit_transform(x)

X_train, X_test, y_train, y_test = train_test_split(X, Y.ravel(), test_size = 0.25, random_state = 0)

#Create the SVM model

classifier = SVC(kernel = 'linear', random_state = 0)
#Fit the model for the data

classifier.fit(X_train, y_train)

#Make the prediction
y_pred = classifier.predict(X_test)

accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Final df Accuracy: {:.2f} %".format(accuracies.mean()*100))
