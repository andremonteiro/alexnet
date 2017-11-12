from __future__ import print_function

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import os
from os import walk
from random import randint

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

conv1 = []
conv5 = []
fc7 = []

# Diretório onde se encontram as amostras TODO deixar como parâmetro
directory = "features";
# Aqruivo com as classes
fileClasses = "classes.txt";

y = []

# Faz leitura do arquivo de classes
f = open(fileClasses)
classes = f.readlines()

numClasses = [0] * 10177

print("Realizando leitura das classes...")
numeroLeituras = 0
amostras = os.listdir(directory)
for item in amostras:
    # Faz leitura da classe
	linha = classes[int(item)-1]
	classe = int(linha.split(' ')[1])
	numClasses[classe-1] = int(numClasses[classe-1]) + 1

	# Incremente número de leituras
	numeroLeituras = numeroLeituras + 1

print("Leitura das classes realizada!")

# Clases só são consideradas válidas se possuírem mais de 5 amostras (número de k folds)
classesValidas = []
i = 0
for item in numClasses:
    if (item > 10):
        classesValidas.append(i+1)
    i = i + 1

# Realiza leitura das amostras, só utilizando aquelas que pertencem a uma classe válida
numeroLeituras = 0
print("Realizando leitura das amostras...")
amostras = os.listdir(directory)
for item in amostras:
    print("Lendo amostra "+str((numeroLeituras+1))+". Faltam "+str(len(amostras)-numeroLeituras))

    # Faz leitura da classe
    linha = classes[int(item)-1]
    classe = int(linha.split(' ')[1])    

    # Realiza leitura das features apenas se for de uma classe válida
    if classe in classesValidas:
        y.append(classe)

        # Faz leitura das caraterísticas da camada conv1
        #arquivo = directory + "/" + item + "/conv1.txt";
        #with open(arquivo) as f:
        #   content = f.readlines()
        #   conv1.append([float(x.strip()) for x in content])

        # Faz leitura das caraterísticas da camada conv5
        #arquivo = directory + "/" + item + "/conv5.txt";
        #with open(arquivo) as f:
        #   content = f.readlines()
        #   conv5.append([float(x.strip()) for x in content])

        # Faz leitura das caraterísticas da camada fc7
        arquivo = directory + "/" + item + "/conv7.txt";
        with open(arquivo) as f:
           content = f.readlines()
           fc7.append([float(x.strip()) for x in content])

    numeroLeituras = numeroLeituras + 1

X = fc7

# Divide o dataset em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.7, random_state=0)

# Parâmetros para busca no grid-search
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]}]

# Realiza grid-search
print("\nRealiza Grid-Search para precisão")

clf = GridSearchCV(SVC(), tuned_parameters, cv=5, scoring='precision_macro', n_jobs=-1)
clf.fit(X, y)

# Imprime resultados do grid-search
print("\nMelhores parâmetros encontrados:")
print(clf.best_params_)

print("\nDetalhamento:\n")
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))

print("\nRealiza etapa de testes:")
scores = cross_val_score(clf, X, y, cv=5, n_jobs = -1)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


#y_true, y_pred = y_test, clf.predict(X_test)
#print(classification_report(y_true, y_pred))