from __future__ import print_function

from sklearn.ensemble import RandomForestClassifier
import numpy as np
import statistics
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
import os
from os import walk
from random import randint

# Divide um array em N partes iguais. Usado para o cross validation
def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

# Retorna um classificador SVM linear treinado
def linear_svm(X, y):
    # Parâmetros para busca no grid-search
    tuned_parameters = [{'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
    #tuned_parameters = [{'kernel': ['linear'], 'C': [100]}]

    clf = GridSearchCV(SVC(), tuned_parameters, cv=5, scoring='precision_macro', n_jobs=-1)
    clf.fit(X, y)

    return clf

# Grava matriz de confusão no disco
def grava_matriz_confusao(conf_arr, file):
    norm_conf = []
    for i in conf_arr:
        a = 0
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            tmp_arr.append(float(j)/float(a))
        norm_conf.append(tmp_arr)

    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet, 
                    interpolation='nearest')

    width, height = conf_arr.shape

    for x in range(width):
        for y in range(height):
            ax.annotate(str(conf_arr[x][y]), xy=(y, x), 
                        horizontalalignment='center',
                        verticalalignment='center')

    cb = fig.colorbar(res)
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    plt.xticks(range(width), alphabet[:width])
    plt.yticks(range(height), alphabet[:height])

    # Grava resultados    
    plt.savefig(file, format='png')


# Realiza a classificação com features e classes passadas como parâmetro
# Testa um classificador SVM-RBF com parâmetros definidos via grid-search
def classify(X, y, dir, nome):
    # Divide a base em dois conjuntos: treinamento + validação e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.7, random_state=0)

    # Parâmetros para busca no grid-search
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                         'C': [1, 10, 100, 1000]}]
    '''tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3],
                         'C': [100]}]'''

    # Realiza grid-search
    print("\n   Realiza Grid-Search para precisão")

    clf = GridSearchCV(SVC(), tuned_parameters, cv=5, scoring='precision_macro', n_jobs=-1)
    clf.fit(X_train, y_train)

    print("\n   Realiza etapa de testes:")

    clf2 = SVC(kernel='rbf', gamma=clf.best_params_['gamma'], C=clf.best_params_['C'])

    scores = cross_val_score(clf2, X_test, y_test, cv=5)

    mean = statistics.mean(scores)
    std = statistics.stdev(scores)

    print("\n   Resultado")
    print("   %0.3f (+/-%0.03f)" % (mean, std))    

    y_true = y_test
    y_pred = clf.predict(X_test)

    # Formata matriz de confusão
    conf_arr = confusion_matrix(y_true, y_pred)

    # Grava resultados
    grava_matriz_confusao(conf_arr, dir+'confusion_matrix_'+nome+'.png')
    file = open(dir+"precision_"+nome+".txt", 'w')
    file.write("%s\n" % mean)
    file.write("%s\n" % std)
    file.close();

# Faz votação dos classificadores
def voting(r_conv1, r_conv5, r_fc2, test_y):
    # Faz voting
    resultados = []
    precisao = 0
    for vote in range(0, len(test_y)):
        # O que fazer em caso de empate? Considerando como certa a predição da FC2
        r = r_fc2[vote]
        v_conv1 = r_conv1[vote]
        v_conv5 = r_conv5[vote]
        v_fc2 = r_fc2[vote]

        if v_conv1 == v_conv5:
            r = v_conv1

        if r == test_y[vote]:
            precisao = precisao + 1

        resultados.append(r)

    return resultados, (precisao/len(test_y))

def realiza_voting_svm_linear(conv1, conv5, fc2, y):
    # Divide a base em dois conjuntos: treinamento + validação e teste
    ams = []
    for i in range(0, len(fc2)):
        ams.append(i)

    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        ams, y, test_size=0.7, random_state=0)

    # Garante que as três camadas usem as mesmas amostras como treinamento e teste
    X_train_conv1 = []
    X_train_conv5 = []
    X_train_fc2 = []
    y_train_conv1 = []
    y_train_conv5 = []
    y_train_fc2 = []
    for i in range(0, len(X_train_r)):
        indice = X_train_r[i]
        X_train_conv1.append(conv1[indice])
        X_train_conv5.append(conv5[indice])
        X_train_fc2.append(fc2[indice])
        y_train_conv1.append(y[indice])
        y_train_conv5.append(y[indice])
        y_train_fc2.append(y[indice])

    X_test_conv1 = []
    X_test_conv5 = []
    X_test_fc2 = []
    y_test_conv1 = []
    y_test_conv5 = []
    y_test_fc2 = []
    for i in range(0, len(X_test_r)):
        indice = X_test_r[i]
        X_test_conv1.append(conv1[indice])
        X_test_conv5.append(conv5[indice])
        X_test_fc2.append(fc2[indice])
        y_test_conv1.append(y[indice])
        y_test_conv5.append(y[indice])
        y_test_fc2.append(y[indice])

    # Treina classificadores lineares SVM para cada camada
    print("\n   Treinando SVM Linear para CONV1")
    linear_conv1 = linear_svm(X_train_conv1, y_train_conv1)
    print("\n   Treinando SVM Linear para CONV5")
    linear_conv2 = linear_svm(X_train_conv5, y_train_conv5)
    print("\n   Treinando SVM Linear para FC2")
    linear_fc2 = linear_svm(X_train_fc2, y_train_fc2)

    # Cria classificador de voting com cross validation
    # Divide base de testes em 5 partes para realizar cross-validation
    folds = chunkIt(X_test_r, 5)
    precisoes = []
    # Faz 5 iterações, para o cross validation 5-fold
    print("\n   Realizando voting cross-validation")
    for k in range(0, 5):
        print("\n      Iteração " + str(k))
        train_conv1_x = []
        train_conv5_x = []
        train_fc2_x = []
        test_conv1_x = []
        test_conv5_x = []
        test_fc2_x = []
        train_y = []
        test_y = []
        for i in range(0, len(folds)):        
            for j in range(0, len(folds[i])):
                # Pega parte de teste        
                if i == k:
                    elem = folds[i][j]
                    test_conv1_x.append(conv1[i])
                    test_conv5_x.append(conv5[i])
                    test_fc2_x.append(fc2[i])
                    test_y.append(y[i])                
                # Pega parte de treino
                else:
                    elem = folds[i][j]
                    train_conv1_x.append(conv1[i])
                    train_conv5_x.append(conv5[i])
                    train_fc2_x.append(fc2[i])
                    train_y.append(y[i])

        # Realiza treinamento dos SVMs
        linear_conv1.fit(train_conv1_x, train_y);
        linear_conv2.fit(train_conv5_x, train_y);
        linear_fc2.fit(train_fc2_x, train_y);

        # Realiza predição
        r_conv1 = linear_conv1.predict(test_conv1_x)
        r_conv5 = linear_conv2.predict(test_conv5_x)
        r_fc2   = linear_fc2.predict(test_fc2_x)

        resultados, precisao = voting(r_conv1, r_conv5, r_fc2, test_y)

        precisoes.append(precisao)

    # Faz teste no modelo
    r_conv1 = linear_conv1.predict(X_test_conv1);
    r_conv5 = linear_conv2.predict(X_test_conv5);
    r_fc2 = linear_fc2.predict(X_test_fc2);

    r, p = voting(r_conv1, r_conv5, r_fc2, y_test_conv1)

    mean = statistics.mean(precisoes)
    std = statistics.stdev(precisoes)

    print("\n   Resultado")
    print("   %0.3f (+/-%0.03f)" % (mean, std))    

    # Agrega resultados
    grava_matriz_confusao(confusion_matrix(y_test_conv1, r), 'resultados/a/confusion_matrix_vote.png')    
    file = open("resultados/a/precision_vote.txt", 'w')
    file.write("%s\n" % mean)
    file.write("%s\n" % std)
    file.close();

def realiza_classificacao_random_forest(fc2, y):
    X_train, X_test, y_train, y_test = train_test_split(
        fc2, y, test_size=0.7, random_state=0)

    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(X_train, y_train)

    # Faz cross-validation
    scores = cross_val_score(clf, X_test, y_test, cv=5)

    mean = statistics.mean(scores)
    std = statistics.stdev(scores)

    print("\n   Resultado")
    print("   %0.3f (+/-%0.03f)" % (mean, std))    

    y_true = y_test
    y_pred = clf.predict(X_test)

    # Formata matriz de confusão
    conf_arr = confusion_matrix(y_true, y_pred)

    # Grava resultados
    grava_matriz_confusao(conf_arr, 'resultados/a/confusion_matrix_random.png')
    file = open("resultados/a/precision_random.txt", 'w')
    file.write("%s\n" % mean)
    file.write("%s\n" % std)
    file.close();

# Início do programa    
conv1 = []
conv5 = []
fc2 = []

# Diretório onde se encontram as amostras TODO deixar como parâmetro
directory = "features";
# Aqruivo com as classes
fileClasses = "classes.txt";

y = []

# Faz leitura do arquivo de classes
f = open(fileClasses)
classes = f.readlines()

# Realiza leitura das amostras, só utilizando aquelas que pertencem a uma classe válida
print("Realizando leitura das amostras...")
numeroLeituras = 0;
amostras = os.listdir(directory)
for item in amostras:
    print("Lendo amostra "+str((numeroLeituras+1))+". Total de "+str(len(amostras)))

    # Faz leitura da classe
    linha = classes[int(item)-1]
    classe = int(linha.split(' ')[1])    

    if (classe <= 5):
        # Realiza leitura das features apenas se for de uma classe válida
        y.append(classe)

        # Faz leitura das caraterísticas da camada conv1
        arquivo = directory + "/" + item + "/conv1.txt";
        with open(arquivo) as f:
           content = f.readlines()
           conv1.append([float(x.strip()) for x in content])

        # Faz leitura das caraterísticas da camada conv5
        arquivo = directory + "/" + item + "/conv5.txt";
        with open(arquivo) as f:
           content = f.readlines()
           conv5.append([float(x.strip()) for x in content])

        # Faz leitura das caraterísticas da camada fc2
        arquivo = directory + "/" + item + "/conv7.txt";
        with open(arquivo) as f:
           content = f.readlines()
           fc2.append([float(x.strip()) for x in content])

    numeroLeituras = numeroLeituras + 1

print("\n*** Realiza experimentos A")

print("\nRealiza classificação com features da camada CONV1")
classify(conv1, y, "resultados/a/", "conv1")

print("\nRealiza classificação com features da camada CONV5")
classify(conv5, y, "resultados/a/", "conv5")

print("\nRealiza classificação com features da camada FC2")
classify(fc2, y, "resultados/a/", "fc2")

print("\n*** Realiza experimentos B")

# Concatena featreus da conv1, conv5 e fc2
print("\nRealiza classificação early fusion")
early = []
for i in range(0, len(fc2)):
    elem = []
    elem.extend(conv1[i])
    elem.extend(conv5[i])
    elem.extend(fc2[i])
    early.append(elem)
classify(early, y, "resultados/a/", "early")

print("\nRealiza classificação voting")

realiza_voting_svm_linear(conv1, conv5, fc2, y)

print("\n*** Realiza experimentos C")

print("\n   Realizando classificação Random Forest")

# Realiza classificação Random Forest
realiza_classificacao_random_forest(fc2, y)