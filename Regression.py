#!/usr/bin/env python
# coding: utf-8

# «Моделирование вкладов физических лиц»

# Выполнил Куулар Денис +7 926 572 54 90

# Задача: построить и описать модель, где целевой переменной (таргетом) являются помесячные изменения объемов рынка вкладов физических лиц РФ в рублях, которая будет объяснима с точки зрения экономической логики. Дополнительно требуется построить прогноз таргета на 01.2017 – 12.2017.

# Модель: оценивание можно проводить с помощью любых моделей (методов и алгоритмов), однако, они должны показывать, как хорошее качество с точки зрения описания взаимосвязей, так и высокую предсказательную силу на будущий период. Так же модель должна давать согласующиеся с экономической логикой зависимости.
# 

# In[1]:


#загрузим необходимые пакеты
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import IPython
import sklearn
import seaborn as sns 
import mglearn
from pandas.plotting import scatter_matrix
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PowerTransformer, StandardScaler, RobustScaler, PolynomialFeatures, MinMaxScaler
from sklearn.linear_model import LinearRegression, LogisticRegression, ElasticNet
from sklearn.tree import DecisionTreeRegressor


# ЗАГРУЗКА ФАЙЛА

# In[2]:


#загрузка файла
df=pd.read_excel('/Users/deniskuular/Desktop/Tests/BST Consulting/DataSet.xlsx')
df.head()


# In[3]:


# для удобства работы с набором данных произведем небольшие корректировки
df.rename (columns={'Unnamed: 0':'Date',
                    'Прирост вкладов физических лиц в рублях (млн руб)':'y',
                    'Доходность ОФЗ по сроку до 1 года':"X1",
                    'Ключевая ставка':"X2",
                    'Ставка по вкладам в долларах до 1 года':"X3",
                    'Ставка по вкладам в рублях до\xa01\xa0года':"X4",
                    'Нефть марки Юралс, долл./барр':"X5", 
                    'Индекс потребительских цен, ед.':"X6",
                    'М0, руб':"X7",
                    'М2, руб':"X8",
                    'Курс доллара к рублю, руб':"X9",
                    'Номинальная средняя з/п, руб':"X10"},
                      inplace=True)
#формируем массив независимых переменных (features) и  зависимую переменную (target)
features=df.iloc[0:72, 2:] 

target=pd.DataFrame(df.iloc[:72,1])

#отделим из набора массив для прогноза 
features_all=df.iloc[:, 2:] 

#отобразим полученные массивы
display(df), display(features), display(target), display(features_all)


# ПРЕДВАРИТЕЛЬНЫЙ АНАЛИЗ

# In[4]:


print("Размерность набора данных:")
print(df.shape)
print("Статистическая сводка")
features.describe()


# In[5]:


#Корреляционная матрица признаков
correlation=df.corr().abs()
correlation


# Из таблицы видно, что признаки (X1,X2,..., X10) имеют низкую корреляцию по отношению к цели (y), т.е. при изменении признака зависимая переменная практически не измениться.

# ВИЗУАЛИЗАЦИЯ ДАННЫХ

# In[6]:


# Диаграмма размаха
df.plot(kind='box', subplots=True,  sharex=False, sharey=False, figsize=(8,5))
plt.show()


# Из диаграммы размаха видно, по некоторым признакам присутствуют выборосы. Эти выбросы связи с монетарной политикой, решение об повышении ставка ЦБ РФ до 17% принято из-за возросших девальвационных и инфляционных рисков. Данные меры являлись следствием геополитических рисков (присоединение Крыма из-за которого запад начал вводить санкции). Также важно отметить, что на основе ключевой ставки определяются ставки по вкладам, кредитам, ценным бумагам.

# In[7]:


#Матрица диаграмм рассеяния
scatter_matrix(df, figsize=(13,10))
plt.show()


# ФОРМИРОВАНИЕ МОДЕЛЕЙ

# Наша задача построить и описать модель с высокой прогнозной силой и хорошим качеством с точки зрения взаимосвязей. Для выполнения поставленной задачи используем методы машинного обучения: LinearRegeression, LogisticRegression, DecisionTreeRegressor, RandomForestRegressor, KNeighborsRegressor. Данная задача относится к задачам регрессии (regression)

# In[8]:


#разделение обучающую и тестовую выборки
X_train, X_test, y_train, y_test=train_test_split(features, target, test_size=.2, random_state=1)

#загрузка алгоритмы моделей
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr', max_iter=10000)))
models.append(('LNR', LinearRegression()))
models.append(('KNN', KNeighborsRegressor(n_neighbors=4, n_jobs=-1)))
models.append(('CART', DecisionTreeRegressor(max_depth=100)))
models.append(('RF', RandomForestRegressor(n_estimators=100, max_depth=100, random_state=0)))
models.append(('Elastic', ElasticNet(alpha=0.001,random_state=0, max_iter=10000)))

#название модели 
names = []

for name, model in models:
    model_results = model.fit(X_train, np.ravel(y_train))
    names.append(name)
    print('%s: train %f test %f' % (name, model_results.score(X_train, y_train),
                                    model_results.score(X_test, y_test)))
        


# Как видно из результатов расчета, модели линейной регресии имеют более оптимистичные показатели.

# ПОДГОНКА МОДЕЛИ 

# In[9]:


#простой подбора параметров модели ElasticNet 
best_score=0

#обработка данных методом RobustScale
robust=RobustScaler().fit_transform(features)

#разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test=train_test_split(robust, target, test_size=.2, random_state=0)

for alpha in [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]:
        elastic=ElasticNet(alpha=alpha, random_state=0, max_iter=10000)
        elastic.fit(X_train, y_train)
        score=elastic.score(X_test, y_test)
        if score>best_score:
            best_score=score
            best_parameters={'alpha':alpha}
print("Наилучшее значение правильности: {:.4f}".format(best_score))
print("Наилучшее значение параметров: {}".format(best_parameters))


# Определили наилучшие гиперпараметры модели ElasticNet и в произведем корректировку предыдущих расчетов.

# ОБРАБОТКА ДАННЫХ 

# В виду того, что наши данные имеют разный масштаб, целесообразно, провести предварительную обработку данных. Для выполнения этой процедуры воспользуемся методами обработки данных модуля sklear.preprocessing.

# In[10]:


#методы обработки данных
preprocessing=[StandardScaler(),   
               RobustScaler(),
               MinMaxScaler(),
               PowerTransformer(method='box-cox')
              ]
#загрузка алгоритмы моделей
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr', max_iter=10000)))
models.append(('LNR', LinearRegression()))
models.append(('KNN', KNeighborsRegressor(n_neighbors=4, n_jobs=-1)))
models.append(('CART', DecisionTreeRegressor(max_depth=100)))
models.append(('RF', RandomForestRegressor(n_estimators=100, max_depth=100, random_state=0)))
models.append(('Elastic', ElasticNet(alpha=0.001,random_state=0, max_iter=10000)))

#название модели
names = []

for scaler in preprocessing:
    #подгтонка и изменение данных
    features_scaled=scaler.fit_transform(features)
    #разделение набора на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test=train_test_split(features_scaled, target, test_size=.2, random_state=0)
    print(scaler)
    for name, model in models:
        model_results = model.fit(X_train, np.ravel(y_train))
        names.append(name)
        print('%s: train %f test %f' % (name, model_results.score(X_train, y_train),
                                    model_results.score(X_test, y_test)))
    print("------------------------------------------")    
      


# Поскольку в наборе присутствуют выбросы, разумно, использовать метод RobustScale из-за устойчивости к выбросам и высоких показателей качества моделей. Также по результатам значений видно, что в моделях явно присутствуют моменты недообученности. 

# Мы уже знаем, что в наборе данных есть сильно коррелированные признаки. Разумно предположить, что исключение этих признаков положительно скажется на качество модели. И в целях доказательства этой гипотезы попробуем их найти и исключить из набора, далее посмотрим как поведут себя модели.

# In[11]:


#создание корреляционной матрицы признаков с абсолютным значением 
corr_matrix=features.corr().abs()

#выбрать верхний треугольник корреляционной матрицы
upper=corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))

#поиск индекса столбцов признаков с корреляцией больше 0.95
to_drop=[column for column in upper.columns if any(upper[column]>0.95)] #определил признак Х9

print("Сильно коррелированный признак: {}".format(to_drop))

#исключение признака с корреляцией больше 0.95 и создание нового массива признаков
features_drop=features.drop(features[to_drop], axis=1)

#обработка признаков методом RobustScale
robust=RobustScaler().fit_transform(features_drop)

#print(robust)

#разделение набора на обучающую и тестовую выборки
X_train, X_test, y_train, y_test=train_test_split(robust, target, test_size=.2, random_state=1)

#загрузка алгоритмов моделей
models=[('LNR', LinearRegression()),
        ('Elastic', ElasticNet(alpha=0.001,random_state=0, max_iter=10000))]

names = []

#обработчик результатов значений оценки качества выборок
for name, model in models:
    model_results = model.fit(X_train, np.ravel(y_train))
    #results.append(model_results)
    names.append(name)
    print('%s: train %f test %f' % (name, model_results.score(X_train, y_train),
                                    model_results.score(X_test, y_test)))


# Гипотеза о повышении качества модели об исключении сильно коррелированного признака отвергается, поскольку после удаления признака качество моделей снизилось.

# Поскольку наши модели (LNR и ElasticNet) недообучены, есть смысл добавить дополнительные признаки.

# In[12]:


#добавление признаков с исользованием метода Random

#имена столбцов
names=[f'X{i}' for i in range(11,13) ]

#генерируем рандомные числа в формате
X_generated=pd.DataFrame(np.random.random(size=(72,2))*100000, columns=names)

#объединяем признаки с генерированными данными
features_new=pd.concat([features, X_generated], axis=1)

#обработка данных
scaler=RobustScaler().fit_transform(features_new)

#разделение набора на обучающую и тестовую выборки
X_train, X_test, y_train, y_test=train_test_split(scaler, target, test_size=.2, random_state=0)

#загрузка алгоритмов моделей
models=[('LNR', LinearRegression()),
       ('Elastic', ElasticNet(alpha=0.001,random_state=0, max_iter=10000))]

names = []

#обработчик результатов значений оценки качества выборок
for name, model in models:
    model_results = model.fit(X_train, np.ravel(y_train))
    names.append(name)
    print('%s: train %f test %f' % (name, model_results.score(X_train, y_train),
                                    model_results.score(X_test, y_test)))


# Добавление рандомных значений в датасет дает нестабильный результат, однако отслеживается положительный характер повышения качества модели. Для обеспечания стабильности в значениях создадим полиномиальные элементы со degree=2. 

# In[13]:


#создание дополнительных элементов с примением PolynomialFeatures
#методом подбора были взяты 3 признака: X1, X4, X6
X_slice=pd.concat([features.iloc[:,0],features.iloc[:,5], features.iloc[:,3]], axis=1)

#создание полиномиальных элементов
poly = pd.DataFrame(PolynomialFeatures(degree=2, interaction_only=False).fit_transform(X_slice)).iloc[:,4:] #4:
#print(poly)

#добавление poly к features
X=pd.concat([features, poly], axis=1)

X_train, X_test, y_train, y_test=train_test_split(X, target, test_size=.2, random_state=0)

#загрузка алгоритмов моделей
models=[('LNR', LinearRegression()),
       ('Elastic', ElasticNet(alpha=0.001,random_state=0, max_iter=10000))]

names = []

#обработчик результатов значений оценки качества выборок
for name, model in models:
    model_results = model.fit(X_train, np.ravel(y_train))
    names.append(name)
    print('%s: train %f test %f' % (name, model_results.score(X_train, y_train),
                                    model_results.score(X_test, y_test)))


# Коэффициент детерминации метода наименьших квадратов выше, чем коэффициент детерминации модели ElasticNet. R2=0.812 относительно хороший показатель, попробуем составить прогноз с примением метода наименьших квадратов LinearRegression.

# In[14]:


#манупуляция с данными 
X=features_all
X_slice=pd.concat([X.iloc[:,0],X.iloc[:,5], X.iloc[:,3]], axis=1) 
poly = pd.DataFrame(PolynomialFeatures(degree=2, interaction_only=False).fit_transform(X_slice)).iloc[:,4:] #4:
X_new=pd.concat([X, poly], axis=1) #полиномиальные элементы

#разделение набора для составления прогноза
X_restricted=X_new.iloc[:72, :] #y известен
X_for_predict=X_new.iloc[72:,:] #y неизвестен (прогноз)

X_train, X_test, y_train, y_test=train_test_split(X_restricted, target, test_size=.2, random_state=0)
lr=LinearRegression().fit(X_train, y_train)

#определение параметров модели линейной регрессии
coefficients=pd.DataFrame(lr.coef_.T)
intercept=lr.intercept_

#прогнозирование

y_pred_restricted=lr.predict(X_restricted)
y_predict=lr.predict(X_for_predict)

print("Коэффициент детерминации R2: {:.3f}".format(lr.score(X_test, y_test)))
print("Коэффициенты регрессии: \n", coefficients)
print("Точка пересечения: \n ",intercept)
print("-------------------------------------------------")
print("Прогноз таргета на 01.2017 – 12.2017: \n", y_predict)

#визуализация полученных прогнозов
plt.figure(figsize=(13, 4))
plt.semilogy(df['Date'][72:], y_predict, label="Прогнозы LNR")
plt.legend()
plt.xlabel("Дата")
plt.ylabel("Объем рынка вкладов физлиц, руб.")


# Вывод

# В анализе были использованые разные модели, наиболее подходящая - линейная регрессия. Поскольку данные были представлены в разных масштабах в работе проведен подбор методов предобработки данных. В работе была выдвинута гипотеза об исключении сильнокоррелированного признака (Х9), которая в дальнейшем была отвергнута. И в целях повышения качества модели в работе применены методы добавления рандомных значений и полиномиальных элементов. По последнему методу были взяты три признака(Х1, Х4, Х6) на основе которых составлены полиномиальные элементы. После изменения функции коэффициент детерминации значительной степени повысился.
# На основе составленой модели полиномиальной регрессии был составлен прогноз на период с 01.01.2017 по 01.12.2017. 
# Интерпретация модели: при изменении доходность ОФЗ на 1% (признак Х1) уровень объемов рынка вкладов физических лиц упадет на 446,3 тыс. руб. при прочих равных. При изменении ставки по вкладам в долларах до 1 года (признак Х3) объемы увеличаться на 259,1 тыс.руб. при прочих равных. При изменении ставки по вкладам в рублях (признак Х4) объемы рынка вкладов физических лиц снизится на 288, 8 тыс. руб. при прочих равных. При изменении индекса потребительстких цен (Х6) на 1% объемы рынка вкладов снизятся на 171, тыс. руб. при прочих равных.
