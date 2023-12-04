import numpy as np
import pandas as pd
from pandas import Series, DataFrame

csv = pd.read_csv('2019_kbo_for_kaggle_v2.csv')

##############1#############

data = pd.DataFrame(csv, columns=['batter_name', 'H', 'avg', 'HR', 'OBP', 'year'])
data_year=[data[data['year'] == 2015].copy(), data[data['year'] == 2016].copy(),
           data[data['year'] == 2017].copy(), data[data['year'] == 2018].copy()]
rank_index = ['H_r', 'avg_r', 'HR_r', 'OBP_r']
columns_index = ['H', 'avg', 'HR', 'OBP']
year = 2015


for d in data_year:
    for i in range(4):
        d.loc[:, rank_index[i]] = d[columns_index[i]].rank(ascending=False)
        top10_data = d.sort_values(by=rank_index[i]).head(10)
        print("\n*** 1 _",year, columns_index[i],"***")
        print(top10_data['batter_name'].values)
    year+=1


##############2#############

warData = pd.DataFrame(csv, columns=['batter_name', 'cp', 'war', 'year'])

warData18 = warData[warData['year'] == 2018].copy()

position=['포수','1루수','2루수','3루수','유격수','좌익수','중견수','우익수']
print("\n\n***2***")
for p in position:
    pwarData18 = warData18[warData18['cp'] == p].copy()
    pwarData18.loc[:, p+'_r'] = pwarData18['war'].rank(ascending=False)
    highestWar = pwarData18.sort_values(by=p+'_r').head(1)
    print(p + " ")
    print(highestWar['batter_name'].values)


##############3#############

salData = pd.DataFrame(csv, columns=['R', 'H', 'HR', 'RBI', 'SB', 'war', 'avg', 'OBP', 'SLG', 'salary'])

relation=['R', 'H', 'HR', 'RBI', 'SB', 'war', 'avg', 'OBP', 'SLG']

print("\n\n***3***")
for r in relation:
    salData.loc[:, r+'_r']=salData[r].rank(ascending=False)

salData.loc[:, "salary_r"]=salData['salary'].rank(ascending=False)

zero=np.zeros(9)
count=dict(zip(relation,zero))    #다른 값을 갖는 행 개수
for r in relation:
    if salData[r + '_r'].equals(salData['salary_r']):
        count[r] = 0
    else:
        count[r] = sum(salData[r + '_r'] != salData['salary_r'])

leastValue=count['R']
leastKey=""
for r in relation:
    if count[r]<leastValue:
        leastKey=r
        leastValue=count[r]

print("the highest correlation with salary: " + leastKey)