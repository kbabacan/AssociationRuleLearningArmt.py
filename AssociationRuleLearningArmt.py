import pandas as pd
import datetime as dt
from mlxtend.frequent_patterns import apriori,association_rules
pd.set_option('display.max_columns' , None)

## Datayı okut
df= pd.read_csv("Data/armut_data.csv")
df.head()
df.iloc[:, -3:]

df.shape
df.head()
df.info()
df["UserId"].value_counts()
df[df["UserId"]==7256]

##Hizmet adında bir feature yarat, Serviceıd ve CategoryId _ ile yanyana yazarak olsun
df["Hizmet"] = df["ServiceId"].astype(str) + "_" + df["CategoryId"].astype(str)


##CreateDate tarihindeki yıl ve ay değişkenini yeni değişkene ekle
df['CreateDate'] = pd.to_datetime(df['CreateDate'])
df["NewDate"] = df["CreateDate"].dt.year.astype(str) + "-" +df["CreateDate"].dt.month.astype(str)
df["Sepet_ID"] = df["UserId"].astype(str) + "_" + df["CreateDate"].dt.year.astype(str) + "-" +df["CreateDate"].dt.month.astype(str)


##pivot table
df.groupby(["Sepet_ID","Hizmet"]).agg({"CreateDate":"count"}).unstack().iloc[0:5,0:5]

nf=df.groupby(["Sepet_ID","Hizmet"]).agg({"CreateDate":"count"}).unstack().fillna(0)

nf=df.groupby(["Sepet_ID","Hizmet"]).agg({"CreateDate":"count"}).unstack().fillna(0).map(lambda x:1 if x>0 else 0)

############################################
# 5. Sepet Aşamasındaki Kullanıcılara Ürün Önerisinde Bulunmak
############################################

freqItem= apriori(nf,
                  min_support=0.01,
                  use_colnames=True)

freqItem.sort_values("support",ascending=False)

rules = association_rules(freqItem,metric="support",min_threshold=0.01)

rules[(rules["support"]>0.01) & (rules["confidence"]>0.1) & (rules["lift"]>5)]. \
sort_values("confidence", ascending=False)

hizmet_id = '2_0'

sorted_rules = rules.sort_values("lift", ascending=False)
recommendation_list = []


for i, product in enumerate(sorted_rules["antecedents"]):
    print(i, product)
    for j in list(product):
        print(j)
        if j == hizmet_id:
            recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])




