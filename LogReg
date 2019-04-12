import numpy  as np
import pandas as pd
#import nltk   as nl
import pandas as pd
train = pd.read_csv("Train.csv", dtype=object)
test  = pd.read_csv("Test.csv", dtype=object)
df=train
scrapedyear=[]
scrapedmonth=[]
scrapedday=[]
n=len(df.scraped_at)
for i in range(n):
    scrapedyear.append(df.scraped_at[i][0:4])
    scrapedmonth.append(df.scraped_at[i][5:7])
    scrapedday.append(df.scraped_at[i][8:10])
df['scrapedyear']=scrapedyear
df['scrapedmonth']=scrapedmonth
df['scrapedday']=scrapedday
for i in range(n):
    if type(df['meta_keywords'][i])!=str:
        df['meta_keywords'][i]='nan'
for i in range(n):
    if type(df['authors'][i])!=str:
        df['authors'][i]='nan'
for i in range(n):
    if type(df['title'][i])!=str:
        df['title'][i]='nan' 
for i in range(n):
    if type(df['meta_description'][i])!=str:
        df['meta_description'][i]='nan'
for i in range(n):
    if type(df['tags'][i])!=str:
        df['tags'][i]='nan'        
df['content']=df['title']+' '+df['content']+' '+df['authors']+' '+df['scrapedyear']+' '+df['scrapedmonth']+' '+df['meta_keywords']+' '+df['meta_description']+' '+df['tags']
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
y = df['fake']
X = df[['content','scrapedyear']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=53)
count_vectorizer = CountVectorizer(stop_words='english')
count_train = count_vectorizer.fit_transform(X_train.content.values)
count_test = count_vectorizer.transform(X_test.content.values)
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
nb_classifier = LogisticRegression(C=0.1,penalty='l1')
nb_classifier.fit(count_train, y_train)
pred = nb_classifier.predict(count_test)
pred = [pred[i] if X_test.scrapedyear.values[i]!='2016' else 0 for i in range(len(pred))]
score = metrics.accuracy_score(y_test, pred)
print(score)
