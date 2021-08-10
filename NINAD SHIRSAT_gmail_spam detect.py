import pandas as pd
import numpy as np

dt = pd.read_csv("spam.csv")
dt.head(5)

dt['spam'] = dt['type'].map({'spam':1,'ham':0}).astype(int)

print('Columns in the given data: ')
for col in dt.columns:
    print(col)
    
t=len(dt['type'])
print("No. of rows in review column : ",t)
t=len(dt['text'])
print("No. of rows in liked column: ",t)

def tokenizer(text):
    return text.split()
dt['text']=dt['text'].apply(tokenizer)

from nltk.stem.snowball import SnowballStemmer
porter = SnowballStemmer("english", ignore_stopwords=False)

def stem_it(text):
    return [porter.stem(word) for word in text]

dt['text']=dt['text'].apply(stem_it)

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
def lemmit_it(text):
    return[lemmatizer.lemmatize(word, pos = "a") for word in text]
dt['text']=dt['text'].apply(lemmit_it)

from nltk.corpus import stopwords
stop_words = stopwords.words('english')

def stop_it(text):
    review = [word for word in text if not word in stop_words]
    return review

dt['text']=dt['text'].apply(stop_it)

from sklearn.feature_extraction.text import TfidfVectorizer 
tfidf=TfidfVectorizer()
y=dt.spam.values
x=tfidf.fit_transform(dt['text'])

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=1,test_size=0.2,shuffle=False)

from sklearn.linear_model import LogisticRegression
clf=LogisticRegression()
clf.fit(x_train,y_train)
y_predict=clf.predict(y_text)

from sklearn.metrics import accuracy_score
acc_log=accuracy_score(y_pred,y_text) * 100
print("Accuracy: ", acc_log)

from sklearn.svm import LinearSVC
linear_svc = LinearSVC(random_state=0)
linear_svc.fit(x_train,y_train)
y_pred = linear_svc.predict(x_text)
acc_linear_svc = accuracy_score(y_pred, y_text) * 100
print("accuracy :",acc_linear_svc)