import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from sklearn.metrics import f1_score
import numpy as np
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split







twitter_df=pd.read_csv("N:\Machine learning\Algorithms\sentiment_analysis\\train.csv")
# print(twitter_df.head())
# print(twitter_df.info())
twitter_df=twitter_df.dropna()
twitter_df = twitter_df.reset_index()
# print(twitter_df.isnull().any())


twitter_df=twitter_df.drop(['textID'],axis=1)
# print(twitter_df.info())
stopset = set(stopwords.words("english"))

# twitter_df["selected_text"]=twitter_df["selected_text"].str.strip().str.lower()
# twitter_df["text"]=twitter_df["text"].str.strip().str.lower()

twitter_df['sentiment'] = twitter_df['sentiment'].map({'positive': 1,
                             'negative': -1,
                             'neutral': 0},
                             na_action=None)

count=sns.countplot(data= twitter_df, x= 'sentiment',
             order = twitter_df['sentiment'].value_counts().index)

plt.show()



from sklearn.feature_extraction.text import CountVectorizer



#--------EDA----------------

positive = twitter_df[twitter_df['sentiment'] == 1]

plt.rcParams['figure.figsize'] = (10, 10)
plt.style.use('fast')

wc = WordCloud(background_color = 'orange', width = 1500, height = 1500).generate(str(positive['text']))
plt.title('Description Positive', fontsize = 15)

plt.imshow(wc)
plt.axis('off')
plt.show()

negative = twitter_df[twitter_df['sentiment'] == -1]

plt.rcParams['figure.figsize'] = (10, 10)
plt.style.use('fast')

wc = WordCloud(background_color = 'orange', width = 1500, height = 1500).generate(str(negative['text']))
plt.title('Description Negative', fontsize = 15)

plt.imshow(wc)
plt.axis('off')
plt.show()


neutral = twitter_df[twitter_df['sentiment'] == 0]

plt.rcParams['figure.figsize'] = (10, 10)
plt.style.use('fast')

wc = WordCloud(background_color = 'orange', width = 1500, height = 1500).generate(str(neutral['text']))
plt.title('Description Neutral', fontsize = 15)

plt.imshow(wc)
plt.axis('off')
plt.show()





y=twitter_df['sentiment']

x=twitter_df['selected_text']
# print(x.describe)
corpus = []
for i in range(0, len(x)):
  twitter = re.sub(r"@[A-Za-z0-9]+", ' ', x[i])
  twitter = re.sub(r"https?://[A-Za-z0-9./]+", ' ', x[i])
  twitter = re.sub(r"[^a-zA-Z.!?]", ' ', x[i])
  twitter = re.sub(r" +", ' ', x[i])
  twitter = twitter.split()
  ps = PorterStemmer()
  twitter = [ps.stem(word) for word in twitter if not word in set(stopwords.words('english'))]
  twitter = ' '.join(twitter)
  corpus.append(twitter)

cv = CountVectorizer(stop_words='english')

x=cv.fit_transform(corpus)


x_train,x_test,y_train,y_test=train_test_split(x,y,stratify=y,test_size=0.2,random_state=5)

from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

# y_train=y_train.reshape(-1,1)
# y_test=y_test.reshape(-1,1)



print("Accuracy Score = ",accuracy_score(y_test,y_pred))              
print("precision score = ",precision_score(y_test, y_pred,average='weighted'))         
print("recall score = ",recall_score(y_test, y_pred,average='micro'))               
print("f1 score = ",f1_score(y_test, y_pred, average='weighted'))
errors = abs(y_pred - y_test)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

