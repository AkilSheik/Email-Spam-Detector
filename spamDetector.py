import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

#Reading the CSV file
df = pd.read_csv('spam_ham_dataset.csv')

#Check for duplicates
df.drop_duplicates(inplace = True)

nltk.download('stopwords')

def process_text(text):
    #1 Remoe punctuation
    #2 Remove stopwords(useless words)
    #3 return a list of clean text words

    #1 

    #List compression, where loops through each charecter to see if its a punctuation. This is still 
    #a string. 
    nopunc = [char for char in text if char not in string.punctuation]

    #Attaches a space at beggining of nopunc
    nopunc = ''.join(nopunc)

    #2 

    #Loops through every word in nopunc(split seperates text )
    clean_words = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    return clean_words

#Tokenization(shows tokenization)
messages_bow = CountVectorizer(analyzer=process_text).fit_transform(df['text'])
X_train, X_test, y_train, y_test = train_test_split(messages_bow, df['spam'], test_size = 0.20, random_state = 0)




#print(df['text'].head().apply(process_text))







#print(df.head(5))
#print("hello")

#Gets numbers of rows and columns
#print(df.shape)
#print(df.columns)


#print(df.shape)

#Shows NAN data for each column
#print(df.isnull().sum())

