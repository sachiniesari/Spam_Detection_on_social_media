
# coding: utf-8

# In[43]:


import pandas as pd
from random import shuffle
from sklearn.model_selection import  train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from io import StringIO
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

"""
Data file paths
"""
fakeNewlegDataPath=r'C:\Users\sachi\OneDrive\Desktop\TwitterSpamDetectionProject1MSC\Cleaned data\fakeNew\legitimate.csv'
fakeNewspamDataPath=r'C:\Users\sachi\OneDrive\Desktop\TwitterSpamDetectionProject1MSC\Cleaned data\fakeNew\fake.csv'
fakeOldlegDataPath=r'C:\Users\sachi\OneDrive\Desktop\TwitterSpamDetectionProject1MSC\Cleaned data\fakeOld\legitimate.csv'
fakeOldspamDataPath=r'C:\Users\sachi\OneDrive\Desktop\TwitterSpamDetectionProject1MSC\Cleaned data\fakeOld\spam.csv'
hnyNewlegDataPath=r'C:\Users\sachi\OneDrive\Desktop\TwitterSpamDetectionProject1MSC\Cleaned data\hnyNew\legitimate2.csv'
hnyNewspamDataPath=r'C:\Users\sachi\OneDrive\Desktop\TwitterSpamDetectionProject1MSC\Cleaned data\hnyNew\spam2.csv'
hnyOldlegDataPath=r'C:\Users\sachi\OneDrive\Desktop\TwitterSpamDetectionProject1MSC\Cleaned data\hnyOld\legitimate.csv'
hnyOldspamDataPath=r'C:\Users\sachi\OneDrive\Desktop\TwitterSpamDetectionProject1MSC\Cleaned data\hnyOld\spam.csv'



"""
Global variables
"""
totpredRF=0
totsenRF=0
totspcRF=0

totpredNB=0
totsenNB=0
totspcNB=0

totpredKNN=0
totsenKNN=0
totspcKNN=0

TPR = 0
TNR = 0
FPR = 0
FNR = 0

TPK = 0
TNK = 0
FPK = 0
FNK = 0

TPN = 0
TNN = 0
FPN = 0
FNN = 0

"""
Feature extraction methods
"""

def getNumberOfUppercaseLetters(tweet):
    count=0
    for i in tweet:
        if(i.isupper()):
            count=count+1
    return count

def getRationalNumbers(tweet):
    count=0
    for i in tweet:
        if(i.isdigit()):
            count=count+1
    return count

def getLength(tweet):
    tweet=str(tweet)
    return len(tweet)

def getNonAlNumCharacter(tweet):
    count=0
    for i in tweet:
        if(not(i.isalnum())):
            count=count+1
    return count

def getNumURLs(tweet):
    count=tweet.count("http")
    return count

def getNumHashtags(tweet):
    count=tweet.count("#")
    return count

def getNumMentions(tweet):
    count=tweet.count("@")
    return count

"""
Raw wise data creating methods
"""

def createData1(df):
    tweets = df["text"]
    uppercase=[]
    alphaNum=[]
    lengthTweet=[]
    numb=[]
    urls=[]
    hashtags=[]
    mentions=[]
    for i in range(len(df)):
            u=getNumberOfUppercaseLetters(tweets[i])
            uppercase.append(u)
            alpha=getNonAlNumCharacter(tweets[i])
            alphaNum.append(alpha)
            le=getLength(tweets[i])
            lengthTweet.append(le)
            n=getRationalNumbers(tweets[i])
            numb.append(n)
            url=getNumURLs(tweets[i])
            urls.append(url)
            hash=getNumHashtags(tweets[i])
            hashtags.append(hash)
            men=getNumMentions(tweets[i])
            mentions.append(men)
    return [hashtags,mentions,uppercase,alphaNum,urls,lengthTweet,numb]

def createData2(df):
    tweets = df["text"]
    urls = df["num_urls"]
    hashtags=df["num_urls"]
    mentions=df["num_mentions"]
    retweetCount=df["retweet_count"]
    favouriteCount=df["favorite_count"]
    retweeted=df["retweeted"]
    uppercase=[]
    alphaNum=[]
    lengthTweet=[]
    numb=[]
    for i in range(len(df)):
        u=getNumberOfUppercaseLetters(tweets[i])
        uppercase.append(u)
        alpha=getNonAlNumCharacter(tweets[i])
        alphaNum.append(alpha)
        le=getLength(tweets[i])
        lengthTweet.append(le)
        n=getRationalNumbers(tweets[i])
        numb.append(n)
    return [hashtags,mentions,uppercase,alphaNum,urls,lengthTweet,numb]

"""
Preprocessing and Data labeling methods
"""
def legitimateCreator(getData,list1):
    for i in range(1100):
        x=[int(getData[0][i]),int(getData[1][i]),int(getData[2][i]),int(getData[3][i]),int(getData[4][i]),int(getData[5][i]),int(getData[6][i])]
        tot=0
        for j in range(len(x)):
            tot=float(tot)+x[j]
        x=[int(getData[0][i])/tot,int(getData[1][i])/tot,int(getData[2][i])/tot,int(getData[3][i])/tot,int(getData[4][i])/tot,int(getData[5][i])/tot,int(getData[6][i])/tot,'real']
        list1.append(x)
        
def spamCreator(getData,list2):
    for i in range(1100):
        x=[int(getData[0][i]),int(getData[1][i]),int(getData[2][i]),int(getData[3][i]),int(getData[4][i]),int(getData[5][i]),int(getData[6][i])]
        tot=0
        for j in range(len(x)):
            tot=float(tot)+x[j]
        x=[int(getData[0][i])/tot,int(getData[1][i])/tot,int(getData[2][i])/tot,int(getData[3][i])/tot,int(getData[4][i])/tot,int(getData[5][i])/tot,int(getData[6][i])/tot,'spam']
        list2.append(x)

"""
results calculating methods
"""
            
def getResults(list):  
    
    global totpredRF
    global totsenRF
    global totspcRF

    global totpredNB
    global totsenNB
    global totspcNB

    global totpredKNN
    global totsenKNN
    global totspcKNN
    
    global TPR
    global TNR
    global FPR
    global FNR
    
    global TPK
    global TNK
    global FPK
    global FNK
    
    global TPN
    global TNN
    global FPN
    global FNN
    

    for i in range(10):
        shuffle(list)
        bigData=pd.DataFrame(list,columns=['hashtags','mentions','uppercase','alphaNum','urls','length','numbers','label'])
        target=bigData['label']
        X_train,X_test,Y_train,Y_test=train_test_split(bigData.drop(['label'],axis='columns'),target,test_size=0.15)

        modelRF.fit(X_train,Y_train)
        predRF=modelRF.score(X_test,Y_test) 
        totpredRF=totpredRF+predRF
        predictionsRF = modelRF.predict(X_test)
        (tn, fp, fn, tp)=confusion_matrix(Y_test, predictionsRF).ravel()
        FPR=FPR+fp
        FNR=FNR+fn
        TPR=TPR+tp
        TNR=TNR+tn
        (tn, fp, fn, tp)=(float(tn),float(fp),float(fn),float(tp))
        senRF=tp/(tp+fn)
        spcRF=tn/(tn+fp)
        totsenRF=totsenRF+senRF
        totspcRF=totspcRF+spcRF

        modelNB.fit(X_train,Y_train)
        predNB=modelNB.score(X_test,Y_test) 
        totpredNB=totpredNB+predNB
        predictionsNB = modelNB.predict(X_test)
        (tn, fp, fn, tp)=confusion_matrix(Y_test, predictionsNB).ravel()
        FPN=FPN+fp
        FNN=FNN+fn
        TPN=TPN+tp
        TNN=TNN+tn
        (tn, fp, fn, tp)=(float(tn),float(fp),float(fn),float(tp))
        senNB=tp/(tp+fn)
        spcNB=tn/(tn+fp)
        totsenNB=totsenNB+senNB
        totspcNB=totspcNB+spcNB

        modelKNN.fit(X_train,Y_train)
        predKNN=modelKNN.score(X_test,Y_test) 
        totpredKNN=totpredKNN+predKNN
        predictionsKNN = modelKNN.predict(X_test)
        (tn, fp, fn, tp)=confusion_matrix(Y_test, predictionsKNN).ravel()
        FPK=FPK+fp
        FNK=FNK+fn
        TPK=TPK+tp
        TNK=TNK+tn
        (tn, fp, fn, tp)=(float(tn),float(fp),float(fn),float(tp))
        senKNN=tp/(tp+fn)
        spcKNN=tn/(tn+fp)
        totsenKNN=totsenKNN+senKNN
        totspcKNN=totspcKNN+spcKNN
        if(i==0):
            print("") 
            print("Training : ",end =" ")
        print(".",end =" ")
        
        if(i==9):
            print("") 
            print("Train Data Count : ",len(X_train))
            print("Test Data Count : ",len(X_test))
            print("")

def getResults2(bigD):  
    
    global totpredRF
    global totsenRF
    global totspcRF

    global totpredNB
    global totsenNB
    global totspcNB

    global totpredKNN
    global totsenKNN
    global totspcKNN
    
    global TPR
    global TNR
    global FPR
    global FNR
    
    global TPK
    global TNK
    global FPK
    global FNK
    
    global TPN
    global TNN
    global FPN
    global FNN
    
    for i in range(10):

        bigD = bigD.sample(frac=1).reset_index(drop=True)

        tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')

        features = tfidf.fit_transform(bigD.text).toarray()
        labels = bigD.category_id

        X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=0.15)

        modelRF.fit(X_train,Y_train)
        predRF=modelRF.score(X_test,Y_test) 
        totpredRF=totpredRF+predRF
        predictionsRF = modelRF.predict(X_test)
        (tn, fp, fn, tp)=confusion_matrix(Y_test, predictionsRF).ravel()
        
        FPR=FPR+fp
        FNR=FNR+fn
        TPR=TPR+tp
        TNR=TNR+tn
        (tn, fp, fn, tp)=(float(tn),float(fp),float(fn),float(tp))
        senRF=tp/(tp+fn)
        spcRF=tn/(tn+fp)
        totsenRF=totsenRF+senRF
        totspcRF=totspcRF+spcRF

        modelNB.fit(X_train,Y_train)
        predNB=modelNB.score(X_test,Y_test) 
        totpredNB=totpredNB+predNB
        predictionsNB = modelNB.predict(X_test)
        (tn, fp, fn, tp)=confusion_matrix(Y_test, predictionsNB).ravel()
        
        FPN=FPN+fp
        FNN=FNN+fn
        TPN=TPN+tp
        TNN=TNN+tn
        (tn, fp, fn, tp)=(float(tn),float(fp),float(fn),float(tp))
        senNB=tp/(tp+fn)
        spcNB=tn/(tn+fp)
        totsenNB=totsenNB+senNB
        totspcNB=totspcNB+spcNB

        modelKNN.fit(X_train,Y_train)
        predKNN=modelKNN.score(X_test,Y_test) 
        totpredKNN=totpredKNN+predKNN
        predictionsKNN = modelKNN.predict(X_test)
        (tn, fp, fn, tp)=confusion_matrix(Y_test, predictionsKNN).ravel()
        
        FPK=FPK+fp
        FNK=FNK+fn
        TPK=TPK+tp
        TNK=TNK+tn
        (tn, fp, fn, tp)=(float(tn),float(fp),float(fn),float(tp))
        senKNN=tp/(tp+fn)
        spcKNN=tn/(tn+fp)
        totsenKNN=totsenKNN+senKNN
        totspcKNN=totspcKNN+spcKNN
        if(i==0):
            print("") 
            print("Training : ",end =" ")
        print(".",end =" ")
        
        if(i==9):
            print("") 
            print("Train Data Count : ",len(X_train))
            print("Test Data Count : ",len(X_test))
            print("")

"""
print results
"""
def printResults(dataset,totpredRF,totsenRF,totspcRF,totpredNB,totsenNB,totspcNB,totpredKNN,totsenKNN,totspcKNN,TPR,TNR,FPR,FNR,TPK,TNK,FPK,FNK,TPN,TNN,FPN,FNN):
    print("Data set : ",dataset)
    print("")
    print("Algorithm : Random Forest")
    print("Accuracy :",totpredRF/10)
    print("SEN :",totsenRF/10)
    print("SPC :",totspcRF/10)


    print("") 
    print("Algorithm : Naive bayes")
    print("Accuracy :",totpredNB/10)
    print("SEN :",totsenNB/10)
    print("SPC :",totspcNB/10)

    print("") 
    print("Algorithm : KNN")
    print("Accuracy :",totpredKNN/10)
    print("SEN :",totsenKNN/10)
    print("SPC :",totspcKNN/10)
    
    print("") 
    print("TPR,TNR,FPR,FNR : ",int(TPR/10),int(TNR/10),int(FPR/10),int(FNR/10))
    print("TPK,TNK,FPK,FNK : ",int(TPK/10),int(TNK/10),int(FPK/10),int(FNK/10))
    print("TPN,TNN,FPN,FNN : ",int(TPN/10),int(TNN/10),int(FPN/10),int(FNN/10))

    print('............................................')
    
"""
initialize
"""

def initiate():
    global totpredRF
    global totsenRF
    global totspcRF

    global totpredNB
    global totsenNB
    global totspcNB

    global totpredKNN
    global totsenKNN
    global totspcKNN
    
    global TPR
    global TNR
    global FPR
    global FNR
    
    global TPK
    global TNK
    global FPK
    global FNK
    
    global TPN
    global TNN
    global FPN
    global FNN
    
    totpredRF=0
    totsenRF=0
    totspcRF=0

    totpredNB=0
    totsenNB=0
    totspcNB=0

    totpredKNN=0
    totsenKNN=0
    totspcKNN=0

    TPR = 0
    TNR = 0
    FPR = 0
    FNR = 0

    TPK = 0
    TNK = 0
    FPK = 0
    FNK = 0

    TPN = 0
    TNN = 0
    FPN = 0
    FNN = 0
"""
/////////////////////////////D1/D2 implementation////////////////////////////////////////////
"""

print("")
print("")
print(" D1/D2 IMPLEMENTATION")
print("")
print("")
i=0
"""
fakeProjectNew
"""
cols= ["num","created_at","id","text","source","user_id","truncated","in_reply_to_status_id","in_reply_to_user_id","in_reply_to_screen_name","geo","place","contributors","retweet_count","favorite_count","favorited","retweeted"]
legDf = pd.read_csv(fakeNewlegDataPath, sep=",", header=None, names=cols)
spamDf = pd.read_csv(fakeNewspamDataPath, sep=",", header=None, names=cols)

getData=createData1(legDf)
list1=[]
legitimateCreator(getData,list1)

getData=createData1(spamDf)
list2=[]
spamCreator(getData,list2)

list=list1+list2
print("legitimate tweets : ",len(list1))
print("spam tweets : ",len(list2))
print("Tot : ",len(list))
print("Training started ....")

modelRF=RandomForestClassifier(n_estimators=50)
modelNB=GaussianNB()
modelKNN = KNeighborsClassifier(n_neighbors=45)

getResults(list)
printResults("fakeProjectNew",totpredRF,totsenRF,totspcRF,totpredNB,totsenNB,totspcNB,totpredKNN,totsenKNN,totspcKNN,TPR,TNR,FPR,FNR,TPK,TNK,FPK,FNK,TPN,TNN,FPN,FNN)

"""
fakeProjectOld
"""
initiate()
cols =["id","text","source","user_id","truncated","in_reply_to_status_id","in_reply_to_user_id","in_reply_to_screen_name","retweeted_status_id","geo","place","contributors","retweet_count","reply_count","favorite_count","favorited","retweeted","possibly_sensitive","num_hashtags","num_urls","num_mentions","created_at","timestamp","crawled_at","updated"]
legDf = pd.read_csv(fakeOldlegDataPath, sep=",", header=None, names=cols)

cols =["created_at","id","text","source","user_id","truncated","in_reply_to_status_id","in_reply_to_user_id","in_reply_to_screen_name","retweeted_status_id","geo","place","contributors","retweet_count","reply_count","favorite_count","favorited","retweeted","possibly_sensitive","num_hashtags","num_urls","num_mentions","timestamp"]
spamDf = pd.read_csv(fakeOldspamDataPath, sep=",", header=None, names=cols)

getData=createData2(legDf)
list1=[]
legitimateCreator(getData,list1)

getData=createData2(spamDf)
list2=[]
spamCreator(getData,list2)

list=list1+list2
print("")
print("legitimate tweets : ",len(list1))
print("spam tweets : ",len(list2))
print("Tot : ",len(list))
print("Training started ....")

modelRF=RandomForestClassifier(n_estimators=100)
modelNB=GaussianNB()
modelKNN = KNeighborsClassifier(n_neighbors=67)

getResults(list)
printResults("fakeProjectOld",totpredRF,totsenRF,totspcRF,totpredNB,totsenNB,totspcNB,totpredKNN,totsenKNN,totspcKNN,TPR,TNR,FPR,FNR,TPK,TNK,FPK,FNK,TPN,TNN,FPN,FNN)

"""
hnyPotNew
"""
cols =["userId","text","createdAt"," "]
legDf = pd.read_csv(hnyNewlegDataPath, sep=",", header=None, names=cols)
spamDf = pd.read_csv(hnyNewspamDataPath, sep=",", header=None, names=cols)

getData=createData1(legDf)
list1=[]
legitimateCreator(getData,list1)

getData=createData1(spamDf)
list2=[]
spamCreator(getData,list2)

list=list1+list2
print("")
print("legitimate tweets : ",len(list1))
print("spam tweets : ",len(list2))
print("Tot : ",len(list))
print("Training started ....")

modelRF=RandomForestClassifier(n_estimators=50)
modelNB=GaussianNB()
modelKNN = KNeighborsClassifier(n_neighbors=45)
initiate()

getResults(list)
printResults("hnyPotNew",totpredRF,totsenRF,totspcRF,totpredNB,totsenNB,totspcNB,totpredKNN,totsenKNN,totspcKNN,TPR,TNR,FPR,FNR,TPK,TNK,FPK,FNK,TPN,TNN,FPN,FNN)

"""
hnyPotOld
"""
cols =["userId","tweetID","text","createdAt"]
legDf = pd.read_csv(hnyOldlegDataPath, sep="\t", header=None, names=cols)
spamDf = pd.read_csv(hnyOldspamDataPath, sep="\t", header=None, names=cols)

getData=createData1(legDf)
list1=[]
legitimateCreator(getData,list1)

getData=createData1(spamDf)
list2=[]
spamCreator(getData,list2)

list=list1+list2
print("")
print("legitimate tweets : ",len(list1))
print("spam tweets : ",len(list2))
print("Tot : ",len(list))
print("Training started ....")

modelRF=RandomForestClassifier(n_estimators=50)
modelNB=GaussianNB()
modelKNN = KNeighborsClassifier(n_neighbors=45)
initiate()

getResults(list)
printResults("hnyPotOld",totpredRF,totsenRF,totspcRF,totpredNB,totsenNB,totspcNB,totpredKNN,totsenKNN,totspcKNN,TPR,TNR,FPR,FNR,TPK,TNK,FPK,FNK,TPN,TNN,FPN,FNN)


"""
/////////////////////////////D4 implementation////////////////////////////////////////////
"""

print("")
print("")
print(" VSM IMPLEMENTATION")
print("")
print("")

"""
hnypotnew
"""
cols = ["userId","text","createdAt"," "]
df = pd.read_csv(hnyNewspamDataPath, sep=",", header=None, names=cols)
df['label'] = 'spam'
df=df[:1100]

cols = ["userId","text","createdAt"," "]
df2 = pd.read_csv(hnyNewlegDataPath, sep=",", header=None, names=cols)
df2=df2[:1100]
df2['label'] = 'legitimate'

bigD=pd.concat([df,df2])
bigD=bigD.reset_index(drop=True)

col = ['label', 'text']
bigD = bigD[col]
bigD.columns = ['label', 'text']
bigD['category_id'] = bigD['label'].factorize()[0]
category_id_df2 = bigD[['label', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id2 = dict(category_id_df2.values)
id_to_category2 = dict(category_id_df2[['category_id', 'label']].values)
initiate()

modelRF=RandomForestClassifier(n_estimators=100)
modelNB=GaussianNB()
modelKNN = KNeighborsClassifier(n_neighbors=67)

bigD = bigD.sample(frac=1).reset_index(drop=True)
print("data : " , len(bigD))
print("starting .... ")
getResults2(bigD)
printResults("hnyPotNew",totpredRF,totsenRF,totspcRF,totpredNB,totsenNB,totspcNB,totpredKNN,totsenKNN,totspcKNN,TPR,TNR,FPR,FNR,TPK,TNK,FPK,FNK,TPN,TNN,FPN,FNN)

"""
hnypotold
"""
cols = ["userId","tweetID","text","createdAt"]

df = pd.read_csv(hnyOldspamDataPath, sep="\t", header=None, names=cols)
df['label'] = 'spam'
df=df[:1100]

df2 = pd.read_csv(hnyOldlegDataPath, sep="\t", header=None, names=cols)
df2['label'] = 'legitimate'
df2=df2[:1100]
bigD=pd.concat([df,df2])
bigD=bigD.reset_index(drop=True)

col = ['label', 'text']
bigD = bigD[col]
bigD.columns = ['label', 'text']
bigD['category_id'] = bigD['label'].factorize()[0]
category_id_df2 = bigD[['label', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id2 = dict(category_id_df2.values)
id_to_category2 = dict(category_id_df2[['category_id', 'label']].values)
initiate()

modelRF=RandomForestClassifier(n_estimators=100)
modelNB=GaussianNB()
modelKNN = KNeighborsClassifier(n_neighbors=67)

bigD = bigD.sample(frac=1).reset_index(drop=True)
print("data : " , len(bigD))
print("starting .... ")
getResults2(bigD)
printResults("hnyPotOld",totpredRF,totsenRF,totspcRF,totpredNB,totsenNB,totspcNB,totpredKNN,totsenKNN,totspcKNN,TPR,TNR,FPR,FNR,TPK,TNK,FPK,FNK,TPN,TNN,FPN,FNN)

"""
fakeProjectOld
"""
cols =["created_at","id","text","source","user_id","truncated","in_reply_to_status_id","in_reply_to_user_id","in_reply_to_screen_name","retweeted_status_id","geo","place","contributors","retweet_count","reply_count","favorite_count","favorited","retweeted","possibly_sensitive","num_hashtags","num_urls","num_mentions","timestamp"]

df = pd.read_csv(fakeOldspamDataPath, sep=",", header=None, names=cols)
df['label'] = 'spam'
df=df[:1100]
cols =["id","text","source","user_id","truncated","in_reply_to_status_id","in_reply_to_user_id","in_reply_to_screen_name","retweeted_status_id","geo","place","contributors","retweet_count","reply_count","favorite_count","favorited","retweeted","possibly_sensitive","num_hashtags","num_urls","num_mentions","created_at","timestamp","crawled_at","updated"]

df2 = pd.read_csv(fakeOldlegDataPath, sep=",", header=None, names=cols)
df2['label'] = 'legitimate'
df2=df2[:1100]
bigD=pd.concat([df,df2])
bigD=bigD.reset_index(drop=True)

col = ['label', 'text']
bigD = bigD[col]
bigD.columns = ['label', 'text']
bigD['category_id'] = bigD['label'].factorize()[0]
category_id_df2 = bigD[['label', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id2 = dict(category_id_df2.values)
id_to_category2 = dict(category_id_df2[['category_id', 'label']].values)
initiate()

modelRF=RandomForestClassifier(n_estimators=100)
modelNB=GaussianNB()
modelKNN = KNeighborsClassifier(n_neighbors=67)

bigD = bigD.sample(frac=1).reset_index(drop=True)
print("data : " , len(bigD))
print("starting .... ")
getResults2(bigD)
printResults("fakeProjectOld",totpredRF,totsenRF,totspcRF,totpredNB,totsenNB,totspcNB,totpredKNN,totsenKNN,totspcKNN,TPR,TNR,FPR,FNR,TPK,TNK,FPK,FNK,TPN,TNN,FPN,FNN)

"""
fakeProjectNew
"""
cols= ["num","created_at","id","text","source","user_id","truncated","in_reply_to_status_id","in_reply_to_user_id","in_reply_to_screen_name","geo","place","contributors","retweet_count","favorite_count","favorited","retweeted"]

df = pd.read_csv(fakeNewspamDataPath, sep=",", header=None, names=cols)
df['label'] = 'spam'
df=df[:1100]

df2 = pd.read_csv(fakeNewlegDataPath, sep=",", header=None, names=cols)
df2['label'] = 'legitimate'
df2=df2[:1100]
bigD=pd.concat([df,df2])
bigD=bigD.reset_index(drop=True)

col = ['label', 'text']
bigD = bigD[col]
bigD.columns = ['label', 'text']
bigD['category_id'] = bigD['label'].factorize()[0]
category_id_df2 = bigD[['label', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id2 = dict(category_id_df2.values)
id_to_category2 = dict(category_id_df2[['category_id', 'label']].values)
initiate()

modelRF=RandomForestClassifier(n_estimators=100)
modelNB=GaussianNB()
modelKNN = KNeighborsClassifier(n_neighbors=67)

bigD = bigD.sample(frac=1).reset_index(drop=True)
print("data : " , len(bigD))
print("starting .... ")
getResults2(bigD)
printResults("fakeProjectNew",totpredRF,totsenRF,totspcRF,totpredNB,totsenNB,totspcNB,totpredKNN,totsenKNN,totspcKNN,TPR,TNR,FPR,FNR,TPK,TNK,FPK,FNK,TPN,TNN,FPN,FNN)

