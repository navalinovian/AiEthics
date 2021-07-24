from django.shortcuts import render, redirect, reverse
from django.conf import settings
from django.contrib.sessions.models import Session
from rest_framework import generics
from .serializers import SentimentSerializer
from .models import Sentiment, Implement
import pandas as pd
import re, sys, json
import numpy as np
import base64
import spacy
from io import BytesIO
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
from pylab import figure
import srsly
import os

from googleapiclient.discovery import build


def Preprocessing(comments):
    stopword_result = []
    regex_result = []
    lemmatizing_result = []
    nlp = spacy.load('en_core_web_sm')
    patterns = srsly.read_json(os.path.join(settings.BASE_DIR, 'ar_patterns.json'))
    nlp.remove_pipe("attribute_ruler")
    ar = nlp.add_pipe("attribute_ruler", before="lemmatizer")
    ar.add_patterns(patterns)
    
    comments = comments.loc[:, ~comments.columns.str.contains('^Unnamed')]
    comments = comments.drop(comments[comments.label == 'Irrelevant'].index)
    comments = comments.drop(comments[comments.label == 'x'].index)
    comments = comments[comments['label'].notna()]
    comments = comments[comments['text'].notna()]
    
    for line in comments['text']:
        lowercase = line.lower()
        lemmatize = nlp(lowercase)
        sentence = [token.lemma_ for token in lemmatize]
        lemmatizing_result.append(sentence)
        
        removestopword = (word for word in sentence if  nlp.vocab[word].is_stop is False)
        join = ' '.join(removestopword)
        regex = re.sub("[^a-zA-Z0-9 ]+", " ", join)
        regex_result.append(regex)
        stopword_result.append(join)
     
    comments['lemmatized'] = lemmatizing_result
    comments['stopword'] = stopword_result
    comments['preprocessed'] = regex_result
    return comments

def Implementing(dataFrame,test):
    tfidfv = TfidfVectorizer(min_df = 0.01 ,max_df = 0.3, stop_words='english')
    nb = MultinomialNB(alpha=0.8)
    label_types = dataFrame.label.value_counts().sort_index()
    dataFrame['labelNumber'] = dataFrame['label']
    for i in range(0,len(label_types)):
        dataFrame['labelNumber'] = dataFrame['labelNumber'].replace('{}'.format(label_types.index[i]),'{}'.format(i))
    
    comment = dataFrame['preprocessed']
    label = dataFrame['labelNumber']

    comment_train_vector = tfidfv.fit_transform(comment)
    
    test_vector = tfidfv.transform(test)
    nb.fit(comment_train_vector, label)
    predict = nb.predict(test_vector)

    for i in range(0,len(label_types)):
        predict = np.where(predict=="{}".format(i), label_types.index[i],predict)

    return predict

def Process(dataFrame, fold):
    train_accuracies = []
    test_accuracies = []
    confusions = []
    reports = []
    
    tfidfv = TfidfVectorizer(min_df = 0.01 ,max_df = 0.3, stop_words='english')
    nb = MultinomialNB(alpha=1.5)
    
    label_types = dataFrame.label.value_counts().sort_index()
    dataFrame['labelNumber'] = dataFrame['label']
    for i in range(0,len(label_types)):
        dataFrame['labelNumber'] = dataFrame['labelNumber'].replace('{}'.format(label_types.index[i]),'{}'.format(i))
    
    comment = dataFrame['preprocessed']
    label = dataFrame['labelNumber']
    
    kf = KFold(n_splits=fold)
    
    for train_index, test_index in kf.split(comment) :
        X_train, X_test = comment.iloc[train_index], comment.iloc[test_index]
        y_train, y_test = label.iloc[train_index], label.iloc[test_index]
        comment_train_vector = tfidfv.fit_transform(X_train)
        comment_test_vector = tfidfv.transform(X_test)

        nb.fit(comment_train_vector, y_train)
        prediction = nb.predict(comment_test_vector)
        expect = y_test
        train_accuracy = accuracy_score(y_train, nb.predict(comment_train_vector))
        test_accuracy = accuracy_score(expect, prediction)
        confusion = confusion_matrix(expect,prediction, labels=nb.classes_)
        precision = precision_score(expect, prediction, average='weighted')
        recall = recall_score(expect, prediction, average='weighted')
        f1 = f1_score(expect, prediction, average='weighted')
        report = {
            'precision' : precision,
            'recall': recall,
            'f1':f1
        } 
        test_accuracies.append(test_accuracy)
        train_accuracies.append(train_accuracy)
        confusions.append(confusion)
        reports.append(report)

    result =  {
        'confusions' : confusions,
        'train_accuracies' : train_accuracies,
        'test_accuracies' : test_accuracies,
        'reports' : reports,
    }
    return result

def get_graph():
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()

    graphic = base64.b64encode(image_png)
    graphic = graphic.decode('utf-8')

    return graphic

# Create your views here.
class SentimentView(generics.ListAPIView):
    queryset = Sentiment.objects.all()
    serializer_class = SentimentSerializer

def sentiment_data_upload_view(request):
    print(sys.version)
    return render(request, "example.html",{})

def sentiment_data_view(request):
    if request.method =="POST":
        file_excel = request.FILES['name_file']
        Sentiment.objects.all().delete()
        Session.objects.all().delete()
        df = pd.read_excel(file_excel , engine='openpyxl')
        for index, row in df.iterrows():
            text = row['comment']
            label = row['label_result']
            Sentiment(
                text=text,
                label=label
            ).save()
        
    if request.session.get('for_processing')== None:
        data = pd.DataFrame(list(Sentiment.objects.all().values()))

    else:
        dictionary = request.session.get('for_processing')
        data_json = json.loads(dictionary)
        data =  pd.DataFrame.from_dict(data_json, orient='columns')    
        
    data_column = data.columns.values
    data_numpy = data.to_numpy()

    context =  {
        'data' : data_numpy,
        'data_column' : data_column,
        'next_path': 'data-preprocessing',
    }
    request.session['for_processing'] = data.to_json()
    return render(request, "data_view.html", context)

def sentiment_data_preprocessing(request):
    if request.session.get('for_evaluation')== None:
        dictionary = request.session.get('for_processing')
        data_json = json.loads(dictionary)
        data =  pd.DataFrame.from_dict(data_json, orient='columns')    
        data = Preprocessing(data)
    else:
        dictionary = request.session.get('for_evaluation')
        data_json = json.loads(dictionary)
        data =  pd.DataFrame.from_dict(data_json, orient='columns')

    data_column = ['Before Preprocessing','After Preprocessing','label']
    data_numpy = data[['text','preprocessed','label']].head().to_numpy()
    context =  {
        'data' : data_numpy,
        'fold' : range(4,11),
        'data_column' : data_column,
        'next_path': 'evaluate',
    }
    request.session['for_evaluation'] = data.to_json()
    return render(request, "data_view.html", context)

def sentiment_evaluate(request, surrogate=0):
    if request.method =="POST":
        fold = request.POST.get('k_amount')
        request.session['fold'] = fold

    dictionary = request.session.get('for_evaluation')
    data_json = json.loads(dictionary)
    data =  pd.DataFrame.from_dict(data_json, orient='columns')
    fold_session = request.session.get('fold')
    process_result = Process(data, int(fold_session))
    # request.session['process_result'] = process_result
   
    confusions = process_result['confusions']
    accuracyScore = process_result['test_accuracies']
    report = process_result['reports'][surrogate]
    data_confusion = confusions[surrogate]
    disp = ConfusionMatrixDisplay(confusion_matrix=data_confusion)
    disp = disp.plot(cmap="Blues")

    context =  {
        'accuracy':accuracyScore[surrogate],
        'data' : get_graph(),
        'fold' : range(int(fold_session)),
        'report': report,
        'next_path': 'youtube',
    }
    return render(request, "evaluate.html", context)


def get_comment(pageToken):
    comments = []
    api_key = 'AIzaSyCJPLzGVlcC8kLE6b7hDgdSQMHz-rn-hus'
    service = build('youtube','v3',developerKey=api_key)
    channelId='UwsrzCVZAb8'
    
    response = service.commentThreads().list(part='snippet',videoId=channelId, textFormat='plainText', order="time",
    searchTerms="ai", maxResults=100, pageToken=pageToken).execute()
    for item in response['items']:
        comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
        comments.append(comment)
    if 'nextPageToken' in response:
        nextPageToken = response['nextPageToken']
    else:
        nextPageToken = '0'
    result = {
        'comments' : comments,
        'nextPageToken' :nextPageToken
    }
    
    return result

def youtube_api(request, **kwargs):
    dictionary = request.session.get('for_evaluation')
    data_json = json.loads(dictionary)
    data =  pd.DataFrame.from_dict(data_json, orient='columns')
    data2 = pd.DataFrame(list(Implement.objects.all().values()))

    state = True
    database = np.array(data['text'])
    TableImplement = np.array(data2['text'])
    if 'text' in data2:
        database = np.concatenate((database, TableImplement))
    temp= []
    max_loop = 10
    next_page_token = ''

    while state:
        print(next_page_token)
        results = get_comment(next_page_token)
        comments = np.array(results['comments'])
        comments = comments[~np.isin(comments, database)]
        temp = np.concatenate((temp, comments))
        temp = np.unique(temp,axis=0)
        if len(temp) >= 20 or max_loop<=0 or results['nextPageToken']=='0' :
            state = False
        else:
            next_page_token = results['nextPageToken']

    if len(temp)==0:
        temp = np.array(['Data Kosong'])
        sentiment = '-'
    else:
        sentiment = pd.DataFrame(Implementing(data, temp))
        sentiment = Implementing(data, temp)
        label = pd.DataFrame(sentiment)
        print(label)
        for index, row in label.iterrows():
            Implement(
                        text=temp[index],
                        label=row[0]
                    ).save()

    context =  {
        'data' : zip(temp,sentiment), 
    }
    return render(request, "youtube_api.html", context)

def select_kfold(request):
    return render(request, "kfold.html",{})
