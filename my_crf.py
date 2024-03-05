#importing all the needed libraries     
import sklearn_crfsuite
from sklearn.metrics import classification_report
from sklearn_crfsuite import metrics
from sklearn.model_selection import train_test_split
from read_corpus import tagged_sentences
from collections import Counter
import pickle
print('imported')

nltk_data = tagged_sentences

print(nltk_data)
# split data into training and validation and test set
train_set,test_set = train_test_split(nltk_data,train_size=0.85,test_size=0.15,random_state=101)
train_set,val_set = train_test_split(nltk_data,train_size=0.80,test_size=0.20,random_state=1)

# extract features from a given sentence
def word_features(sent, i):
    word = sent[i][0]
    pos = sent[i][1]
    
    # first word
    if i==0:
        prevword = '<START>'
        prevpos = '<START>'
    else:
        prevword = sent[i-1][0]
        prevpos = sent[i-1][1]
        
    # first word
    if i==0 or i==1:
        prev2word = '<START>'
        prev2pos = '<START>'
    else:
        prev2word = sent[i-2][0]
        prev2pos = sent[i-2][1]
    
    # last word
    if i == len(sent)-1:
        nextword = '<END>'
        nextpos = '<END>'
    else:
        nextword = sent[i+1][0]
        nextpos = sent[i+1][1]
    
    # suffixes and prefixes
    pref_1, pref_2, pref_3, pref_4 = word[:1], word[:2], word[:3], word[:4]
    suff_1, suff_2, suff_3, suff_4 = word[-1:], word[-2:], word[-3:], word[-4:]
    
    # rule_state = rule_based_tagger.tag([word])[0][1]
    
    return {'word':word,            
            'prevword': prevword,
            'prevpos': prevpos,  
            'nextword': nextword, 
            'nextpos': nextpos,          
            'suff_1': suff_1,  
            'suff_2': suff_2,  
            'suff_3': suff_3,  
            'suff_4': suff_4, 
            'pref_1': pref_1,  
            'pref_2': pref_2,  
            'pref_3': pref_3, 
            'pref_4': pref_4,
            'prev2word': prev2word,
            'prev2pos': prev2pos           
           }  

# let's check if our word feature is working correctly:
print(train_set[0][0:6])

# defining a few more functions to extract featrues, postags and words from sentences

def sent2features(sent):
    return [word_features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [postag for word, postag in sent]

def sent2tokens(sent):
    return [word for word, postag in sent]   

 # create training, validation and test sets
X_train = [sent2features(s) for s in train_set]
y_train = [sent2labels(s) for s in train_set]

X_valid = [sent2features(s) for s in val_set]
y_valid = [sent2labels(s) for s in val_set]

X_test = [sent2features(s) for s in test_set]
y_test = [sent2labels(s) for s in test_set]

# check the train set produced
print(X_train[0][0:10])
print(y_train[0][0:10])

# fitting crf with arbitrary hyperparameters
crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.01,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True   
)
crf.fit(X_train, y_train)

with open('model/crf_model.pickle', 'wb') as f:
    pickle.dump(crf, f)

labels = list(crf.classes_)

ypred = crf.predict(X_train)
print('F1 score on the train set = {}\n'.format(metrics.flat_f1_score(y_train, ypred, average='weighted', labels=labels)))
print('Accuracy on the train set = {}\n'.format(metrics.flat_accuracy_score(y_train, ypred)))

sorted_labels = sorted(
    labels,
    key=lambda name: (name[1:], name[0])
)
y_train_flat = [item for sublist in y_train for item in sublist]
ypred_flat = [item for sublist in ypred for item in sublist]
print('Train set classification report: \n\n{}'.format(classification_report(y_train_flat, ypred_flat, labels=sorted_labels, digits=3)))

#obtaining metrics such as accuracy, etc. on the test set
ypred = crf.predict(X_test)
print('F1 score on the test set = {}\n'.format(metrics.flat_f1_score(y_test, ypred,
average='weighted', labels=labels)))
print('Accuracy on the test set = {}\n'.format(metrics.flat_accuracy_score(y_test, ypred)))
sorted_labels = sorted(
    labels,
    key=lambda name: (name[1:], name[0])
)

y_test_flat = [item for sublist in y_test for item in sublist]
ypred_flat = [item for sublist in ypred for item in sublist]
print('Train set classification report: \n\n{}'.format(classification_report(y_test_flat, ypred_flat, labels=sorted_labels, digits=3)))

def print_transitions(transition_features):
    for (label_from, label_to), weight in transition_features:
        print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))

print("Top 10 likely transitions - \n")
print_transitions(Counter(crf.transition_features_).most_common(10))

print("\nTop 10 unlikely transitions - \n")
print_transitions(Counter(crf.transition_features_).most_common()[-10:])