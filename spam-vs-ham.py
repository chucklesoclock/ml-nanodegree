from __future__ import division
import pandas as pd
from string import punctuation
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer   # , ENGLISH_STOP_WORDS
from sklearn.cross_validation import train_test_split
# from sklearn.model_selection import train_test_split #scikit-learn 0.18
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_table('SMSSpamCollection', sep='\t', header=None, names=['label', 'sms'])
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# ## MANUAL BAG OF WORDS IMPLEMENTATION
# preprocessed_documents = [sms.lower().translate(None, punctuation).split() for sms in df['sms']]
# frequency_list = [Counter(sms_tokenized) for sms_tokenized in preprocessed_documents]

# ## scikit-learn IMPLEMENTATION
# count_vector = CountVectorizer(stop_words='english')  # ENGLISH_STOP_WORDS.union())
# count_vector.fit(df['sms'])
# doc_array = count_vector.transform(df['sms']).toarray()
# frequency_matrix = pd.DataFrame(doc_array, columns=count_vector.get_feature_names())

X_train, X_test, y_train, y_test = train_test_split(df['sms'], df['label'], random_state=1)

count_vector = CountVectorizer()
training_data = count_vector.fit_transform(X_train)
testing_data = count_vector.transform(X_test)

## Example of Bayes Theorem
d = .01
not_d = 1-d
pos_given_d = .9
neg_given_not_d = .9

pos = (d * pos_given_d) + (not_d * (1-neg_given_not_d))
d_given_pos = d*pos_given_d / pos
neg = 1-pos
not_d_given_pos = not_d * (1-neg_given_not_d) / pos

## Naive Bayes Theorem
# f = saying "freedom", i = saying "immigration", e = saying "environment"
# j = Jill Stein
j = .5
f_given_j = .1
i_given_j = .1
e_given_j = .8
# g = Gary Johnson
g = .5
f_given_g = .7
i_given_g = .2
e_given_g = .1
# probs Jill/Gary will saying "freedom" and "immigration"
f = f_given_j*j + f_given_g*g
i = i_given_j*j + i_given_g*g
j_text = j * f_given_j * i_given_j
g_text = g * f_given_g * i_given_g
fi = j_text + g_text
j_given_fi = j_text / fi
g_given_fi = g_text / fi

## scikit-learn Naive Bayes
naive_bayes = MultinomialNB().fit(training_data, y_train)
predictions = naive_bayes.predict(testing_data)
print 'Accuracy score:', format(accuracy_score(y_test, predictions))
print 'Precision score:', format(precision_score(y_test, predictions))
print 'Recall score:', format(recall_score(y_test, predictions))
print 'F1 score:', format(f1_score(y_test, predictions))
