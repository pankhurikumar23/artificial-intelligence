"""
Artificial Intelligence - Programming Homework 3
Implement a Spam Classifier for text messages, using
Naive Bayes Probabilities.

@author: PANKHURI KUMAR (PK2569)

# 1. Original Performance: k = 1, c = 1
# dev.txt: Precision:0.9508196721311475 Recall:0.8656716417910447 F-Score:0.9062499999999999 Accuracy:0.9784560143626571
# test.txt: Precision:0.9491525423728814 Recall:0.8888888888888888 F-Score:0.9180327868852458 Accuracy:0.9820466786355476

# 2. Modified Parameters, k = 1, c = 0.01
# dev.txt: Precision:0.9830508474576272 Recall:0.8656716417910447 F-Score:0.9206349206349207 Accuracy:0.9820466786355476
# test.txt: Precision:0.9649122807017544 Recall:0.873015873015873 F-Score:0.9166666666666667 Accuracy:0.9820466786355476

# 3. Modified Stopwords, k = 1, c = 0.12
# dev.txt: Precision:0.967741935483871 Recall:0.8955223880597015 F-Score:0.930232558139535 Accuracy:0.9838420107719928
# test.txt: Precision:0.95 Recall:0.9047619047619048 F-Score:0.9268292682926829 Accuracy:0.9838420107719928
"""

import sys
import string
import codecs
import math

#returns a list of words from 'text', converted to lowercase and without punctuation
def extract_words(text):
    lowerText = text.lower()
    lowerText.strip()
    t = str.maketrans("", "", string.punctuation)
    lowerText = lowerText.translate(t)
    # print(lowerText)
    words = lowerText.split()
    # print(words)
    return words


class NbClassifier(object):

    def __init__(self, training_filename, stopword_file = None):
        self.attribute_types = set()
        self.label_prior = {}    
        self.word_given_label = {}
        self.stopwords = set()
        self.full_vocab = {}

        if stopword_file != None:
            self.read_stop_words(stopword_file)
        self.collect_attribute_types(training_filename, 1)
        self.train(training_filename)

    #creating a list of stopwords and storing in self.stopwords
    def read_stop_words(self, stopword_file):
        f = codecs.open(stopword_file, 'r', 'UTF8')

        for line in f:
            line = line.strip()
            self.stopwords.add(line)

    #creating a list of atrributes from the training data set, to compute naive bayes probabilities
    def collect_attribute_types(self, training_filename, k):
        f = codecs.open(training_filename, 'r', 'UTF8')

        for line in f:
            label, text = line.split('\t', 1)
            preProcessed = extract_words(text)
            for word in preProcessed:
                if (word in self.full_vocab) and (word not in self.stopwords):
                    self.full_vocab[word] += 1
                elif (word not in self.full_vocab) and (word not in self.stopwords):
                    self.full_vocab[word] = 1

        f.close()

        #adding words with frequency above 'k' to attribute list
        for word in self.full_vocab:
            # print(word + ' ' + str(full_vocab[word]))
            if self.full_vocab[word] >= k:
                self.attribute_types.add(word)

    #computing naive bayes probabilities
    def train(self, training_filename):
        f = codecs.open(training_filename, 'r', 'UTF8')
        c = 0.12
        word_count = {}
        line_count = 0

        #initial preprocessing - counting only labels and words already in attribute list
        for line in f:
            line_count += 1
            label, text = line.split('\t', 1)
            preProcessed = extract_words(text)
            if label in self.label_prior:
                self.label_prior[label] += 1
            else:
                self.label_prior[label] = 1
            if label in word_count:
                word_count[label] += len(preProcessed)
            else:
                word_count[label] = len(preProcessed)
            for word in preProcessed:
                if word in self.attribute_types:
                    if (word, label) in self.word_given_label:
                        self.word_given_label[(word, label)] += 1
                    else:
                        self.word_given_label[(word, label)] = 1
        f.close()

        #calculating prior probability of ham/spam labels
        for label in self.label_prior:
            self.label_prior[label] = self.label_prior[label] / line_count

        #calculating the conditional probability of each attribute, given the label
        for word in self.attribute_types:
            for label in self.label_prior:
                if (word, label) in self.word_given_label:
                    self.word_given_label[(word, label)] = (self.word_given_label[(word, label)] + c) / (word_count[label] + c * len(self.attribute_types))
                else:
                    self.word_given_label[(word, label)] = c / (word_count[label] + c * len(self.attribute_types))

    #predicting label for a message, based on conditional probabilities of its attributes
    def predict(self, text):
        prediction = {}
        preProcessed = extract_words(text)

        for label in self.label_prior:
            prediction[label] = math.log(self.label_prior[label])
            for word in preProcessed:
                if word in self.attribute_types:
                    prediction[label] += math.log(self.word_given_label[(word,label)])

        return prediction

    #calculating accuracy of the spam classifier using true/false positives and true/false negatives
    def evaluate(self, test_filename):
        f = codecs.open(test_filename, 'r', 'UTF8')
        tp = 0
        tn = 0
        fp = 0
        fn = 0

        for line in f:
            label, text = line.split('\t', 1)
            prediction = self.predict(text)
            if prediction['ham'] < prediction['spam']:
                p_label = 'spam'
                if label == p_label:
                    tp += 1
                else:
                    fp += 1
            else:
                p_label = 'ham'
                if label == p_label:
                    tn += 1
                else:
                    fn += 1

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        fscore = (2*precision*recall)/(precision+recall)
        accuracy = (tp+tn)/(tp+tn+fp+fn)
        return precision, recall, fscore, accuracy


def print_result(result):
    print("Precision:{} Recall:{} F-Score:{} Accuracy:{}".format(*result))


if __name__ == "__main__":

    classifier = NbClassifier(sys.argv[1], sys.argv[3])
    result = classifier.evaluate(sys.argv[2])
    print_result(result)
