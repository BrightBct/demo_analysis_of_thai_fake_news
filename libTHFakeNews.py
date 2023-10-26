import ast
import math
import os
import random
import re

import numpy as np
import pythainlp.corpus
import pythainlp.tag
import pythainlp.util

random.seed(1)


def remove_special_character(text_array: np.ndarray) -> list:
    pattern = re.compile(r"[^\u0E00-\u0E7Fa-zA-Z' ]|^'|'$|''")
    array = []
    
    for text in text_array:
        char_to_remove = re.findall(pattern, text)
        list_with_char_removed = [char for char in text if char not in char_to_remove]
        array.append(''.join(list_with_char_removed))
        
    return array


def remove_excess_spaces(text_array: np.ndarray) -> np.ndarray: 
    emoji_pattern = re.compile(pattern="[" u"\U0001F600-\U0001F64F" "]+", flags=re.UNICODE)
    array = []
    for row in range(len(text_array)):
        sub_array = []
        for col in range(len(text_array[0])):
            text_remove_emoji = emoji_pattern.sub(r'', text_array[row][col])
            sub_array.append(' '.join(text_remove_emoji.split()))
        array.append(sub_array)
    return np.array(array, dtype=object)


def split_text(text: str, method='pythainlp') -> list:
    return np.array(pythainlp.word_tokenize(text, engine='newmm', keep_whitespace=False), dtype=object)

def remove_stop_words(text_array: np.ndarray) -> list:
    array = []
    
    for text in text_array:
        word_list = []
        for word in text:
            if word not in list(pythainlp.corpus.thai_stopwords()):
                word_list.append(word)
        array.append(word_list)
    return np.array(array, dtype=object)


class ThFN:
    
    
    def __init__(self, method: str, \
        true_pre='ข่าวจริง', false_pre='ข่าวปลอม', unknown_pre='ข่าวที่ยังระบุไม่ได้', \
        save_vocab=False, load_vocab=False, \
        save_model=False, load_model=False):
        
        self._method = method
        self._true_pre = true_pre
        self._false_pre = false_pre
        self._unknown_pre = unknown_pre
        self._save_vocab = save_vocab
        self._load_vocab = load_vocab
        self._save_model = save_model
        self._load_model = load_model

        # Count Vector & TF-IDF Vector
        self._vocabulary = None
        self._count_of_word = None
        self._dict_of_word = None
        self._vector = None

        # Naive Bayes
        self._n_news = None
        self._n_true_news = None
        self._n_false_news = None
        
        self._n_news_words = None
        self._n_true_news_words = None
        self._n_false_news_words = None
        
        self._dictionary_news_true = None
        self._dictionary_news_false = None
        self._prob_news_true = None
        self._prob_news_false = None
        
        # etc
        self._prob_dict_of_word = None
        self._list_of_true_news_word = None
        self._list_of_false_news_word = None
        self._true_news_word_prob = None
        self._false_news_word_prob = None

    # Count Vector & TF-IDF Vector
    def _fit_vocab(self, data):
        self._count_of_word = 0
        self._dict_of_word = dict()
        self._vector = []
        
        for row in range(len(data)):
            for word in data[row]:
                if word not in self._dict_of_word:
                    self._dict_of_word[word] = self._count_of_word
                    self._count_of_word += 1
        self._vocabulary = list(self._dict_of_word.keys())
        if self._save_vocab:
            self._save_vocab_def()

    def _vocab_transform(self, data):

        if self._load_vocab:
            self._load_vocab_def()
        
        if type(data).__module__ != np.__name__ and type(data) is not list:
            x = np.array([[data]], dtype=object)
        else:
            x = data.copy()
        
        vector = []
        for row in range(len(x)):
            word_array = [0 for _ in range(self._count_of_word)]
            for word in x[row]:
                if word in self._dict_of_word:
                    word_array[self._dict_of_word[word]] += 1
            vector.append(word_array)

        if self._method == 'cv':
            self._vector = vector
            return self._vector

        elif self._method == 'tf-idf':
            col_cal = []
            for col in zip(*vector):
                count_col = 0
                for row in col:
                    if row != 0:
                        count_col += 1
                col_cal.append(count_col)
            tfidf = []
            for row in range(len(vector)):
                cal_word = []
                for col in range(len(vector[row])):
                    cal_word.append(int(math.ceil(vector[row][col] * (math.log((1 + len(vector)) / (1 + col_cal[col])) + 1))))
                tfidf.append(cal_word)
            self._vector = tfidf
            return tfidf

    def _save_vocab_def(self):
        f = open('Txt_File/Pre_Vectorizer.txt', 'w+', encoding='utf-8')

        # _count_of_word
        f.write(str(self._count_of_word) + '\n')
        # _vocabulary
        f.write(str(self._vocabulary) + '\n')
        # _dict_of_word
        f.write(str(self._dict_of_word))
        f.close()

    def _load_vocab_def(self):
        file_name = 'Txt_File/Pre_Vectorizer.txt'
        if os.path.exists(file_name):
            f = open(file_name, 'r', encoding='utf-8')
            contents = []
            for content in f:
                contents.append(content.replace('\n', ''))

            self._count_of_word = int(contents[0])
            self._vocabulary = ast.literal_eval(contents[1])
            self._dict_of_word = ast.literal_eval(contents[2])

    # Naive Bayes
    def fit(self, x, y):

        y_train = y.copy()
        self._fit_vocab(x)

        x_train = self._vocab_transform(x)

        self._n_news = len(x_train)
        self._n_true_news = 0
        self._n_false_news = 0

        self._n_true_news_words = 0
        self._n_false_news_words = 0

        for row in range(len(y_train)):
            if y_train[row] == self._true_pre:
                self._n_true_news_words += sum(x_train[row])
                self._n_true_news += 1
            elif y_train[row] == self._false_pre:
                self._n_false_news_words += sum(x_train[row])
                self._n_false_news += 1

        true_news = [x_train[row] for row in range(len(x_train)) if y_train[row] == self._true_pre]
        false_news = [x_train[row] for row in range(len(x_train)) if y_train[row] == self._false_pre]

        # col_list = [col for col in range(len(x_train[0]))]
        # self._dictionary_news_true = dict(zip(col_list, np.sum(np.array(true_news), axis=0)))
        # self._dictionary_news_false = dict(zip(col_list, np.sum(np.array(false_news), axis=0)))

        self._dictionary_news_true = {k: v for k, v in enumerate(np.sum(np.array(true_news), axis=0))}
        self._dictionary_news_false = {k: v for k, v in enumerate(np.sum(np.array(false_news), axis=0))}

        # self._vector = x_train
        self._n_news_words = len(x_train[0])
        self._prob_news_true = self._n_true_news / len(y_train)
        self._prob_news_false = self._n_false_news / len(y_train)

        if self._save_model:
            self._save_model_def()

    def predict(self, x, threshold=0.5):

        if self._load_model:
            self._load_model_def()

        x_test = self._vocab_transform(x)

        news_predict = []
        divisor_true = self._n_true_news_words + self._n_news_words
        divisor_false = self._n_false_news_words + self._n_news_words
        for row in range(len(x_test)):
            predict_true = self._prob_news_true
            predict_false = self._prob_news_false
            for col in range(len(x_test[row])):
                if x_test[row][col] == 0:
                    continue
                predict_true *= (self._dictionary_news_true[col] + 1) / divisor_true
                predict_false *= (self._dictionary_news_false[col] + 1) / divisor_false

            news_predict_false = predict_false / (predict_true + predict_false)

            if news_predict_false > threshold:
                news_predict.append(self._false_pre)
            else:
                news_predict.append(self._true_pre)

        return news_predict

    def predict_proba(self, x):

        if self._load_model:
            self._load_model_def()

        x_test = self._vocab_transform(x)

        news_predict = []
        divisor_true = self._n_true_news_words + self._n_news_words
        divisor_false = self._n_false_news_words + self._n_news_words
        for row in range(len(x_test)):
            predict_true = self._prob_news_true
            predict_false = self._prob_news_false
            for col in range(len(x_test[row])):
                if x_test[row][col] == 0:
                    continue
                predict_true *= (self._dictionary_news_true[col] + 1) / divisor_true
                predict_false *= (self._dictionary_news_false[col] + 1) / divisor_false

            news_predict_true = predict_true / (predict_true + predict_false)
            news_predict_false = predict_false / (predict_true + predict_false)

            news_predict.append([news_predict_true, news_predict_false])

        return news_predict

    def predict_plus_unknown(self, x, threshold=0.5):

        if self._load_model:
            self._load_model_def()

        x_test = self._vocab_transform(x)

        news_predict = []
        divisor_true = self._n_true_news_words + self._n_news_words
        divisor_false = self._n_false_news_words + self._n_news_words
        for row in range(len(x_test)):
            predict_true = self._prob_news_true
            predict_false = self._prob_news_false
            for col in range(len(x_test[row])):
                if x_test[row][col] == 0:
                    continue
                predict_true *= (self._dictionary_news_true[col] + 1) / divisor_true
                predict_false *= (self._dictionary_news_false[col] + 1) / divisor_false

            news_predict_false = predict_false / (predict_true + predict_false)

            if threshold - 0.1 <= news_predict_false <= threshold + 0.1:
                news_predict.append(self._unknown_pre)
            else:
                if news_predict_false > threshold:
                    news_predict.append(self._false_pre)
                else:
                    news_predict.append(self._true_pre)

        return news_predict

    def _save_model_def(self):

        f = None

        if self._method == 'cv':
            f = open('Txt_File/Count_Vectorizer.txt', 'w+', encoding='utf-8')
        elif self._method == 'tf-idf':
            f = open('Txt_File/TF_IDF_Vectorizer.txt', 'w+', encoding='utf-8')

        f.write(str(self._n_true_news_words) + '\n')
        f.write(str(self._n_false_news_words) + '\n')
        f.write(str(self._dictionary_news_true) + '\n')
        f.write(str(self._dictionary_news_false) + '\n')
        f.write(str(self._vector) + '\n')
        f.write(str(self._n_news_words) + '\n')
        f.write(str(self._prob_news_true) + '\n')
        f.write(str(self._prob_news_false))
        f.close()

    def _load_model_def(self):
        file_name = None
        if self._method == 'cv':
            file_name = 'Txt_File/Count_Vectorizer.txt'
        elif self._method == 'cv':
            file_name = 'Txt_File/TF_IDF_Vectorizer.txt'

        if os.path.exists(file_name):
            f = open(file_name, 'r', encoding='utf-8')

            contents = []
            for content in f:
                contents.append(content.replace('\n', ''))

            self._n_true_news_words = int(contents[0])
            self._n_false_news_words = int(contents[1])
            self._dictionary_news_true = ast.literal_eval(contents[2])
            self._dictionary_news_false = ast.literal_eval(contents[3])
            self._vector = contents[4]
            self._n_news_words = int(contents[5])
            self._prob_news_true = float(contents[6])
            self._prob_news_false = float(contents[7])

    # Naive Bayes Findings
    def sentences_probability(self, x):

        if self._load_model:
            self._load_model_def()

        x_test = self._vocab_transform(x)

        news_predict = []
        divisor_true = self._n_true_news_words + self._n_news_words
        divisor_false = self._n_false_news_words + self._n_news_words

        for row in range(len(x_test)):
            predict_true = self._prob_news_true
            predict_false = self._prob_news_false
            for col in range(len(x_test[row])):
                if x_test[row][col] == 0:
                    continue
                predict_true *= (self._dictionary_news_true[col] + 1) / divisor_true
                predict_false *= (self._dictionary_news_false[col] + 1) / divisor_false

            news_predict_true = predict_true / (predict_true + predict_false)
            news_predict_false = predict_false / (predict_true + predict_false)

            news_predict.append([news_predict_true, news_predict_false])

        return news_predict

    def words_in_sentence_probability(self, x):

        if self._load_model:
            self._load_model_def()

        x_test = self._vocab_transform(x)

        news_predict = []
        divisor_true = self._n_true_news_words + self._n_news_words
        divisor_false = self._n_false_news_words + self._n_news_words
        for row in range(len(x_test)):
            words_predict = []
            for col in range(len(x_test[row])):
                if x_test[row][col] == 0:
                    continue
                words_predict.append([(self._dictionary_news_true[col] + 1) / divisor_true,
                                      (self._dictionary_news_false[col] + 1) / divisor_false])
            news_predict.append(words_predict)

        return news_predict

    def words_in_news_probability(self, x):

        if self._load_model:
            self._load_model_def()
        
        x_test = self._vocab_transform(x)

        news_predict = []
        divisor_true = self._n_true_news_words + self._n_news_words
        divisor_false = self._n_false_news_words + self._n_news_words
        for row in range(len(x_test)):
            words_predict = []
            for col in range(len(x_test[row])):
                if x_test[row][col] == 0:
                    continue
                predict_true = self._prob_news_true * ((self._dictionary_news_true[col] + 1) / divisor_true)
                predict_false = self._prob_news_false * ((self._dictionary_news_false[col] + 1) / divisor_false)
                words_predict.append([predict_true / (predict_true + predict_false),
                                      predict_false / (predict_true + predict_false)])
            news_predict.append(words_predict)

        return news_predict

    def words_in_news_probability_2(self, x):

        if self._load_model:
            self._load_model_def()
        
        x_test = self._vocab_transform(x)

        news_predict = []
        divisor_true = self._n_true_news_words + self._n_news_words
        divisor_false = self._n_false_news_words + self._n_news_words
        for row in range(len(x_test)):
            words_predict = []
            for col in range(len(x_test[row])):
                if x_test[row][col] == 0:
                    continue
                predict_true = self._prob_news_true * ((self._dictionary_news_true[col] + 1) / divisor_true)
                predict_false = self._prob_news_false * ((self._dictionary_news_false[col] + 1) / divisor_false)
                words_predict.append([predict_true / (predict_true + predict_false),
                                      predict_false / (predict_true + predict_false)])
            news_predict.append(words_predict)

        return news_predict

    def words_plus_unknown_in_sentences(self, x):

        if self._load_model:
            self._load_model_def()

        if type(x).__module__ != np.__name__:
            x_test = self._vocab_transform(np.array(x))
        else:
            x_test = self._vocab_transform(x.copy())
        
        news_predict = []
        divisor_true = self._n_true_news_words + self._n_news_words
        divisor_false = self._n_false_news_words + self._n_news_words
        for row in range(len(x_test)):
            words_predict = []
            for col in range(len(x_test[row])):
                if x_test[row][col] == 0:
                    continue
                predict_true = self._prob_news_true * ((self._dictionary_news_true[col] + 1) / divisor_true)
                predict_false = self._prob_news_false * ((self._dictionary_news_false[col] + 1) / divisor_false)

                predict_false = predict_false / (predict_true + predict_false)
                
                if 0.4 <= predict_false <= 0.6:
                    continue
                else:
                    words_predict.append(self._vocabulary[col])
                        
            news_predict.append(words_predict)

        return news_predict

    # etc.
    def _calculate_etc(self):

        if self._dict_of_word is not None:
            self._prob_dict_of_word = self._dict_of_word.copy()
            self._prob_dict_of_word = dict.fromkeys(self._prob_dict_of_word, list())
            for key in self._prob_dict_of_word.keys():
                # self._prob_dict_of_word[key] = self._probability(key)
                self._prob_dict_of_word[key] = self.words_in_news_probability(key)[0][0]

            # if value[0] == value[1], it will not appear in set true word or set false word
            self._list_of_true_news_word = [key for key, value in
                                            zip(self._prob_dict_of_word.keys(), self._prob_dict_of_word.values())
                                            if value[0] > value[1]]
            self._list_of_false_news_word = [key for key, value in
                                             zip(self._prob_dict_of_word.keys(), self._prob_dict_of_word.values())
                                             if value[1] > value[0]]
            self._true_news_word_prob = [[key, value] for key, value in
                                            zip(self._prob_dict_of_word.keys(), self._prob_dict_of_word.values())
                                            if value[0] > value[1]]
            self._false_news_word_prob = [[key, value] for key, value in
                                             zip(self._prob_dict_of_word.keys(), self._prob_dict_of_word.values())
                                             if value[1] > value[0]]

    def y_transform(self, y):
        return [1 if value == self._false_pre else 0 for value in y]

    # Get function
    def get_dict_of_word(self) -> dict:
        return self._dict_of_word
    
    def get_vector(self) -> list:
        return self._vector
    
    def get_vocab(self) -> list:
        return self._vocabulary
    
    def get_true_news_word(self, save_result=False):
        if self._list_of_true_news_word is None:
            self._calculate_etc()
        if save_result:
            f = open('Result/True_News_Words_List.txt', 'w+', encoding='utf-8')
            f.write(str(self._list_of_true_news_word))
            f.close()
        return self._list_of_true_news_word

    def get_false_news_word(self, save_result=False):
        if self._list_of_false_news_word is None:
            self._calculate_etc()
        if save_result:
            f = open('Result/False_News_Words_List.txt', 'w+', encoding='utf-8')
            f.write(str(self._list_of_false_news_word))
            f.close()
        return self._list_of_false_news_word

    def get_true_news_word_prob(self, save_result=False):
        if self._true_news_word_prob is None:
            self._calculate_etc()
        if save_result:
            f = open('Result/True_News_Words_Prob.txt', 'w+', encoding='utf-8')
            f.write(str(self._true_news_word_prob))
            f.close()
        return self._true_news_word_prob

    def get_false_news_word_prob(self, save_result=False):
        if self._false_news_word_prob is None:
            self._calculate_etc()
        if save_result:
            f = open('Result/False_News_Words_Prob.txt', 'w+', encoding='utf-8')
            f.write(str(self._false_news_word_prob))
            f.close()
        return self._false_news_word_prob

    def get_count_of_words(self, data, data_check = None):
        
        dict_word = dict()
        if isinstance(data, list):
            for word in data:
                if data_check is not None:
                    if word in self._dict_of_word and word in data_check:
                        if word not in dict_word:
                            dict_word[word] = 1
                        else:
                            dict_word[word] += 1
                else:
                    if word in self._dict_of_word:
                        if word not in dict_word:
                            dict_word[word] = 1
                        else:
                            dict_word[word] += 1
        elif type(data).__module__ == np.__name__:
            for row in range(len(data)):
                for word in data[row]:
                    if data_check is not None:
                        if word in self._dict_of_word and word in data_check:
                            if word not in dict_word:
                                dict_word[word] = 1
                            else:
                                dict_word[word] += 1
                    else:
                        if word in self._dict_of_word:
                            if word not in dict_word:
                                dict_word[word] = 1
                            else:
                                dict_word[word] += 1
        return dict_word

    def get_words_probs(self, data):
        return self.words_in_news_probability(data)

    def get_vocab_transform(self, x):
        return self._vocab_transform(x)


class EvaluateModel:

    def __init__(self, true_pre='ข่าวจริง', false_pre='ข่าวปลอม', unknown_pre=None):
        self._label = 2
        self._true_pre = true_pre
        self._false_pre = false_pre
        if unknown_pre is not None:
            self._unknown_pre = unknown_pre
            self._label = 3

    def _label_class_to_number(self, y_pred, y_test):
        num_true_pre = 0
        num_false_pre = 1
        num_unknown_pre = 2

        y_pred_num = []
        y_test_num = []
        
        if self._label == 2:
            for y1, y2 in zip(y_pred, y_test):
                if y1 == self._true_pre:
                    y_pred_num.append(num_true_pre)
                elif y1 == self._false_pre:
                    y_pred_num.append(num_false_pre)
                else:
                    continue

                if y2 == self._true_pre:
                    y_test_num.append(num_true_pre)
                elif y2 == self._false_pre:
                    y_test_num.append(num_false_pre)

        elif self._label == 3:
            for y in y_pred:
                if y == self._true_pre:
                    y_pred_num.append(num_true_pre)
                elif y == self._false_pre:
                    y_pred_num.append(num_false_pre)
                else:
                    y_pred_num.append(num_unknown_pre)
            for y in y_test:
                if y == self._true_pre:
                    y_test_num.append(num_true_pre)
                elif y == self._false_pre:
                    y_test_num.append(num_false_pre)
                else:
                    y_test_num.append(num_unknown_pre)

        return y_pred_num, y_test_num

    def k_fold_cross_validation(self, model, x, y, folds=10, fun_method='normal', threshold=0.5):

        from sklearn.metrics import (accuracy_score, confusion_matrix,
                                     f1_score, precision_score, recall_score)
        from sklearn.model_selection import StratifiedKFold

        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1)

        acc = []
        precision = []
        recall = []
        f1 = []

        tpr = []
        tnr = []
        fpr = []
        fnr = []

        for train_index, test_index in cv.split(x, y):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            model.fit(x_train, y_train, False, False)

            y_pred = None
            if fun_method == 'normal':
                y_pred = model.predict(x_test, threshold)
            elif fun_method == 'plus':
                y_pred = model.predict_plus_unknown(x_test, threshold)

            y_pred_num, y_test_num = self._label_class_to_number(y_pred=y_pred, y_test=y_test)

            acc.append(accuracy_score(y_pred_num, y_test_num))
            precision.append(precision_score(y_pred_num, y_test_num))
            recall.append(recall_score(y_pred_num, y_test_num))
            f1.append(f1_score(y_pred_num, y_test_num))

            tn, fp, fn, tp = confusion_matrix(y_pred_num, y_test_num).ravel()

            # Sensitivity, hit rate, recall, or true positive rate
            tpr.append(tp/(tp+fn))
            # Specificity or true negative rate
            tnr.append(tn/(tn+fp))
            # Fall out or false positive rate
            fpr.append(fp/(fp+tn))
            # False negative rate
            fnr.append(fn/(tp+fn))

        return np.array([np.array(acc), 
                            np.array(precision), 
                            np.array(recall), 
                            np.array(f1), 
                            np.array(tpr), 
                            np.array(tnr), 
                            np.array(fpr), 
                            np.array(fnr)])

    def k_fold_cross_validation_roc_auc_threshold(self, model, x, y, folds=10, threshold=0.5, method_mean=True):

        from sklearn.metrics import auc, roc_curve
        from sklearn.model_selection import StratifiedKFold

        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1)

        thresholds = []
        roc_auc_array = []

        for train_index, test_index in cv.split(x, y):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            model.fit(x_train, y_train, False, False)

            y_proba = model.predict_proba(x_test)
            y_proba = np.array(y_proba)[:, 1]

            y_test_transform_np = np.array(model.y_transform(y_test))

            fpr, tpr, threshold = roc_curve(y_test_transform_np, y_proba)

            roc_auc = auc(fpr, tpr)
            roc_auc_array.append(roc_auc)

            max_index = np.where(tpr - fpr == np.amax(tpr - fpr))

            if method_mean:
               thresholds.append(np.mean(threshold[max_index]))
            else:
                if len(threshold[max_index]) > 1:
                    for threshold_value in threshold[max_index]:
                        thresholds.append(threshold_value)
                elif len(threshold[max_index]) == 1:
                    thresholds.append(threshold[max_index][0])

        return np.array([np.mean(roc_auc_array), np.std(roc_auc_array), 
                            np.mean(thresholds), np.std(thresholds)])
