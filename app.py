import os
import sys
import inspect
import ast
import numpy as np

import libTHFakeNews as lTH
from libTHFakeNews import ThFN

from flask import Flask, render_template, redirect, url_for, request
 
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/results', methods = ['POST'])
def results():
    if request.method == 'POST':
        head_news = request.form['head_news']
        head_news_split = lTH.split_text(head_news)
        fn = ThFN(method='cv', \
            true_pre='ข่าวจริง', false_pre='ข่าวปลอม', unknown_pre='ข่าวที่ยังระบุไม่ได้', \
            save_vocab=False, load_vocab=True, save_model=False, load_model=True) 

        file_name = 'result/False_News_Words_List.txt'

        false_news_words_list = None

        if os.path.exists(file_name):
            f = open(file_name, 'r', encoding='utf-8')
            for value in f:
                false_news_words_list = ast.literal_eval(value)

        head_news_false_news_words = [word for word in head_news_split if word in false_news_words_list]

        text_print = None

        if len(head_news_false_news_words) == 0:
            head_news_false_news_words = '-'
            text_print = '-'
        else:
            text_print = []
            file_name = 'result/False_News_Words_Prob.txt'
            false_news_words_prob = None
            if os.path.exists(file_name):
                f = open(file_name, 'r', encoding='utf-8')
                for value in f:
                    false_news_words_prob = ast.literal_eval(value)
            false_news_words_prob = np.array(false_news_words_prob, dtype=object)
            for word in head_news_false_news_words:
                if word in false_news_words_prob[::, 0]:
                    index_location = np.where(false_news_words_prob[::, 0] == word)
                    text_print.append(str(false_news_words_prob[index_location[0], 0][0]) + ': ' + str(round(false_news_words_prob[index_location[0], 1][0][1], 2)))

        head_news_predict = fn.predict(x=[head_news_split])

        return render_template('results.html', \
            head_news=head_news, \
            head_news_split=head_news_split, \
            head_news_false_news_words=head_news_false_news_words, \
            text_print=text_print, \
            head_news_predict=head_news_predict
            )

# @app.route('/about')
# def about():
#     return render_template('about.html')

if __name__ == '__main__':
    app.run(debug = True)