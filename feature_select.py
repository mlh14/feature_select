#coding:utf-8

import re
from collections import Counter, defaultdict
import math
import sys
import cPickle
import os

def running_time(func):
    import datetime
    import functools
    @functools.wraps(func)
    def wrapper(*args, **kw):
        start_time = datetime.datetime.now()
        ret = func(*args, **kw)
        end_time = datetime.datetime.now()
        print '[%s()] done, run time : %r sec' % (func.__name__, (end_time - start_time).seconds)
        return ret
    return wrapper

class TextPreprocess(object):
    """
    pre-process raw text
    """
    hanzi_pattern = re.compile(u'[\u4E00-\u9FA5]')

    def __init__(self, vocab=None, cwf=None, nsc=None):
        self.vocab = vocab
        self.class_word_freq = cwf
        self.n_sample_of_classes = nsc

    @classmethod
    @running_time
    def from_text(cls, fname):
        separate_pattern = re.compile(ur'\s+')
        # word and frequency
        vocab = Counter()
        # word freq in each class
        class_word_freq = defaultdict(Counter)
        # n_sample of each class
        n_sample_of_classes = Counter()

        for line in file(fname):
            line = line.strip().decode('utf-8')
            line = separate_pattern.split(line)
            if not line or len(line) < 2:
                continue
            label, words = int(line[0]), line[1:]
            words = [word.strip() for word in words if cls.hanzi_pattern.match(word)]
            words = list(set(words))
            vocab.update(words)
            class_word_freq[label].update(words)
            n_sample_of_classes.update([label])

        print 'vocab size: ', len(vocab)
        return cls(vocab=vocab, cwf=class_word_freq, nsc=n_sample_of_classes)

    def save(self, dest_file):
        if not os.path.exists(dest_file):
            print "{0} doesn't exist !!!".format(dest_file)
            sys.exit(1)
        cPickle.dump(self, open(dest_file + '.pickle', 'wb'))

    @classmethod
    def load(cls, src_file):
        if not src_file.endswith('.pickle'):
            src_file += '.pickle'
        if not os.path.exists(src_file):
            print "{0} doesn't exist !!!".format(src_file)
            sys.exit(1)

        self = cPickle.load(open(src_file, 'rb'))
        return self

class FeatureGenerator(object):
    """docstring for FeatureGenerator"""
    def __init__(self, fname, option='-chi 1 -pre 0'):
        b_chi, b_pre = self.parse_option(option)

        if b_pre:
            self.text_prep = TextPreprocess().from_text(fname)
            self.text_prep.save(fname)
        else:
            self.text_prep = TextPreprocess().load(fname)

        self.n_class   = len(self.text_prep.class_word_freq.keys())
        self.class_word_sum, self.word_total, self.N = self._prepare()
        self.chi_square_final = defaultdict(Counter)

        if not b_chi:
            self.select()
        else:
            self.chi_square_select()

    def parse_option(self, option):
        b_chi, b_pre = False, False
        option = option.strip().split()
        i = 0
        while i < len(option):
            if option[i][0] != '-': break
            if option[i] == '-chi':
                if int(option[i+1]) != 0:
                    b_chi = True
            elif option[i] == '-pre':
                if int(option[i+1]) != 0:
                    b_pre = True
            i+=2

        return b_chi, b_pre

    def _prepare(self):
        class_word_sum = {label:sum(self.text_prep.class_word_freq[label].values())
        for label in self.text_prep.class_word_freq}

        word_total = float(sum(self.text_prep.vocab.values()))

        N = sum(self.text_prep.n_sample_of_classes.values())
        return class_word_sum, word_total, N

    def chi_square_compute(self, word):
        for label in self.text_prep.class_word_freq.iterkeys():
            A = self.text_prep.class_word_freq[label][word]
            B = self.text_prep.vocab[word] - A
            C = self.text_prep.n_sample_of_classes[label] - A
            D = self.N - A - B - C
            chi_square = math.log(float((A * D - B * C) ** 2) / ((A + B) * (C + D)))
            self.chi_square_final[label].update({word:chi_square})

    @running_time
    def chi_square_select(self):
        for word in self.text_prep.vocab.iterkeys():
            self.chi_square_compute(word)
        with open('chi_square', 'w') as fo:
            for label in self.chi_square_final:
                fo.write('*************[%s]**************\n' % label)
                for word, value in self.chi_square_final[label].most_common(100):
                    fo.write('{0}\t{1}\n'.format(value, word.encode('utf-8')))

    def info_gain_compute(self, word):
        N = float(self.N)
        # define some alais
        vocab = self.text_prep.vocab
        class_word_freq = self.text_prep.class_word_freq

        func_log = lambda x, y: (float(x) / (y + 1)) * math.log(float(x) / (y + 1))
        Entropy = - sum(map(func_log,
            self.text_prep.n_sample_of_classes.values(),
            [N] * self.n_class))

        appeard_cnt = sum([class_word_freq[label][word] for label in class_word_freq])
        P_t = appeard_cnt / N
        P_non_t = 1 - P_t

        Sum_P_c_t = - sum(map(func_log,
            [class_word_freq[label].get(word, 1) for label in class_word_freq],
            [appeard_cnt] * self.n_class
            ))
        Sum_P_c_non_t = - sum(map(func_log,
            [self.class_word_sum[label] - class_word_freq[label][word] for label in self.class_word_sum],
            [N - appeard_cnt] * self.n_class
            ))

        Conditional_Entropy = P_t * Sum_P_c_t + P_non_t * Sum_P_c_non_t
        IG = Entropy - Conditional_Entropy
        return IG * 1000

    @running_time
    def select(self):
        word_weight_dic = Counter()
        for word in self.text_prep.vocab.iterkeys():
            IG = self.info_gain_compute(word)
            word_weight_dic[word] = IG
        word_weight_dic = (sorted(word_weight_dic.iteritems(),
            key=lambda x:x[1], reverse=True))
        with open('IG', 'w') as fo:
            for word, weight in word_weight_dic:
                fo.write('{0}\t{1}\n'.format(word.encode('utf-8'), weight))


fg = FeatureGenerator('merged', option='-pre 0')




