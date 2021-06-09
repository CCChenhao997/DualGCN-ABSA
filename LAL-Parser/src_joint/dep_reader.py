__author__ = 'max'

import KM_parser
tokens = KM_parser

class Sentence(object):
    def __init__(self, words, postags):
        self.words = words
        self.postags = postags

    def length(self):
        return len(self.words)


class DependencyInstance(object):
    def __init__(self, sentence, postags, heads, types):
        self.sentence = sentence
        self.postags = postags
        self.heads = heads
        self.types = types

    def length(self):
        return self.sentence.length()


class CoNLLXReader(object):
    def __init__(self, file_path, type_vocab = None):
        self.__source_file = open(file_path, 'r')
        self.type_vocab = type_vocab

    def close(self):
        self.__source_file.close()

    def getNext(self):
        line = self.__source_file.readline()
        # skip multiple blank lines.
        while len(line) > 0 and len(line.strip()) == 0:
            line = self.__source_file.readline()
        if len(line) == 0:
            return None

        lines = []
        while len(line.strip()) > 0:
            line = line.strip()
            #line = line.decode('utf-8')
            lines.append(line.split('\t'))
            line = self.__source_file.readline()

        length = len(lines)
        if length == 0:
            return None

        words = []
        postags = []
        types = []
        heads = []
        gold_pos = []

        # words.append(KM_parser.ROOT)
        # postags.append(KM_parser.ROOT)
        # types.append(KM_parser.ROOT)
        # heads.append(0)

        for tokens in lines:

            word = tokens[1]
            pos = tokens[4]
            gold_pos.append(tokens[3])
            head = int(tokens[6])
            type = tokens[7]

            words.append(word)

            postags.append(pos)

            types.append(type)

            heads.append(head)

        # words.append(parse_nk.STOP)
        # postags.append(parse_nk.STOP)
        # types.append(parse_nk.STOP)
        # heads.append(0)

        return DependencyInstance(Sentence(words, postags), gold_pos, heads, types)