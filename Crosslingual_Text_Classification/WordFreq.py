import nltk
from nltk.corpus import udhr

class WordFreq:
    """ Detect the type of language

    Detect the type of language based on comparing the total word frequency in different languages

    Attributes:
         languages: A list contains the type of language
         corpus: A dictionary contains the word and its language
         genre_word: a list of tuples. Each tuple contains a word and its language
         cfd: a ConditionalFreqDist class, inited with the genre word

    """
    def __init__(self):
        """

        Inits the WordFreq with different language corpus in nltk.udhr

        """
        self.languages = ['English','German_Deutsch']
        self.corpus = dict((language, udhr.words(language + '-Latin1')) for language in self.languages)
        self.genre_word = [(genre, word.lower()) for genre in self.languages for word in self.corpus[genre] if word.isalnum()]  # get all the word wih lower case
        self.cfd = nltk.ConditionalFreqDist(self.genre_word)


    def cal_score(self, char, language):
        """ calculate the conditional frequency of each word in a sentence

        Computing the score of sentence based on the sum of frequency of each token under the condition of a language.

        Arguments:
            char: n_token list, tokenized sentence
            language: string, desired language, under which condition the frequency will be calculated.
        Return:
            score sum: An integer, sum of the word frequency of each word under the condition of desired language
        """
        score = []
        for char_val in char:
            score1 = self.cfd[language].freq(char_val)
            score.append(score1)
            score_sum = sum(score)

        return score_sum

    def guess_language(self, text):
        """ detect the type of language

        detect the type of language based on comparing the score of a text in different language

        Arguments:
            text: n_token list, tokenized sentence

        Return:
             flag: A string, detected type of language of a text
        """
        score = {}
        for lang in self.languages:
            score[lang] = self.cal_score(text, lang)
        flag = max(score, key=score.get)  # the language with maximum score is the detected type of this text

        return flag
