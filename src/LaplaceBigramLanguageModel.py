import math, collections

class LaplaceBigramLanguageModel:

  def __init__(self, corpus):
    """Initialize your data structures in the constructor."""
    self.bigramWordsCount = collections.defaultdict(lambda: 0)
    self.unigramWordsCount = collections.defaultdict(lambda: 0)
    self.train(corpus)

  def train(self, corpus):
    """ Takes a corpus and trains your language model. 
        Compute any counts or other corpus statistics in this function.
    """  
    for sentence in corpus.corpus:
        dataUnits = sentence.data
        for index in range(len(dataUnits)):
            secondBigramToken = dataUnits[index].word
            self.unigramWordsCount[secondBigramToken] += 1
            if index > 0:
                firstBigramToken = dataUnits[index - 1].word
                bigramKey = firstBigramToken + "," + secondBigramToken
                self.bigramWordsCount[bigramKey] += 1

  def score(self, sentence):
    """ Takes a list of strings as argument and returns the log-probability of the 
        sentence using your language model. Use whatever data you computed in train() here.
    """
    score = 0
    v = 0
    for i in range(len(sentence)):
        if i > 0:
            bigramKey = sentence[i - 1] + "," + sentence[i]
            bigramCount = self.bigramWordsCount[bigramKey]
            unigramCount = self.unigramWordsCount[sentence[i-1]]
            score += math.log(bigramCount + 1)
            score -= math.log(unigramCount + len(self.bigramWordsCount))
    return score
