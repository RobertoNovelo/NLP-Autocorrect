import collections, math

class LaplaceUnigramLanguageModel:

  def __init__(self, corpus):
    """Initialize your data structures in the constructor."""
    self.laplaceUnigramWordsCount = collections.defaultdict(lambda: 0)
    self.total = 0
    self.train(corpus)

  def train(self, corpus):
    """ Takes a corpus and trains your language model. 
        Compute any counts or other corpus statistics in this function.
    """  
    for sentence in corpus.corpus:
        for singleData in sentence.data:
            token = singleData.word
            self.laplaceUnigramWordsCount[token] = self.laplaceUnigramWordsCount[token] + 1
            self.total += 1

  def score(self, sentence):
    """ Takes a list of strings as argument and returns the log-probability of
    the sentence using your language model. Use whatever data you computed in
    train() here.  """

    score = 0
    for token in sentence:
        count = self.laplaceUnigramWordsCount[token]
        score += math.log(count + 1)
        score -= math.log(self.total + len(self.laplaceUnigramWordsCount))
    return score