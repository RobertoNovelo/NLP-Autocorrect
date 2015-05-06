import math, collections

class StupidBackoffLanguageModel:

  def __init__(self, corpus):
    """Initialize your data structures in the constructor."""
    self.trigramWordsCount = collections.defaultdict(lambda: 0)
    self.bigramWordsCount = collections.defaultdict(lambda: 0)
    self.unigramWordsCount = collections.defaultdict(lambda: 0)
    self.total = 0
    self.train(corpus)

  def train(self, corpus):
    """ Takes a corpus and trains your language model. 
        Compute any counts or other corpus statistics in this function.
    """  
    for sentence in corpus.corpus:
        dataUnits = sentence.data
        dataUnitsLength = len(dataUnits)
        for i in range(dataUnitsLength):
            self.total += 1
            thirdToken = dataUnits[i].word
            self.unigramWordsCount[thirdToken] += 1
            if i >= 1:
                secondToken = dataUnits[i - 1].word
                bigramKey = secondToken + "," + thirdToken
                self.bigramWordsCount[bigramKey] += 1
                if i >= 2:
                    firstToken = dataUnits[i - 2].word
                    trigramKey = firstToken + "," + secondToken + "," + thirdToken
                    self.trigramWordsCount[trigramKey] += 1

  def score(self, sentence):
    """ Takes a list of strings as argument and returns the log-probability of the 
        sentence using your language model. Use whatever data you computed in train() here.
    """
    score = 0
    unigramWordsCountLength = len(self.unigramWordsCount)
    sentencesLength = len(sentence)
    for i in range(sentencesLength):
        if i>=2:

            trigramKey = sentence[i - 2] + "," + sentence[i - 1] + "," + sentence[i]     
            trigramCount = self.trigramWordsCount[trigramKey]
            bigramKey = sentence[i - 2] + "," + sentence[i - 1]
            bigramCount = self.bigramWordsCount[bigramKey]
            unigramKey = sentence[i - 2]
            unigramCount = self.unigramWordsCount[unigramKey]

            if trigramCount > 0:
                score += math.log(trigramCount) + math.log(bigramCount) - math.log(unigramCount)

            else:
                bigramKey = sentence[i - 1] + "," + sentence[i]
                bigramCount = self.bigramWordsCount[bigramKey]
                unigramKey = sentence[i - 1]
                unigramCount = self.unigramWordsCount[unigramKey]
                if bigramCount > 0:
                    score += math.log(bigramCount) - math.log(unigramCount)

                else: 
                    unigramCount = self.unigramWordsCount[sentence[i]]
                    score += math.log(unigramCount + 1) + math.log(0.4)
                    score -= math.log(self.total + unigramWordsCountLength)

    return score
