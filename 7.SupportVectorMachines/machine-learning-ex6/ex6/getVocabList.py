
def getVocabList():
    '''
    vocabList = GETVOCABLIST() reads the tab separated vocabulary list 
    in vocab.txt and returns a dictionary with string keys and 
    corresponding number as values
    '''
    import pandas as pd

    ## Read the fixed vocabulary list
    vocabList = pd.read_csv('vocab.txt', sep='\t', header=None)
    keys = vocabList[1]
    values = vocabList[0]
    vocabList = vocabList = dict(zip(keys, values))

    return vocabList