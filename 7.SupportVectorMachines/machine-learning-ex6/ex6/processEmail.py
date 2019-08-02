# ROCESSEMAIL preprocesses a the body of an email and
# returns a list of word_indices 

def processEmail(email_contents):
    '''
    word_indices = PROCESSEMAIL(email_contents) preprocesses 
    the body of an email and returns a list of indices of the 
    words contained in the email. 
    '''

    from getVocabList import getVocabList
    import re
    from nltk.stem.porter import PorterStemmer
    
    # Load Vocabulary
    vocabList = getVocabList()

    # ========================== Preprocess Email ===========================

    # Lower case
    email_contents = email_contents.lower()

    # Strip all HTML
    # Looks for any expression that starts with < and ends with > and 
    # does not have any < or > in the tag and replace it with a space
    email_contents = re.sub('<[^<>]+>', ' ', email_contents)

    # Handle Numbers
    # Look for one or more characters between 0-9
    email_contents = re.sub('[0-9]+', 'number', email_contents)

    # Handle URLS
    # Look for strings starting with http:// or https://
    email_contents = re.sub(r'(http|https)://[^\s]+', 'httpaddr', email_contents)
    
    # Handle Email Addresses
    # Look for strings with @ in the middle
    email_contents = re.sub(r'[^\s]+@[^\s]+', 'emailaddr', email_contents)

    # Handle $ sign
    email_contents = re.sub('[$]+', 'dollar', email_contents)

    # ========================== Tokenize Email ===========================

    # Tokenize and also get rid of any punctuation (any non alphanumeric characters)
    token_str = re.split(r'[\s]', email_contents)
    token_str = [re.sub('[^a-zA-Z0-9]', '', l) for l in token_str] # list comprehension

    # Remove empty strings from the list
    token_str = list(filter(None, token_str))

    # Output the email to screen as well
    print('\n==== Processed Email ====\n')
    print(token_str, '\n')

    # Stem the word using the Porter Stemming algorithm
    porter_stemmer = PorterStemmer()

    word_stem = []

    for word in token_str:
        word_stem.append(porter_stemmer.stem(word))
    
    # Look up the word in the dictionary and add the index
    # to word_indices if found

    word_indices = []

    for word in word_stem:
        if vocabList.get(word): # if it exists
            word_indices.append(vocabList.get(word))

    return word_indices