# Machine Learning Online Class
#  Exercise 6 | Spam Classification with SVMs

from readFile import *
from processEmail import *
from emailFeatures import *
from scipy.io import loadmat
from svmTrain import *
import numpy as np
from getVocabList import *

# ==================== Part 1: Email Preprocessing ====================
#  To use an SVM to classify emails into Spam v.s. Non-Spam, you first need
#  to convert each email into a vector of features. In this part, you will
#  implement the preprocessing steps for each email. You should
#  complete the code in processEmail.m to produce a word indices vector
#  for a given email.

print('\nPreprocessing sample email (emailSample1.txt)\n')

# Read file
file_contents = readFile('emailSample1.txt')

word_indices = processEmail(file_contents)

# Print Stats
print('==== Word Indices ==== \n')
print(word_indices, '\n')

input('Program paused. Press enter to continue.\n')

# ==================== Part 2: Feature Extraction ====================
#  Now, you will convert each email into a vector of features in R^n.

print('\nExtracting features from sample email (emailSample1.txt)\n')

# Extract Features
features = emailFeatures(word_indices)

# Print Stats
print('Length of feature vector: %d\n' % (len(features)))
print('Number of non-zero entries: %d\n' % sum(features > 0))

input('Program paused. Press enter to continue.\n')

# =========== Part 3: Train Linear SVM for Spam Classification ========
#  In this section, you will train a linear classifier to determine if an
#  email is Spam or Not-Spam.

# Load the Spam Email dataset
# You will have X, y in your environment
data = loadmat('spamTrain.mat')
X = data['X']
y = data['y']

print('\nTraining Linear SVM (Spam Classification)\n')
print('(this may take 1 to 2 minutes) ...\n')

C = 0.1
model = svmTrain(X, y, C, 'linear')
p = model.predict(X)

print('Training Accuracy: {:.1f}%\n'.format(
    np.mean((p == y.flatten()).astype(int))*100))

# =================== Part 4: Test Spam Classification ================
#  After training the classifier, we can evaluate it on a test set. We have
#  included a test set in spamTest.mat

# Load the test dataset

data = loadmat('spamTest.mat')
Xtest = data['Xtest']
ytest = data['ytest']

print('\nEvaluating the trained Linear SVM on a test set ...\n')

p = model.predict(Xtest)

print('Test Accuracy: {:.1f}%\n'.format(
    np.mean((p == ytest.flatten()).astype(int))*100))

# ================= Part 5: Top Predictors of Spam ====================
#  Since the model we are training is a linear SVM, we can inspect the
#  weights learned by the model to understand better how it is determining
#  whether an email is spam or not. The following code finds the words with
#  the highest weights in the classifier. Informally, the classifier
#  'thinks' that these words are the most likely indicators of spam.

# Sort the weights and obtin the vocabulary list
# 'argsort' returns the position on ascending order

weight = np.sort(model.coef_[0])
weight = weight[::-1]  # mirror it to get descending order
idx = np.argsort(model.coef_[0])
idx = idx[::-1]

print('\nTop predictors of spam: \n')

vocabList = getVocabList()

for i in range(15):
    print('Weight:', np.round(weight[i],2), ',', 
          'Word:', [key for key, values in vocabList.items() if values == idx[i]+1])


input('\nProgram paused. Press enter to continue.\n')

## =================== Part 6: Try Your Own Emails =====================
#  Now that you've trained the spam classifier, you can use it on your own
#  emails! In the starter code, we have included spamSample1.txt,
#  spamSample2.txt, emailSample1.txt and emailSample2.txt as examples.
#  The following code reads in one of these emails and then uses your
#  learned SVM classifier to determine whether the email is Spam or
#  Not Spam

# Set the file to be read in (change this to spamSample2.txt,
# emailSample1.txt or emailSample2.txt to see different predictions on
# different emails types). Try your own emails as well!

filename = 'spamSample1.txt'

# Read and predict
file_contents = readFile(filename)
word_indices = processEmail(file_contents)
x = emailFeatures(word_indices)
p = model.predict(x.T)

print('\nProcessed %s\n\nSpam Classification: %d\n' % (filename, p))
print('(1 indicates spam, 0 indicates not spam)\n\n')
