"""
Improvements for Task 5:
- Smoothing: Replaced Laplace smoothing with add-k smoothing. I tested a range of k values on the 
training data and chose a k of 5.5. (Laplace smoothing corresponds to k = 1)
- Negation words: Negative or diminishing modifiers such as not, no, never, little, etc. that change the following 
word are counted as a single, distinct word to preserve context
- Removal of correlated words (disabled by default): Words that are correlated with another word are removed.
This task is time-intensive and so is disabled by default, but it can be enabled for the best model by setting 
self.ENABLE_CORR_FEATURES = True
"""

import sys
import getopt
import os
from collections import defaultdict
import math
import operator


class NaiveBayes:

    class TrainSplit:
        """Represents a set of training/testing data. self.train is a list of Examples, as is self.test.
        """

        def __init__(self):
          self.train = []
          self.test = []

    class Example:
        """Represents a document with a label. klass is 'pos' or 'neg' by convention.
           words is a list of strings.
        """

        def __init__(self):
            self.klass = ''
            self.words = []

    def __init__(self):
        """NaiveBayes initialization"""
        #############################################################################
        # TODO TODO TODO TODO TODO
        # Implement the Multinomial Naive Bayes classifier and the Naive Bayes Classifier with
        # Boolean (Binarized) features.
        # If the BOOLEAN_NB flag is true, your methods must implement Boolean (Binarized)
        # Naive Bayes (that relies on feature presence/absence) instead of the usual algorithm
        # that relies on feature counts.
        #
        # If the BEST_MODEL flag is true, include your new features and/or heuristics that
        # you believe would be best performing on train and test sets.
        #
        # If any one of the FILTER_STOP_WORDS, BOOLEAN_NB and BEST_MODEL flags is on, the
        # other two are meant to be off. That said, if you want to include stopword removal
        # or binarization in your best model, write the code accordingl
        
        # Defaults for model flags (overridden later)
        self.BOOLEAN_NB = False
        self.BEST_MODEL = False
        self.FILTER_STOP_WORDS = False
        self.APPLY_NEGATIONS = False
        
        # Whether to filter correlated features (time-intensive)
        self.ENABLE_CORR_FEATURES = False
        
        # Smoothing constant for the best model
        self.K_SMOOTH_BEST = 5.5
        
        # Negation words
        self.negWords = ["no", "not", "couldn’t", "wasn’t", "didn’t", "wouldn’t", "shouldn’t", "weren’t", "don’t",
                         "doesn’t", "haven’t", "hasn’t", "won't", "wont", "hadn’t", "never", "nobody", "nothing",
                         "neither", "nor", "nowhere", "isn’t", "can’t", "cant", "cannot", "mustn’t", "without",
                         "needn’t", "hardly", "less", "little", "rarely", "scarcely", "seldom"]
        
        # Stop word list
        self.stopList = set(self.readFile('data/english.stop'))
        
        # Number of folds
        self.numFolds = 10
                
        # Duplicate removal
        self.REMOVE_DUPLICATES = False
                
        # Number of examples (also called documents) in all classes       
        self.numExamples = 0
        
        # Number of examples in each class
        self.numExamplesByClass = {
            "pos": 0,
            "neg": 0
        }
        
        # Number of words in the entire vocabulary
        self.numWords = 0
        
        # Number of words in each class
        self.numWordsByClass = {
            "pos": 0,
            "neg": 0
        }
        
        # Number of times each word occurs in all classes 
        self.wordCount = defaultdict(lambda: 0)
        
        # Number of times each word occurs in each class
        self.wordCountByClass = {
            "pos": defaultdict(lambda: 0),
            "neg": defaultdict(lambda: 0)
        }

        # Number of examples in which two words appear together
        self.corrWords = defaultdict(lambda: 0)
        
        # Correlated words to filter
        self.stopListCorr = []
        
        # Flag to update the correlations
        self.corrUpdate = False

    def classify(self, words):
        """ TODO
            'words' is a list of words to classify. Return 'pos' or 'neg' classification.
        """
        # Process the word list
        words = self.processWords(words)
        
        # Exclude correlated words
        if self.BEST_MODEL and self.ENABLE_CORR_FEATURES:
            if self.corrUpdate:
                self.updateCorrelations()
            
            words = self.filterCorrWords(words)
            
        # Determine the classification and return
        return "pos" if self.getScore("pos", words) >= self.getScore("neg", words) else "neg"

    def addExample(self, klass, words):
        """
         * TODO
         * Train your model on an example document with label klass ('pos' or 'neg') and
         * words, a list of strings.
         * You should store whatever data structures you use for your classifier
         * in the NaiveBayes class.
         * Returns nothing
        """
        
        # Convert all words to to lower case
        wordsLower = [w.lower() for w in words]
        
        # Filter out anything that isn't a word
        words = []
        for w in wordsLower:
            if w.isalpha():
                words.append(w)
                
        # Process the word list
        words = self.processWords(words)
        
        # Number of examples in all classes
        self.numExamples += 1
        
        # Number of examples in this class
        self.numExamplesByClass[klass] += 1
        
        # Loop over all words in the example
        for w in words:
            
            # Number of words in this class
            self.numWordsByClass[klass] += 1
            
            # Number of words in the entire vocabulary
            if w not in self.wordCount:
                self.numWords += 1
            
            # Number of times this word occurs in all classes 
            self.wordCount[w] += 1
            
            # Number of times this word occurs in this class
            self.wordCountByClass[klass][w] += 1
            
            # Add to this dictionary for the current example
            if w not in self.corrWords:
                self.corrWords[w] = [self.numExamples]
            elif self.corrWords[w][-1] != self.numExamples:
                self.corrWords[w].append(self.numExamples)
        
        # Set the flag to update the correlations
        self.corrUpdate = True     
      
    def processWords(self, words):
        # Settings for the binary NB model and the best model
        if self.BOOLEAN_NB:
            self.REMOVE_DUPLICATES = True
            self.FILTER_STOP_WORDS = True
            self.APPLY_NEGATIONS = False
        
        if self.BEST_MODEL:
            self.REMOVE_DUPLICATES = True
            self.FILTER_STOP_WORDS = True
            self.APPLY_NEGATIONS = True
            
        # Filter words if applicable
        if self.FILTER_STOP_WORDS:
            words = self.filterStopWords(words)
            
        # Remove duplicates if applicable
        if self.REMOVE_DUPLICATES:
            tempSet = set(words)
            words = list(tempSet)
        
        # Apply negation modifier
        if self.APPLY_NEGATIONS:
            wordsNew = []
            wneg = ""
            
            for w in words: 
                if (w in self.negWords):
                    # Negation word: concatenate
                    wneg += "_" + w
                
                else:
                    if not wneg:
                        # Word with no preceding negation
                        wordsNew.append(w)
                    else:
                        # Word with preceding negation
                        wordsNew.append(wneg + "_" + w)
                        wneg = ""
                          
            if len(wneg) > 0:
                # Negation at the end of an example
                wordsNew.append(wneg)
            
            words = wordsNew
            
        return words
            
    def getScore(self, klass, words):
        # Laplace smoothing parameter
        if self.BEST_MODEL:
            K_SMOOTH = self.K_SMOOTH_BEST
        else:
            K_SMOOTH = 1.0
            
        # First factor in the score (class probability)
        score = math.log(self.numExamplesByClass[klass]) - math.log(self.numExamples)
        
        # Loop over all words and add their factors into the score
        for w in words:
            score += math.log(self.wordCountByClass[klass][w] + K_SMOOTH)
            score -= math.log(self.numWordsByClass[klass] + (self.numWords * K_SMOOTH))
        return score
      
    def filterCorrWords(self, words):
        filtered = []
        for word in words:
            if not word in self.stopListCorr and word.strip() != '':
                filtered.append(word)
        return filtered
      
    def updateCorrelations(self):
        self.stopListCorr = []
        
        i = 0
        print("Updating correlations table...")
        for w in self.corrWords:
            len_w = len(self.corrWords[w])
            for r in self.corrWords:
                if (r not in self.stopListCorr):
                  len_r = len(self.corrWords[r])
                  if (len_r > len_w) and (all(x in self.corrWords[r] for x in self.corrWords[w])):
                      self.stopListCorr.append(w)
                      break
                    
            i += 1
            if (i % 100 == 0): 
                print(str(i) + " of " + str(len(self.corrWords)))
            
        self.corrUpdate = False
    
    # END TODO (Modify code beyond here with caution)
    #############################################################################

    def readFile(self, fileName):
        """
         * Code for reading a file.  you probably don't want to modify anything here,
         * unless you don't like the way we segment files.
        """
        contents = []
        f = open(fileName)
        for line in f:
            contents.append(line)
        f.close()
        result = self.segmentWords('\n'.join(contents))
        return result
  
    def segmentWords(self, s):
        """
         * Splits lines on whitespace for file reading
        """
        return s.split()
  
    def trainSplit(self, trainDir):
        """Takes in a trainDir, returns one TrainSplit with train set."""
        split = self.TrainSplit()
        posTrainFileNames = os.listdir('%s/pos/' % trainDir)
        negTrainFileNames = os.listdir('%s/neg/' % trainDir)
        for fileName in posTrainFileNames:
            example = self.Example()
            example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
            example.klass = 'pos'
            split.train.append(example)
        for fileName in negTrainFileNames:
            example = self.Example()
            example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
            example.klass = 'neg'
            split.train.append(example)
        return split

    def train(self, split):
        for example in split.train:
            words = example.words
            if self.FILTER_STOP_WORDS:
                words = self.filterStopWords(words)
            self.addExample(example.klass, words)

    def crossValidationSplits(self, trainDir):
        """Returns a lsit of TrainSplits corresponding to the cross validation splits."""
        # splits = []
        posTrainFileNames = os.listdir('%s/pos/' % trainDir)
        negTrainFileNames = os.listdir('%s/neg/' % trainDir)
        # for fileName in trainFileNames:
        for fold in range(0, self.numFolds):
            split = self.TrainSplit()
            for fileName in posTrainFileNames:
              example = self.Example()
              example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
              example.klass = 'pos'
              if fileName[2] == str(fold):
                  split.test.append(example)
              else:
                  split.train.append(example)
            for fileName in negTrainFileNames:
                example = self.Example()
                example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
                example.klass = 'neg'
                if fileName[2] == str(fold):
                    split.test.append(example)
                else:
                    split.train.append(example)
            yield split

    def test(self, split):
        """Returns a list of labels for split.test."""
        labels = []
        for example in split.test:
            words = example.words
            if self.FILTER_STOP_WORDS:
                words = self.filterStopWords(words)
            guess = self.classify(words)
            labels.append(guess)
        return labels
  
    def buildSplits(self, args):
        """Builds the splits for training/testing"""
        # trainData = []
        # testData = []
        splits = []
        trainDir = args[0]
        if len(args) == 1:
            print('[INFO]\tPerforming %d-fold cross-validation on data set:\t%s' % (self.numFolds, trainDir))

            posTrainFileNames = os.listdir('%s/pos/' % trainDir)
            negTrainFileNames = os.listdir('%s/neg/' % trainDir)
            for fold in range(0, self.numFolds):
                split = self.TrainSplit()
                for fileName in posTrainFileNames:
                    example = self.Example()
                    example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
                    example.klass = 'pos'
                    if fileName[2] == str(fold):
                        split.test.append(example)
                    else:
                        split.train.append(example)
                for fileName in negTrainFileNames:
                    example = self.Example()
                    example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
                    example.klass = 'neg'
                    if fileName[2] == str(fold):
                        split.test.append(example)
                    else:
                        split.train.append(example)
                splits.append(split)
        elif len(args) == 2:
            split = self.TrainSplit()
            testDir = args[1]
            print('[INFO]\tTraining on data set:\t%s testing on data set:\t%s' % (trainDir, testDir))
            posTrainFileNames = os.listdir('%s/pos/' % trainDir)
            negTrainFileNames = os.listdir('%s/neg/' % trainDir)
            for fileName in posTrainFileNames:
                example = self.Example()
                example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
                example.klass = 'pos'
                split.train.append(example)
            for fileName in negTrainFileNames:
                example = self.Example()
                example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
                example.klass = 'neg'
                split.train.append(example)

            posTestFileNames = os.listdir('%s/pos/' % testDir)
            negTestFileNames = os.listdir('%s/neg/' % testDir)
            for fileName in posTestFileNames:
                example = self.Example()
                example.words = self.readFile('%s/pos/%s' % (testDir, fileName))
                example.klass = 'pos'
                split.test.append(example)
            for fileName in negTestFileNames:
                example = self.Example()
                example.words = self.readFile('%s/neg/%s' % (testDir, fileName))
                example.klass = 'neg'
                split.test.append(example)
            splits.append(split)
        return splits
  
    def filterStopWords(self, words):
        """Filters stop words."""
        filtered = []
        for word in words:
            if not word in self.stopList and word.strip() != '':
                filtered.append(word)
        return filtered


def test10Fold(args, FILTER_STOP_WORDS, BOOLEAN_NB, BEST_MODEL):
    nb = NaiveBayes()
    splits = nb.buildSplits(args)
    avgAccuracy = 0.0
    fold = 0
    classifier = None
    for split in splits:
        classifier = NaiveBayes()
        classifier.FILTER_STOP_WORDS = FILTER_STOP_WORDS
        classifier.BOOLEAN_NB = BOOLEAN_NB
        classifier.BEST_MODEL = BEST_MODEL
        accuracy = 0.0
        for example in split.train:
            words = example.words
            classifier.addExample(example.klass, words)

        for example in split.test:
            words = example.words
            guess = classifier.classify(words)
            if example.klass == guess:
                accuracy += 1.0
        accuracy = accuracy / len(split.test)
        avgAccuracy += accuracy
        print('[INFO]\tFold %d Accuracy: %f' % (fold, accuracy))
        fold += 1
    avgAccuracy = avgAccuracy / fold
    print('[INFO]\tAccuracy: %f' % avgAccuracy)
        
    # interpret the decision rule of the model of the last fold
    pos_signal_words, neg_signal_words = analyze_model(classifier)
    print('[INFO]\tWords for pos class: %s' % ','.join(pos_signal_words))
    print('[INFO]\tWords for neg class: %s' % ','.join(neg_signal_words))

    
def classifyFile(FILTER_STOP_WORDS, BOOLEAN_NB, BEST_MODEL, trainDir, testFilePath):
    classifier = NaiveBayes()
    classifier.FILTER_STOP_WORDS = FILTER_STOP_WORDS
    classifier.BOOLEAN_NB = BOOLEAN_NB
    classifier.BEST_MODEL = BEST_MODEL
    trainSplit = classifier.trainSplit(trainDir)
    classifier.train(trainSplit)
    testFile = classifier.readFile(testFilePath)
    print(classifier.classify(testFile))

    
def main(): 
    FILTER_STOP_WORDS = False
    BOOLEAN_NB = False
    BEST_MODEL = False
    (options, args) = getopt.getopt(sys.argv[1:], 'fbm')
    
    # This code allows setting options without the command line
    if not args:
      options = [('-m', '')]
      args = ["./data/imdb"]
    
    if ('-f', '') in options:
        FILTER_STOP_WORDS = True
    elif ('-b', '') in options:
        BOOLEAN_NB = True
    elif ('-m', '') in options:
        BEST_MODEL = True
    
    if len(args) == 2 and os.path.isfile(args[1]):
        classifyFile(FILTER_STOP_WORDS, BOOLEAN_NB, BEST_MODEL, args[0], args[1])
    else:
        test10Fold(args, FILTER_STOP_WORDS, BOOLEAN_NB, BEST_MODEL)


def analyze_model(nb_classifier):
    # TODO: This function takes a <nb_classifier> as input, and outputs two word list <pos_signal_words> and
    #  <neg_signal_words>. <pos_signal_words> is a list of 10 words signaling the positive klass, and <neg_signal_words>
    #  is a list of 10 words signaling the negative klass.
    
    # Positive
    sorted_list = dict(sorted(nb_classifier.wordCountByClass["pos"].items(), key=lambda item: item[1], reverse=True)[:10])
    pos = sorted_list.keys()
    
    sorted_list = dict(sorted(nb_classifier.wordCountByClass["neg"].items(), key=lambda item: item[1], reverse=True)[:10])
    neg = sorted_list.keys()
    return [pos, neg]
    
    
if __name__ == "__main__":
    main()
