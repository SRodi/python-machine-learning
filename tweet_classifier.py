# Rodigari Simone
import codecs
import re
import time
import matplotlib.pyplot as plt
import numpy as np
from nltk.corpus import stopwords
# common words to remove while cleaning
stop_words = set(stopwords.words('english'))
# initialize path
path = '/' # <<--------------PLEASE ENTER YOUR LOCAL PATH HERE
# initialize data_set
dataFilePos = codecs.open("train/trainPos.txt", 'r', encoding="ISO-8859-1")
dataFileNeg = codecs.open("train/trainNeg.txt", 'r', encoding="ISO-8859-1")
# declare set
unique_words = set()
unique_pre_processed = set()
# dictionaries for test Positive and Negative tweet files
testPos = dict()
testNeg = dict()
# declare dictionaries for positive and negative files
dictionaryNeg = dict()
dictionaryPos = dict()  # dict.fromkeys(x, 0)
# probability dictionary (key has +' '+'pos/neg')
probDict = dict()
# outcome lists used for plotting
positiveOutcome = list()  # positive
negativeOutcome = list()
positiveOutcome1 = list()  # negative
negativeOutcome1 = list()
# counting vars for final evaluation on accuracy performance
count_pos_tweets_tested = 80
count_neg_tweets_tested = 80
pp_test_file_count = 0
nn_test_file_count = 0
pn_test_file_count = 0
np_test_file_count = 0

''' --STAGE 1--
    get number of occurrences of 
    a word in the file, add words to a set
    and add number of occurrences to a dictionary    
'''


def populate_pos_neg_dictionaries(pre_processing):
    print('please wait..')

    print("==positive train file===")
    # for loops to populate dictionaries and sets
    for linePos in dataFilePos:  # loop through file

        if pre_processing:
            formatted = format_string(linePos)

            if len(formatted) > 0:

                for word_positive in formatted:
                    unique_pre_processed.add(word_positive)  # add word to pp set
                    dictionaryPos.setdefault(word_positive, 0)  # add word to dictionary
                    dictionaryPos[word_positive] += 1  # increment word count in dictionary
                    # print(word_positive)
        else:
            for wordPos in linePos.strip().split(" "):
                word_positive = wordPos.lower()
                unique_words.add(word_positive)  # add word to set
                dictionaryPos.setdefault(word_positive, 0)  # add word to dictionary
                dictionaryPos[word_positive] += 1  # increment word count in dictionary
                # print(word_positive)

    print("==negative train file===")
    # for loops to populate dictionaries and sets
    for lineNeg in dataFileNeg:  # loop through file
        if pre_processing:

            formatted_neg = format_string(lineNeg)

            if len(formatted_neg) > 0:
                for word_negative in formatted_neg:
                    unique_pre_processed.add(word_negative)  # add word to pp set
                    dictionaryNeg.setdefault(word_negative, 0)  # add word to dictionary
                    dictionaryNeg[word_negative] += 1  # increment word count in dictionary
                    # print(word_negative)
        else:
            for wordNeg in lineNeg.strip().split(" "):
                word_negative = wordNeg.lower()
                unique_words.add(word_negative)  # add word to set
                dictionaryNeg.setdefault(word_negative, 0)
                dictionaryNeg[word_negative] += 1  # add word to dictionary
                # print(word_negative)

    start_calculation()

    print("set length (unique words pre-processed)", len(unique_pre_processed))
    print("set length (unique words)", len(unique_words))
    print("length of dict positive: ", len(dictionaryPos))
    print("length of dict negative: ", len(dictionaryNeg))
    print("occurrence of word 'hello' in positive dict: ", dictionaryPos['hello'])
    print("occurrence of word 'hello' in negative dict: ", dictionaryNeg['hello'])


''' --STAGE 2--
    work out the conditional probabilities 
    (probability that one event will occur given that a second event has already occurred)
    for all words (for each class) for each word w you should work out the P(w|positive) and P(w|negative)
    Bayesian Classification , multinomial model    
    P(c | d) = ( P(d | c) P(c) ) / P(d) 
'''


# calculate the probability based on total number of occurrences in
# the positive file and negative file


def calculate_prob(pos_or_neg, pp):
    dict1 = {}
    dict2 = {}
    if pos_or_neg == 'pos':
        dict1 = dictionaryPos
        dict2 = dictionaryNeg
    if pos_or_neg == 'neg':
        dict1 = dictionaryNeg
        dict2 = dictionaryPos
    if pp:
        denominator = len(unique_pre_processed)
    else:
        denominator = len(unique_words)

    for word in dict1:
        # ================= LaPlace theorem ===================
        numerator = dict1[word] + 1
        denominator += dict1[word]

        if word in dict2:
            denominator += dict2[word]  # tot pos + tot neg + tot unique words

        probability_field = word + ' ' + pos_or_neg
        probDict[probability_field] = round(numerator / denominator, 10) * 100

        # ================= initial model ===================
        # if word in dict2:
        #     denominator = dict2[word] + dict1[word]
        # else:
        #     denominator = dict1[word]
        #
        # probability_field = word + ' ' + pos_or_neg
        # probDict[probability_field] = round((dict1[word] + 1) / denominator, 6)
        # =====================================================


''' --STAGE 3--
The final section of your code will take as input a new tweet (a tweet that has not been used for training the
algorithm) and classify the tweet as a positive or negative review. You will need to read all words from
the tweet and determine the probability of that tweet being positive and the probability of it being negative.
'''


def start_calculation():  # this method initiates the calculation of probabilities
    print('calculating probabilities now..')

    calculate_prob('pos', False)
    print('positive probability dictionary initialized..')

    calculate_prob('neg', False)
    print('negative probability dictionary initialized..')


def eval_tweet_probability(p, n, choice): # this method evaluates probability of the tweet
    global pp_test_file_count
    global nn_test_file_count
    global pn_test_file_count
    global np_test_file_count
    global count_pos_tweets_tested
    global count_neg_tweets_tested

    # classify as positive
    if p > n:
        # print("the tweet is POSITIVE")
        if choice == 'pos':
            pp_test_file_count += 1
        elif choice == 'neg':
            np_test_file_count += 1
        else:
            print('not testing accuracy of files')
    # classify as negative
    elif n > p:
        # print("the tweet is NEGATIVE")
        if choice == 'neg':
            nn_test_file_count += 1
        elif choice == 'pos':
            pn_test_file_count += 1
        else:
            print('not testing accuracy of files')
    # do not add the tweet to count of tweet tested
    else:
        if choice == 'neg':
            count_neg_tweets_tested -= 1
        elif choice == 'pos':
            count_pos_tweets_tested -= 1
        # print("the tweet could be either POSITIVE OR NEGATIVE")

    # append to list if testing on positive or negative file
    if choice == 'pos':
        positiveOutcome.append(p)
        negativeOutcome.append(n)
        # print('added to pos outcomes')
    elif choice == 'neg':  # choice == 'neg' & (positive != 1 & negative != 1):
        positiveOutcome1.append(p)
        negativeOutcome1.append(n)
        # print('added to neg outcomes')
    else:
        print('no outcome added for plotting')

    print('probability of this tweet to be positive: ', p, '%')
    print('probability of this tweet to be negative: ', n, '%')


def calculate_word_probability(tweet, choice): # this method calculation probabilities of a word
    try:
        tweet_words_array = tweet.strip().split(' ')
    except AttributeError:
        print('AE uploading from test file..')
        tweet_words_array = tweet
    # initialize probability variables
    negative = 1
    positive = 1
    # for loop to iterate through tweet sentence by user
    for word in tweet_words_array:

        if word in dictionaryNeg:
            # word = format_string(word)
            probability_negative = word + ' neg'
            negative *= probDict[probability_negative]

            print(word, ' - negative probability: ', negative)

        if word in dictionaryPos:
            # word = format_string(word)
            probability_positive = word + ' pos'
            positive *= probDict[probability_positive]
            print(word, ' - positive probability: ', positive)

    p = positive * 100
    n = negative * 100

    try:
        eval_valid = positive + negative  # add probability of word to be either pos and neg
        if eval_valid < 1:  # if below 1 (ie no 100% probability)
            eval_tweet_probability(p, n, choice)

    except TypeError:
        print('type error--')


def test_tweets_upload(pp):  # this method uploads test files and pre-process where required
    dataTestPos = codecs.open('test/testPos.txt', 'r', encoding="ISO-8859-1")
    dataTestNeg = codecs.open('test/testNeg.txt', 'r', encoding="ISO-8859-1")

    key = 0
    print("==positive test file===")
    # for loops to populate test dictionaries
    for linePos in dataTestPos:  # loop through file

        if pp:
            formatted = format_string(linePos)

            for word_positive in formatted:
                testPos[key] = word_positive  # increment word count in dictionary
                key += 1

        else:
            for wordPos in linePos.strip().split(" "):
                word_positive = wordPos.lower()
                testPos[key] = word_positive
                key += 1

    key = 0
    print("==negative test file===")
    # for loops to populate test dictionaries
    for lineNeg in dataTestNeg:  # loop through file

        if pp:
            formattedNeg = format_string(lineNeg)

            for word_negative in formattedNeg:
                testNeg[key] = word_negative  # increment word count in dictionary
                key += 1

        else:
            for wordNeg in linePos.strip().split(" "):
                word_negative = wordNeg.lower()
                testNeg[key] = word_negative
                key += 1


def test_file(choice, pp):  # this method provides final evaluation of test files
    test_tweets_upload(pp)
    try:
        if choice == "pos":
            for i in range(count_pos_tweets_tested):
                calculate_word_probability(testPos[i], choice)
        else:
            for i in range(count_neg_tweets_tested):
                calculate_word_probability(testNeg[i], choice)
        print('test completed on ' + choice.upper() + ' file.')
    except KeyError:
        print('key error')

    if choice == 'pos':
        print('accuracy of positive pos file: ', round((pp_test_file_count / count_pos_tweets_tested) * 100, 2), ' %')
        print('accuracy of positive neg file: ', round((pn_test_file_count / count_pos_tweets_tested) * 100, 2), ' %')

    if choice == 'neg':
        print('accuracy of negative neg file: ', round((nn_test_file_count / count_neg_tweets_tested) * 100, 2), ' %')
        print('accuracy of negative pos file: ', round((np_test_file_count / count_neg_tweets_tested) * 100, 2), ' %')

    clear_global_vars()


def clear_global_vars():  # this method cleares variables used for test files evaluation
    global pp_test_file_count
    global pn_test_file_count
    global nn_test_file_count
    global np_test_file_count
    pp_test_file_count = 0
    pn_test_file_count = 0
    nn_test_file_count = 0
    np_test_file_count = 0


def user_interaction():  # this method allows user to enter a tweet to evaluate probability for
    # input from user
    tweet_words = str(input("type a tweet.."))
    calculate_word_probability(tweet_words, '')


''' --STAGE 4--

This section is for the investigations of the impact of some pre-processing techniques on the accuracy. Common 
techniques include lowering the case of all words, punctuations removal, stop-word removal, n-grams, etc.
The regular expression library in Python may prove useful in performing pre-processing techniques. 
(re module https://docs.python.org/3.6/library/re.html ). This provides capabilities for extracting

whole words and removing punctuation. See example on the next page. You can find a tutorial on regular expression at 
https://developers.google.com/edu/python/regular-expressions . An alternative is the use of NLTK, Pythons natural 
language toolkit (http://nltk.org/ ). Note to use this from Spyder you will need to run nltk.download('all'). 
It is a power library that provides a range of capabilities including stemming, lemmatization, etc.
'''


def format_string(string_to_format):  # this method clean the tweet passed in as a parameter

    # remove all user names (starting with @)
    no_username = re.sub(r'@\w+', '', string_to_format)

    # remove all html quote tags (starting with &)
    no_html_quote = re.sub(r'&\w+', '', no_username)

    # remove all chars not alphabetic
    no_special_chars = re.sub(r"[^a-zA-Z]+", ' ', no_html_quote)

    # remove short words
    shortword = re.sub(r'\W*\b\w{1,3}\b', '', no_special_chars)

    # remove all punctuation
    result_no_punctuation = re.sub(r'\W+', ' ', shortword)

    # remove all stopwords
    words = [w for w in str(result_no_punctuation).split() if not w.lower() in stop_words]

    # return result in lower characters
    return ' '.join(words).lower().split()


''' --STAGE 5--

You need to visualize the classification results before and after the pre-processing using two- dimensional graphs.
You can use matplotlib and seaborn Python libraries.

'''


def plot(file_choice):  # this method plots results in a double bar chart for positive test file
    try:
        plt.close()
    except ValueError:
        print('nothing to close')
    try:
        if file_choice == "pos":
            positive = positiveOutcome
            negative = negativeOutcome
            label = 'positive'
        else:
            positive = positiveOutcome1
            negative = negativeOutcome1
            label = 'negative'
        num_locations = len(positiveOutcome)
        fig, ax = plt.subplots()

        ind = np.arange(num_locations)  # the x locations for the groups
        width = 0.15  # the width of the bars
        p1 = ax.bar(ind, positive, width, color='g', bottom=0)  # * cm, yerr=menStd)
        p2 = ax.bar(ind + width, negative, width, color='r', bottom=0)  # * cm, yerr=womenStd)

        ax.set_title('Scores of ' + label + ' test file')
        ax.legend((p1[0], p2[0]), ('positive', 'negative'))
        ax.autoscale_view()
        plt.show()
    except IndexError or ValueError:
        print("no data")


def plotNeg():  # this method plots results in a double bar chart for negative test file
    try:
        plt.close()
    except ValueError:
        print('nothing to close')
    try:
        N = len(positiveOutcome1)
        fig, ax = plt.subplots()

        ind = np.arange(N)  # the x locations for the groups
        width = 0.15  # the width of the bars
        p1 = ax.bar(ind, positiveOutcome1, width, color='g', bottom=0)  # * cm, yerr=menStd)
        p2 = ax.bar(ind + width, negativeOutcome1, width,
                    color='r', bottom=0)  # * cm, yerr=womenStd)

        ax.set_title('Scores of negative test file')
        ax.legend((p1[0], p2[0]), ('positive', 'negative', 'ppPos', 'ppNeg'))
        ax.autoscale_view()

        plt.show()
    except IndexError:
        print("no data")


def clear_outcomes():  # this method clears outcomes variables for plotting
    global positiveOutcome
    global negativeOutcome
    global positiveOutcome1
    global negativeOutcome1

    positiveOutcome.clear()
    negativeOutcome.clear()
    positiveOutcome1.clear()
    negativeOutcome1.clear()


'''
main execution of the program
'''


def process_choice():  # this method takes user choice and directs accordingly
    try:
        choice = int(input('chose from above..'))

        if choice == 9:
            # loop = False
            exit(0)
        if choice == 1:
            user_interaction()
        if choice == 2:
            test_file('pos', False)
        if choice == 3:
            test_file('neg', False)
        if choice == 4:
            t0 = time.time()
            unique_words.clear()
            populate_pos_neg_dictionaries(True)  # do format
            print('cleaning upload done.')
            test_file('pos', True)  # do format
            test_file('neg', True)  # do format
            t1 = time.time()
            print("elapsed: ", t1 - t0)
        if choice == 5:
            plot('pos')
        if choice == 6:
            plotNeg()  # ('neg')
        if choice == 7:
            clear_outcomes()
        if choice == 8:
            t0 = time.time()
            populate_pos_neg_dictionaries(False)  # false means no pre processing
            t1 = time.time()
            print("elapsed: ", t1 - t0)

    except ValueError:
        print('enter a number')


def execution():  # main execution with menu and call to process method within a while loop

    loop = True
    try:
        while loop:
            print('1. user interaction\n'
                  '2. testing on positive file\n'
                  '3. testing on negative file\n'
                  '4. pre_processing\n'
                  '5. plot positive file\n'
                  '6. plot negative file\n'
                  '7. clear outcomes\n'
                  '8. load dictionaries (no pre_processing)\n'
                  '9. exit')

            process_choice()

            print("total unique words : ", len(unique_words))
            print("total unique pp words : ", len(unique_pre_processed))
            print("total positive: ", len(dictionaryPos))
            print("total negative: ", len(dictionaryNeg))
            print("total probabilities: : ", len(probDict))
    except KeyboardInterrupt:
        print('\nbye')


execution()  # start
