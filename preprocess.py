import time
from nltk import word_tokenize

"""
Timing decorator to track how long a function takes
@timeit
"""
def timeit(f):

    def timed(*args, **kw):

        ts = time.time()
        result = f(*args, **kw)
        te = time.time()

        print 'func:%r took: %2.4f sec' % \
          (f.__name__, te-ts)
        return result

    return timed

# preprocess tweets module 
#@timeit
def filter_tweet(tweet, wordset):
	#O(1) lookup instead of O(n) for a list
	#Much faster
	if any([True for word in word_tokenize(tweet) if word in wordset]):
		return True
	else:
		return False