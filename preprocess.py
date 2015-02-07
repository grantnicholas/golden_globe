import time
from nltk import word_tokenize
import json
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

def create_preprocess_wordset():
	wordlist = []
	with open('./preprocess.txt', 'r') as f:
		wordlist = f.read().replace('\n','').split(',')
		return set(wordlist) 
@timeit
def main():
	origin_filename = './goldenglobes2015.json'
	new_filename = './processed.json'
	count = 0
	wordset = create_preprocess_wordset()

	n = open(new_filename, 'w')
	n.truncate()
	with open(origin_filename, 'r') as f:
		for line in f:
			if count > 1000000:
				break
			text = json.loads(line)['text']
			lower_text = text.lower()
			if filter_tweet(lower_text, wordset):
				n.write(line)
			count+=1
		

if __name__ == "__main__":
	main()