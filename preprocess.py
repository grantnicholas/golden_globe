# preprocess tweets module 
def filter_tweet(tweet, wordlist):
	if any(word in tweet for word in wordlist):
		return True
	else:
		return False