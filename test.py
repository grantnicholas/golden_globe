import nltk
from nltk import sent_tokenize, word_tokenize, ne_chunk
from nltk.corpus import stopwords
from stemming.porter2 import stem

import json
from pprint import pprint
import time

import nltk.data, nltk.tag
tagger = nltk.data.load(nltk.tag._POS_TAGGER)


"""
TODO:
List of or word matches in important words
IE): must match "best actor drama [movie|film|motion picture]
where one of movie film or motion picture must be a thing

"""
	
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

# class keywords:

# 	def __init__(self, required_words, or_words):
# 		self.required_words = [word.split(' ') for word in required_words]
# 		self.or_words = [word.split(' ') for word in or_words]

# 	def eq(self, wordlist):
# 		setofwords = set(wordlist)

# 		if not all([word in setofwords for word in self.required_words]):
# 			return False

# 		return any([word in setofwords for word in self.or_words])



def fake_globals():
	STOP_WORDS = set(stopwords.words('english'))
	IMPORTANT_WORDS = ['hosts', 
					   'winners', 
					   'presenters', 
					   'nominees', 
					   'best actor drama', 
					   'best actress drama', 
					   'best actor comedy musical',
					   'best actress comedy musical',
					   'best actor tv',
					   'best actress tv',
					   'best animated',
					   'best foreign language',
					   'best actor supporting',
					   'best actress supporting',
					   'best director',
					   'best screenplay',
					   'best original score',
					   'best original song',
					   'best tv drama',
					   'best tv comedy musical',
					   'best actor tv comedy musical',
					   'best actress tv comedy musical',
					   'best miniseries tv',
					   'best actress miniseries tv',
					   'best actor miniseries tv'
					   ]
	_IMPORTANT_WORDS = [('hosts'),
						('winners'),
						('presenters'),
						('nominees'), 
					    ('actor drama'), 
					    ('actress drama'), 
					    ('actor','comedy musical'),
					    ('actress','comedy musical'),
					    ('actor','tv television'),
					    ('actress','tv television'),
					    ('animated'),
					    ('foreign language'),
					    ('actor supporting'),
					    ('actress supporting'),
					    ('director'),
					    ('screenplay'),
					    ('original score'),
					    ('original song'),
					    ('drama','tv television'),
					    ('comedy musical', 'tv television'),
					    ('actor comedy musical','tv television'),
					    ('actress comedy musical', 'tv television'),
					    ('miniseries','tv television'),
					    ('actress miniseries','tv television'),
					    ('actor miniseries','tv television')
						]

	FILT_WORDS = 		['golden',
				  		 'globes',
				  		 'goldenglobes',
					  	 'RT',
					  	 'wins',
					  	 'movie',
					  	 'film',
					  	 'series'
					  	]
	SOW = set([])
	FOW = set([])

	for phrase in IMPORTANT_WORDS:
		for word in phrase.split(' '):
			FOW.add(stem(word.lower()))
			SOW.add(stem(word.lower()))

	for word in STOP_WORDS:
		FOW.add(stem(word.lower()))

	for word in FILT_WORDS:
		FOW.add(stem(word.lower()))

	IMPORTANT_WORDS = tokenize_stem_list(IMPORTANT_WORDS)
	MEMORY = {k: {} for k in IMPORTANT_WORDS}

	return STOP_WORDS, IMPORTANT_WORDS, MEMORY, SOW, FOW

"""
Returns a list of tuples
Note: we end up hashing the phrases in important words, so we must convert the list to a tuple
"""
def tokenize_stem_list(alist):
	def tokenize_stem_one(thestr):
		return tuple(map(stem, thestr.split(' ')))

	return map(tokenize_stem_one, alist)


def count_tokens(tokens):
	token_dict = {}
	for t in tokens:
		if t in token_dict:
			token_dict[t]+=1
		else:
			token_dict[t] = 1

	return token_dict

def remove_stopwords(stopwords, tokens):
	sanitized_tokens = [w for w in tokens if w not in stopwords]
	return sanitized_tokens

def stem_words(tokens):
	return [stem(w) for w in tokens]

"""
Function composition: create a dictionary of token counts from text
"""
def text_to_token_dict(text, stopwords=None, dostem=False):
	tokens = word_tokenize(text)
	token_dict = {}

	if dostem ==True:
		tokens = stem_words(tokens)

	if stopwords !=None:
		tokens = remove_stopwords(stopwords, tokens)

	token_dict = count_tokens(tokens)
	return token_dict

"""
SIDE EFFECTS; MUTATES DICTIONARY
"""
def memorize_dict(memory, token_dict):
	for t in token_dict:
		if t in memory:
			memory[t]+=1
		else:
			memory[t]=1

"""
SIDE EFFECTS; MUTATES DICTIONARY
"""
def memorize_people_if_tokens_match(token_dict, people_dict, memory, important_words):
	for phrase in important_words:
		thebool=1
		for word in phrase:
			if word not in token_dict:
				thebool=0
		if thebool==1:
			memorize_dict(memory[phrase], people_dict)


"""
SIDE EFFECTS; MUTATES DICTIONARY
"""
def _memorize_people_if_tokens_match(token_dict, people_dict, memory, important_words):
	for phrase in important_words:
		required_words, optional_words = phrase
		if all([word in token_dict for word in required_words]):
			if any([word in token_dict for word in optional_words]):
				memorize_dict(memory[phrase], people_dict)
		


"""
Extract entities from text which are labeled if they are a person or not
"""
def extract_entities(text):
    entities = []
    for sentence in sent_tokenize(text):
        chunks = ne_chunk(tagger.tag(word_tokenize(sentence)))
        entities.extend([chunk for chunk in chunks if hasattr(chunk, 'label')])
    return entities

def extract_people(entities):
	return [e for e in entities if e.label()=='PERSON']

"""
Collect the names of the people from the list of entities [ie: REMOVE the labels]
"""
def get_people_names(people):
	people_names = []
	for p in people:
		full_name = []
		for leaf in p.leaves():
			full_name.append(leaf[0])
		people_names.append(full_name)

	return _convert_full_names(people_names)

def _convert_full_names(people_names):

	def _lambda(name):
		if len(name)>1:
			return " ".join(name)
		else:
			return name[0]
	
	return map(_lambda, people_names)

"""
Function composition: create dictionary of extracted people name counts from text
"""
@timeit
def text_to_people_dict(text):
	people = extract_people(extract_entities(text))
	people_names = get_people_names(people)
	people_dict  = count_tokens(people_names)
	return people_dict



"""
Naive capturing of people names with capitalization:
intended to reduce computation time by a fuckload
"""
def _get_people_names(text, SOW):
	names = []
	for sentence in sent_tokenize(text):
		names.extend(filter(lambda x: x[0].isupper(), word_tokenize(sentence)))
	filt_names = [name for name in names if stem(name.lower()) not in SOW]
	return filt_names

"""
Naive
"""
def text_to_people_dict_naive_fast(text, SOW):
	people_names = _get_people_names(text, SOW)
	people_dict  = count_tokens(people_names)
	return people_dict


def get_top_n_vals(memory, n):
	for d in memory:
		print d
		for w in sorted(memory[d], key=memory[d].get, reverse=True)[0:n]:
			print '\t', w, memory[d][w]

@timeit
def main():
	STOP_WORDS, IMPORTANT_WORDS, MEMORY, SOW, FOW = fake_globals()
	count = 0

	filename = './goldenglobes.json'
	with open(filename, 'r') as f:
		for line in f:
			if count>1000000:
				break;

			text = json.loads(line)['text']
			lower_text  = text.lower()


			token_dict = text_to_token_dict(lower_text,STOP_WORDS,True)

			"""
			Only do the expensive people parsing step if the tweet has any of the
			keywords we identify at the beginning
			Speedup of around ~5times
			"""
			if any([True for tok in token_dict if tok in SOW]):
				"""This people dict step in NLTK is slow"""
				#people_dict = text_to_people_dict(text)
				people_dict =  text_to_people_dict_naive_fast(text,FOW)

				memorize_people_if_tokens_match(token_dict, people_dict, MEMORY, IMPORTANT_WORDS)




			count+=1


	#pprint(MEMORY)
	get_top_n_vals(MEMORY, 5)

if __name__ == "__main__":
	main()