import nltk
from nltk import sent_tokenize, word_tokenize, pos_tag, ne_chunk
from nltk.corpus import stopwords
from stemming.porter2 import stem

import json
from pprint import pprint

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
					   'best actor supporting role',
					   'best actress supporting role',
					   'best director',
					   'best screenplay',
					   'best original score',
					   'best original song',
					   'best tv drama',
					   'best tv comedy musical',
					   'best actor tv comedy musical',
					   'best actress tv comedy musical',
					   'best mini series tv',
					   'best actress mini series tv',
					   'best actor mini series tv'
					   ]

	IMPORTANT_WORDS = tokenize_stem_list(IMPORTANT_WORDS)

	MEMORY = {k: {} for k in IMPORTANT_WORDS}

	return STOP_WORDS, IMPORTANT_WORDS, MEMORY

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
Extract entities from text which are labeled if they are a person or not
"""
def extract_entities(text):
    entities = []
    for sentence in sent_tokenize(text):
        chunks = ne_chunk(pos_tag(word_tokenize(sentence)))
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


def get_top_n_vals(memory, n):
	for d in memory:
		print d
		for w in sorted(memory[d], key=memory[d].get, reverse=True)[0:n]:
			print '\t', w, memory[d][w]

def main():
	STOP_WORDS, IMPORTANT_WORDS, MEMORY = fake_globals()
	count = 0
	
	filename = './goldenglobes.json'
	with open(filename, 'r') as f:
		for line in f:
			if count>10000:
				break;

			_text = json.loads(line)['text']
			text  = _text.lower()

			# _tokens = word_tokenize(text)
			# tokens  = stem_words(remove_stopwords(STOP_WORDS,_tokens))
			# token_dict = count_tokens(tokens)

			token_dict = text_to_token_dict(text, STOP_WORDS, True)

			people = extract_people(extract_entities(_text))
			people_names = get_people_names(people)
			people_dict  = count_tokens(people_names)
			memorize_people_if_tokens_match(token_dict, people_dict, MEMORY, IMPORTANT_WORDS)




			count+=1


	#pprint(MEMORY)
	get_top_n_vals(MEMORY, 5)

if __name__ == "__main__":
	main()