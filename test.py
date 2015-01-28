import nltk
from nltk import sent_tokenize, word_tokenize, pos_tag, ne_chunk
from nltk.corpus import stopwords
import json
from pprint import pprint

STOP_WORDS = set(stopwords.words('english'))
IMPORTANT_WORDS = ['host', 'winner', 'presenter', 'nominees']
MEMORY = {'host' : {}, 'winner' : {}, 'presenter' : {} , 'nominees' : {} }

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

def search_tokens(token_dict, valist):
	bool_list = [val in token_dict for val in valist]
	return bool_list

#SIDE_EFFECTS
#DICTIONARY WILL BE MUTATED
def memorize_dict(memory, token_dict):
	for t in token_dict:
		if t in memory:
			memory[t]+=1
		else:
			memory[t]=1

def helper(token_dict, people_dict, memory, important_words):
	for word in token_dict:
		if word in important_words:
			memorize_dict(memory[word], people_dict)


def extract_entities(text):
    entities = []
    for sentence in sent_tokenize(text):
        chunks = ne_chunk(pos_tag(word_tokenize(sentence)))
        entities.extend([chunk for chunk in chunks if hasattr(chunk, 'label')])
    return entities

def extract_people(entities):
	return [e for e in entities if e.label()=='PERSON']

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

def main():
	count = 0
	
	filename = './goldenglobes.json'
	with open(filename, 'r') as f:
		for line in f:
			if count>10000:
				break;

			_text = json.loads(line)['text']
			text  = _text.lower()
			_tokens = word_tokenize(text)
			tokens  = remove_stopwords(STOP_WORDS,_tokens)
			token_dict = count_tokens(tokens)
			#bool_list  = search_tokens(token_dict, IMPORTANT_WORDS)
			
			#helper(token_dict, MEMORY, IMPORTANT_WORDS)




			people = extract_people(extract_entities(_text))
			people_names = get_people_names(people)
			people_dict  = count_tokens(people_names)
			helper(token_dict, people_dict, MEMORY, IMPORTANT_WORDS)




			count+=1


	pprint(MEMORY)

if __name__ == "__main__":
	main()