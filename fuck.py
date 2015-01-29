import nltk
from nltk import sent_tokenize, word_tokenize, pos_tag, ne_chunk
from pprint import pprint
from stemming.porter2 import stem

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
		print p
		full_name = []
		for leaf in p.leaves():
			print leaf
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

if __name__ == '__main__':
    text = """
A multi-agency manhunt is under way across several states and Mexico after
police say the former Los Angeles police officer suspected in the murders of a
college basketball coach and her fiance last weekend is following through on
his vow to kill police officers after he opened fire Wednesday night on three
police officers, killing one.
"In this case, we're his target," Sgt. Rudy Lopez from the Corona Police
Department said at a press conference.
The suspect has been identified as Christopher Jordan Dorner, 33, and he is
considered extremely dangerous and armed with multiple weapons, authorities
say. The killings appear to be retribution for his 2009 termination from the
 Los Angeles Police Department for making false statements, authorities say.
Dorner posted an online manifesto that warned, "I will bring unconventional
and asymmetrical warfare to those in LAPD uniform whether on or off duty."
"""

print get_people_names(extract_people(extract_entities(text)))

def tokenize_stem_list(alist):
	def tokenize_stem_one(thestr):
		return tuple(map(stem, thestr.split(' ')))

	return map(tokenize_stem_one, alist)

print tokenize_stem_list(['best actor', 'best actress', 'pooper pooping', 'hosts'])
    