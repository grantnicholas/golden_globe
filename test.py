from nltk.tokenize import word_tokenize
import json

IMPORTANT_WORDS = ['host', 'winner', 'presenter', 'nominees']
def count_tokens(tokens):
	token_dict = {}
	for t in tokens:
		if t in token_dict:
			token_dict[t]+=1
		else:
			token_dict[t] = 1

	return token_dict

def search_tokens(token_dict, vallist):

def main():
	count = 0
	
	filename = './goldenglobes.json'
	with open(filename, 'r') as f:
		for line in f:
			if count>1000:
				break;
			text = json.loads(line)['text'].lower()
			tokens = word_tokenize(text)
			print count_tokens(tokens)












			count+=1




if __name__ == "__main__":
	main()