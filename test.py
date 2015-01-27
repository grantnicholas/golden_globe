from nltk.tokenize import word_tokenize
import nltk
import json

nltk.download()

count = 0
importantobjects = {}
filename = './goldenglobes.json'
with open(filename, 'r') as f:
	for line in f:
		if count>1000:
			break;
		text = json.loads(line)['text']
		tokens = word_tokenize(text)
		print tokens
		count+=1