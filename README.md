*Changelog Feb 2:*
Added preprocess.txt which preprocess the tweets to get rid useless tweets. 
The program runs ~53 seconds now returning almost identifcal result vs w/o
preprocessing. (only differs in small counts, no difference in ranking).
Could add more words to the preprocess list

*Changelog Jan 29:*
Used a naive person detector (ie: is a word capitalized) instead of
the part of speech tagger within NLTK. Sped up the program a shitton. 
Parses ~1million tweets within ~2-3 minutes
The downside is first and last names are not grouped together anymore:
ie) 1. Jennifer 2. Lawrence instead of 1. Jennifer Lawrence
----------------------

Removed goldenglobes.json from the directory
download it from the zip 

