from bs4 import BeautifulSoup
from pyquery import PyQuery as pq
from pprint import pprint

def main():
	filename = './nominees.html'


	with open(filename,'r') as f:
		soup = BeautifulSoup(f)
		table = soup.find("table", { "class" : "wikitable" })
		movies = {}
		title1 = ''; noms1 = []
		title2 = ''; noms2 = []
		for k,row in enumerate(table.findAll('tr')):
			# if k==0 or k%3==0:
			# 	print row.find('th')
			if k==0 or k%3==0:
				noms1 = []
				noms2 = []
			if k==1 or (k-1)%3==0:
				ths = row.findAll('th')
				if ths != []:
					th1 = ths[0]; th2 = ths[1]
					link1 = th1.find('a').attrs
					title1 = link1['title']
					link2 = th2.find('a').attrs
					title2 = link2['title']
					print title1
					print title2
			if k==2 or (k-2)%3==0:
				tds = row.findAll('td')
				if tds != []:
					td1 = tds[0]; td2 = tds[1]
					for a in td1.findAll('a'):
						noms1.append(a['title'])
					for a in td2.findAll('a'):
						noms2.append(a['title'])
			
			if title1 != '' and title2 != '':
				movies[title1] = noms1
				movies[title2] = noms2

		pprint(movies)

if __name__ == "__main__":
	main()