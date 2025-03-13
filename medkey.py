from bs4 import BeautifulSoup as bs

with open('medical_keywords.html', 'r')  as f:
    soup = bs(f, 'html.parser')

divs = soup.find_all('div', {"class":"pageListPos"})

keyw = []
for div in divs:
    keyw.append(div.a.text+'\n')

with open('keywords.txt', 'w') as f:
    f.writelines(keyw)
