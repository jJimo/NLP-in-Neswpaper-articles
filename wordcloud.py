import pandas as pd
import matplotlib.pyplot as plt
from os import path
from wordcloud import WordCloud

#d = path.dirname(__file__)
df=pd.read_csv("train_set.csv",sep="\t")
categories=["Business","Film","Football","Politics","Technology"]
for category in categories:
	text=""
	for index,row in df.iterrows():
		if row["Category"]==category:
			text=text+row["Title"]
	
	wordcloud = WordCloud(max_font_size=40, relative_scaling=.5).generate(text)
	plt.imshow(wordcloud)
	plt.axis("off")
	fig1=plt.gcf()
	plt.show()
	fig1.savefig(category+'.png',dpi=100)






