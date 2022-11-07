from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import pandas as pd
import matplotlib.pyplot as plt

FINVIZ_URL = 'https://finviz.com/quote.ashx?t='
# tickers = ['AMZN', 'GOOG', 'FB']
# tickers = ['AMZN', 'GOOG', 'META']
tickers = ['GOOG']

news_tables = {}
for ticker in tickers:
    url = FINVIZ_URL + ticker
    
    req = Request(url=url, headers={'user-agent': 'my-app'}) 
    response = urlopen(req)
    
    html = BeautifulSoup(response, features='html.parser')
    news_table = html.find(id='news-table')
    news_tables[ticker] = news_table

parsed_data = []
for ticker, news_table in news_tables.items(): 
    print(ticker)
    for row in news_table.findAll('tr'):
        title= row.a.text
        date_data = row.td.text.split(' ')
        
        if len(date_data) == 1:
            time = date_data[0]
        else:
            date = date_data[0] 
            time = date_data[1]
            
        parsed_data.append([ticker, date, time, title])
        
df = pd.DataFrame(parsed_data, columns=['ticker', 'date', 'time','title'])
df['date'] = pd.to_datetime (df.date).dt.date

# nltk.download('vader_lexicon')

vader = SentimentIntensityAnalyzer()
f = lambda title: vader.polarity_scores(title)['compound'] 
df['compound'] = df['title'].apply(f)
print(df)

plt.figure(figsize=(10,6))
mean_df = df.groupby(['ticker', 'date']).mean().unstack() 
mean_df = mean_df.xs('compound', axis="columns").transpose() 
mean_df.plot(kind='bar') 
plt.show()