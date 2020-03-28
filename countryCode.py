

import tweepy
import csv
import pandas as pd 


#define your keys and tokens based on twitter credintials 

consumer_key = 'xxxxxxx'
consumer_secret = 'xxxxxxxx'
access_token = 'xxxxxxx'
access_token_secret = 'xxxxxxxx'
#establish connection to twitter 
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth,wait_on_rate_limit=True)
print("print after athuntication")

#creat a file to save the collected tweets 
#If you want to rewrite over the file then keep the parameter 'w' as is 
#if you want to apppend to a previous file change 'w' to 'a'
csvFile = open('coffee_Morocco_march22.csv', 'w', encoding="utf8")
csvWriter = csv.writer(csvFile)


#the search terms in Arabic it's different how you write it inside the code and 
#I had to define a variable for each word then add it to a list rather than write it directly into the list
term1 = "قهوه"
term2 = 'كاريبو'
term3 = 'كوفي'
term4 = 'ستاربكس'
term5 = 'مقهى'
term6 = 'نسكافيه'
term7 = 'كابتشينو'
term8 = 'لاتيه'
term9 = 'اسبريسو'


search_terms = [term1, term2, term3, term4, term5, term6, term7, term8, term9]

#I created a csv file that contains a list of Morocco cities and their coordinates 
data = pd.read_csv("MoroccoCitiesList.csv") 
print (data.values)
cities = data.values

ids = set()
num = 0 #I defined this varible since I noticed that the id sometimes doesn't update as it should  
print("print before loop")
#loop over each city and collect tweets related to our key words
for e in range(len(cities)):
    print('------------City------------------ ' + str(e+1) + '  ' +str(cities[e][0]))
    #the parameter for tweepy cursor I have to define the latitude and the longtitude in the geocode
    #so I created a lat and long varible and defined it out side
    lat = str(cities[e][1])
    long = str(cities[e][2])
    geo = lat +','+ long + ',50km'
    print('------------geo code------------------ ' + str(geo))
    for x in search_terms:
        print(x)
        for tweet in tweepy.Cursor(api.search, q=x,lang="ar", tweet_mode="extended" , geocode=geo).items(10000):
            if (not tweet.retweeted) and ('RT @' not in tweet.full_text):
                print (tweet.created_at, tweet.full_text) 
                csvWriter.writerow([tweet.created_at, tweet.full_text])
                ids.add(tweet.id) # add new id
                num = num +1
                print ("number of unique ids seen so far: {}",format(len(ids)))
                print ("number of unique ids seen so far: {}",num)
                print("print inside loop")
csvFile.close()

print("print after loop")

