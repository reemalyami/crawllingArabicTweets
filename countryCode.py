#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 16:54:18 2019
return country code
@author: rm
"""

import tweepy




consumer_key = 'ekxAN0tYmXHBIXK2ZpqKm9G6K'
consumer_secret = 'lKiPbxZpHc9RT9W2cJqBiS8qMABAzyhFUWyfTnmjkFVN7qxjAR'
access_token = '585913061-MF8Ag77o9dJbVJVoS1qjJVOq8arbGVcE5ZaULxSx'
access_token_secret = 'ocl1w8LY3AK79hYhYNIAZ0cUzqlREwlEQzsn5mHQaJD49'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth,wait_on_rate_limit=True)
print("print after athuntication")



places = api.geo_search(query="kw", granularity="city")


print(places)
place_id = places[0].id
print('kw id is: ',place_id)