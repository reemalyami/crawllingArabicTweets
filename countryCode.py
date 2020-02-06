#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 16:54:18 2019
return country code
@author: rm
"""

import tweepy




consumer_key = 'xxxxxx'
consumer_secret = 'xxxx'
access_token = 'xxx'
access_token_secret = 'xxx'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth,wait_on_rate_limit=True)
print("print after athuntication")



places = api.geo_search(query="kw", granularity="city")


print(places)
place_id = places[0].id
print('kw id is: ',place_id)
