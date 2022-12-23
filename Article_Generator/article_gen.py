import pandas as pd
import string
import re
import nltk
import itertools
import time
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("punkt")
nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import RegexpTokenizer
from collections import Counter
import json
from joblib import load
from bs4 import BeautifulSoup
import requests
from twython import Twython, TwythonRateLimitError
from t5_common_gen import T5SentenceGeneratorCG
from gpt_2 import GPT2ArticleGenerator
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class CleanSentences():
    def __init__(self, tweets, typeOfClean="senti"):
        self.tweets = tweets
        df = self.tweets.copy(deep=True)
        # extract only the text and target polarity
        # dataset=df[['text','target']]
        dataset = df[['text', 'trend_name']].copy(deep=True)
        # print(dataset.head)

        # update positive to be 1 for simplicity
       # dataset.loc[dataset['target'] == 4, 'target'] = 1

        # convert text so lowercase
        if typeOfClean == "senti":
            dataset['text'] = dataset['text'].str.lower()

        # Defining all english stop words dataset
        stopwordlist = ['a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an',
                        'and', 'any', 'are', 'as', 'at', 'be', 'because', 'been', 'before',
                        'being', 'below', 'between', 'both', 'by', 'can', 'd', 'did', 'do',
                        'does', 'doing', 'down', 'during', 'each', 'few', 'for', 'from',
                        'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here',
                        'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
                        'into', 'is', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma',
                        'me', 'more', 'most', 'my', 'myself', 'now', 'o', 'of', 'on', 'once',
                        'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'own', 're',
                        's', 'same', 'she', "shes", 'should', "shouldve", 'so', 'some', 'such',
                        't', 'than', 'that', "thatll", 'the', 'their', 'theirs', 'them',
                        'themselves', 'then', 'there', 'these', 'they', 'this', 'those',
                        'through', 'to', 'too', 'under', 'until', 'up', 've', 'very', 'was',
                        'we', 'were', 'what', 'when', 'where', 'which', 'while', 'who', 'whom',
                        'why', 'will', 'with', 'won', 'y', 'you', "youd", "youll", "youre",
                        "youve", 'your', 'yours', 'yourself', 'yourselves']

        self.STOPWORDS = set(stopwordlist)
        dataset['text'] = dataset['text'].apply(
            lambda x: self.clean_handles(x))

        # Cleaning and removing punctuation
        punctuation_list = string.punctuation
        self.translator = str.maketrans('', '', punctuation_list)
        if typeOfClean == "senti":
            dataset['text'] = dataset['text'].apply(
                lambda x: self.clean_stopwords(x))
        else:
            dataset['text'] = dataset['text'].apply(
                lambda x: self.clean_wspace(x))

        dataset['text'] = dataset['text'].apply(
            lambda x: self.clean_punctuation(x))

        # Cleaning and removing greater than 2 character repetitiosn
        dataset['text'] = dataset['text'].apply(
            lambda x: self.clean_repeat_chars(x))

        # Cleaning and remvoing URLs
        dataset['text'] = dataset['text'].apply(lambda x: self.clean_URLs(x))

        # Clean numerical chars
        dataset['text'] = dataset['text'].apply(lambda x: self.clean_nums(x))

        if typeOfClean == "senti":
            # Acquire tokenization of tweet text
            tokenizer = RegexpTokenizer(r'\w+')
            dataset['text'] = dataset['text'].apply(tokenizer.tokenize)

            # Stem & Lemmatize data
            self.st = nltk.PorterStemmer()
            dataset['text'] = dataset['text'].apply(
                lambda x: self.stem_text(x))

            self.lm = nltk.WordNetLemmatizer()
            dataset['text'] = dataset['text'].apply(
                lambda x: self.lemmatize_text(x))
        self.final_text = dataset
        # print(dataset.head)

    def clean_stopwords(self, text):
        return " ".join([word for word in str(text).split() if word not in self.STOPWORDS])

    def clean_wspace(self, text):
        return text.lstrip()

    def clean_punctuation(self, text):
        return text.translate(self.translator)

    def clean_repeat_chars(self, text):
        # return re.sub(r'(.)1+', r'1', text)
        return re.sub(r'(.)\1{2,}', r'\1\1', text)

    def clean_URLs(self, text):
        return re.sub('((www.[^s]+)|(https?://[^s]+))', ' ', text)

    def clean_nums(self, text):
        return re.sub('[0-9]+', '', text)

    def stem_text(self, text):
        return [self.st.stem(word) for word in text]

    def lemmatize_text(self, text):
        return [self.lm.lemmatize(word) for word in text]

    def clean_handles(self, text):
        return re.sub('@[^\s]+', '', text)

    def get_cleaned_set(self):
        return self.final_text
    def sentiment_vader_score(self, sent):
        sid_obj = SentimentIntensityAnalyzer()
        sentiment_dict = sid_obj.polarity_scores(sent)
        # return sentiment score from vader
        return sentiment_dict["pos"]
    def generate_sentiment(self):
        sentiment_df = pd.DataFrame(
            columns=["text", "LRSenti", "vaderSenti"])
        sentiment_df["text"] = self.tweets["text"]
        sentiment_df = sentiment_df.reset_index(drop = True)
        for i in range(sentiment_df.shape[0]):
            curr_text = sentiment_df["text"][i]
            sentiment_df.loc[i,"vaderSenti"] = self.sentiment_vader_score(curr_text)
        # space delimits text for correct format for TF-IDF vectoriser
        text_s = [" ".join(tweet) for tweet in self.final_text['text'].values]
        X = vectoriser.transform(text_s)
        y_predLR = LRmodel.predict(X)

        sentiment_df["LRSenti"] = y_predLR
        sentiment_df["avgSenti"] = (sentiment_df["LRSenti"] + sentiment_df["vaderSenti"]) / 2
        return sentiment_df


def get_articles(asset):
    # searches google rss feed
    productUrl = 'https://news.google.com/rss/search?q=intitle:{}&hl=en-GB&gl=GB&ceid=GB%3Aen'.format(
        asset)
    print(productUrl)
    articlearray = []
    res = requests.get(productUrl)
    res.raise_for_status()
    # extracts each article
    soup = BeautifulSoup(res.text, features="xml")
    count = 0
    for news in soup.find_all("item"):
        # perform basic sanitization:
        for t in news.select('script, noscript, style, iframe, nav, footer, header'):
            t.extract()
        title = news.title.text.strip()
        articlearray.append([title])
        count += 1
    links = []
    for i in range(len(articlearray)):
        title = articlearray[i][0]
        links.append(title)
    return links

# import SentimentIntensityAnalyzer class
# from vaderSentiment.vaderSentiment module.


# function to print sentiments
# of the sentence.
def sentiment_scores(sentence):

    sid_obj = SentimentIntensityAnalyzer()

    sentiment_dict = sid_obj.polarity_scores(sentence)

    return sentiment_dict["pos"]

class TrendProcessor():
    def __init__(self):
        self.names = []
        self.urls = []
        self.article_titles = []
        self.articles = []
        self.imgURLs = []
        self.trends_to_skip = []
        self.trends_passed = 0
        with open("twitter_credentials.json", "r") as file:
            self.creds = json.load(file)
    
        self.twitter = Twython(self.creds['APP_KEY'], self.creds['APP_SECRET'])
        # UK unique trending topics
        response = self.twitter.get_place_trends(id=23424975)[0]["trends"]
        self.unique_trends_df = pd.DataFrame(response).drop_duplicates()
        self.trends = list(self.unique_trends_df["name"])
        print(*trends, sep = "\n")
    def process_initial_tweets(self,trend_name, filter_val):
        # non local so there is link to outer function variables
        tweet_closest_sentiment_image = ""

        # gets 100 most recent tweets regarding trend
        print("Processing {}... ".format(trend_name))
        # filter_val is -filter:retweeets = don't include retweets
        query = {'q': trend_name + ' ' + filter_val,
                'count': 100,
                'result_type': 'recent',
                'lang': 'en',
                'tweet.fields': 'public_metrics'
                }
        try:
            tweets = self.twitter.search(**query)["statuses"]
        except TwythonRateLimitError as error:
            remainder = float(self.twitter.get_lastfunction_header(header='x-rate-limit-reset')) - time.time()
            del self.twitter
            print("Rate limited, Pausing for "+ str(int(remainder)) +  " seconds...")
            # adds three seconds to ensure that code doesn't crash due to any slight miscalculation
            time.sleep(remainder + 3)
            self.twitter = Twython(self.creds['APP_KEY'], self.creds['APP_SECRET'])
            tweets = self.twitter.search(**query)["statuses"]
        finally:
            # process tweet set here
            if len(tweets) == 0:
                average_sentiment = 0
                tweet_closest_sentiment_image = "invalid_trend"
            else:
                tweet_set = pd.DataFrame(columns=["trend_name", "text"])
                for i in range(len(tweets)):
                    text = tweets[i]["text"]
                    new_series = pd.DataFrame(
                        pd.Series({"text": text, "trend_name": trend_name})).transpose()
                    tweet_set = pd.concat([tweet_set, new_series])

                # cleans tweets and generates sentiment for the set of tweets collected
                TweetCleaner = CleanSentences(tweet_set)
                sentiment = TweetCleaner.generate_sentiment().reset_index()

                # sentiment taken as average of Vader and best performing sentiment AI model
                average_sentiment = sentiment["avgSenti"].mean()
                print("Average Sentiment : " +
                    str(average_sentiment))
                    # search for most suitable image
                for i in range(sentiment.shape[0]):
                    tweet_closest_sentiment_val = 1
                    diff = abs(
                        sentiment.loc[i, "avgSenti"] - average_sentiment)
                    if diff < tweet_closest_sentiment_val and "media" in tweets[i]["entities"]:
                        possible_extension = tweets[i]["entities"]["media"][0]["media_url"][-3:]
                        if possible_extension == "jpg" or possible_extension == "png":
                            tweet_closest_sentiment_image = tweets[i]["entities"]["media"][0]["media_url"]
                            tweet_closest_sentiment_val = diff
                if tweet_closest_sentiment_image == "":
                    average_sentiment, tweet_closest_sentiment_image = self.process_initial_tweets(trend_name,
                        "-filter:retweets filter:images")
        return (average_sentiment, tweet_closest_sentiment_image)   
    def process_larger_tweet_set(self, trend_name):
        done = False
        counter = 0
        #  analyse larger set of tweets
        tweet_set_large = pd.DataFrame(columns=["trend_name", "text"])
        while done == False:
            query = {'q': trend_name + ' -filter:retweets',
                     'count': 100,
                     'result_type': 'latest',
                     'lang': 'en',
                     'tweet.fields': 'public_metrics'
                     }
            try:
                tweets_large = self.twitter.search(**query)["statuses"]
            except TwythonRateLimitError as error:
                remainder = float(self.twitter.get_lastfunction_header(header='x-rate-limit-reset')) - time.time()
                del self.twitter
                print("Rate limited, Pausing for "+ str(int(remainder)) +  " seconds...")
                # 3 seconds added to be safe if there are issues with rate limit reset value from the origin
                time.sleep(remainder + 3)
                self.twitter = Twython(self.creds['APP_KEY'], self.creds['APP_SECRET'])
                tweets_large = self.twitter.search(**query)["statuses"]
            finally:
                for i in range(len(tweets_large)):
                    if tweets_large[i]["text"] != "":
                        if tweets_large[i]["text"][0:2] == "RT":
                            text = tweets_large[i]["text"][3:]
                        else:
                            text = tweets_large[i]["text"]
                        retweets = tweets_large[i]['retweet_count']
                        # set to -1 as we dont care for retweet count at this moment due to lack of tweets
                        # popular "result_type" not working well enough
                        if retweets > -1:
                            new_series = pd.DataFrame(
                                pd.Series({"text": text, "trend_name": trend_name})).transpose()
                            tweet_set_large = pd.concat(
                                [tweet_set_large, new_series])
            tweet_set_large = tweet_set_large.drop_duplicates(
                subset=['text'], keep='first')
            # cumulates tweets until dataset is large enough or before rate limits potentially kick in
            if (tweet_set_large.shape[0] > 200) or (counter > 70):
                done = True
                print("done looking for new tweets")
                # notsenti parameter so that it doesn't clean stopwords etc.
                TweetCleaner = CleanSentences(tweet_set_large, "notsenti")
                # sends tweets through cleaner
                final_set_large_tweets = TweetCleaner.get_cleaned_set()
            counter = counter + 1
        return final_set_large_tweets
    def find_adjective(self, trend_name, average_sentiment, tweet_adjectives):
        for ta in range(len(tweet_adjectives)):
            tweet_adjectives[ta] = tweet_adjectives[ta].title()
        # same format, drop duplicates
        tweet_adjectives = list(dict.fromkeys(tweet_adjectives))
        adject_df = pd.DataFrame(columns = ["text"], data = tweet_adjectives)
        adject_df["trend_name"] = trend_name
        TweetCleaner = CleanSentences(adject_df)
        sentiment_adjectives = TweetCleaner.generate_sentiment()
        closest_senti_adjective_index = 0
        closest_senti_adjective_val = 0
        for i in range(sentiment_adjectives.shape[0]):
            if closest_senti_adjective_val > abs(sentiment_adjectives.loc[i, "avgSenti"] - average_sentiment):
                closest_senti_adjective_val = abs(sentiment_adjectives.loc[i, "avgSenti"] - average_sentiment)
                closest_senti_adjective_index = i
        relevant_adjective = sentiment_adjectives.loc[closest_senti_adjective_index, "text"]
        return relevant_adjective
    def find_headlines(self,nouns):
        # search for articles from last few hours
        # tried to get most relevant articles by searching in the last two hours
        headlines = []
        # uses top four most common nouns to search
        nums = [0,1,2,3]
        combinations = []
        # generate combinations
        for l in range(len(nums)+1):
            for combination in itertools.combinations(nums, l):
                combinations.append(combination)
        # remove empty set
        combinations.pop(0)
        # for each parameter left
        for c in range(len(combinations)):
            combs =  list(combinations[c])
            # only take parameters that have a length of 3 or 4
            if len(combs) > 2:
                parameter = ""
                # format parameter for Google RSS search
                for i in range(len(combs)):
                    if parameter == "":
                        parameter = nouns[combs[i]]
                    else:
                        parameter = parameter + " + " +  nouns[combs[i]]
                # searches nouns
                print("Searching articles with {}...".format(parameter))
                temp_headlines = get_articles(
                    '{} + when:5h'.format(parameter))
                # if search generated more headlines than the largest set so far
                if len(temp_headlines) > len(headlines):
                    # set as best parameter/headline set
                    headlines = temp_headlines
                    print("Best parameter so far is: " + parameter)
        return headlines
    def clean_headline_nouns(self,headlines_closest_senti, headlines_sentiment):
        # tokenises headlines that meet condition above, keeping alphanumeric characters only
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = []
        for i in range(len(headlines_closest_senti)):
            tokens = tokens + tokenizer.tokenize(headlines_sentiment.loc[headlines_closest_senti[i], "text"])
        tagged = nltk.pos_tag(tokens)
        # extract nouns/verbs from relevant headlines
        headline_nouns = [token for token, pos in tagged if pos.startswith('N')]
        #remove duplicates and punctuation, more cleaning
        remove_items = []
        for h in range(len(headline_nouns)):
            headline_nouns[h] = headline_nouns[h].title()
            no_punct = headline_nouns[h].translate(str.maketrans('', '', string.punctuation))
            if no_punct != headline_nouns[h]:
                remove_items.append(h)
        for i in sorted(remove_items, reverse = True):
            del headline_nouns[i]
        headline_nouns = list(dict.fromkeys(headline_nouns))
        headline_nouns = [x for x in headline_nouns if len(x)>=3]
        return headline_nouns
    def find_clean_common_words(self, final_set_pop_tweets):
        # finds most common words in tweets
        common = Counter(" ".join(final_set_pop_tweets["text"]).split()).most_common(100)
        words = [i[0] for i in common]
        
        # removes emojis
        emoji_pattern = re.compile("["
                u"\U0001F600-\U0001F64F"  # emoticons
                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   "]+", flags=re.UNICODE)
        for i in range(len(words)):
           words[i] = emoji_pattern.sub(r'', words[i])
                        
        tagged = nltk.pos_tag(words)
        # finds most frequently used nouns in tweet set
        nouns = [token for token, pos in tagged if pos.startswith('N')]
        # finds adjectives used on twitter
        tweet_adjectives = [token for token, pos in tagged if pos.startswith('J')]
        return nouns, tweet_adjectives
    def find_sentiment_headlines(self, trend_name, average_sentiment, headlines):
        for i in range(len(headlines)):
            index = headlines[i].rfind('-')
            if index != -1:
                result = headlines[i][0:index]
            headlines[i] = result
        # finds the sentiment of each headline
        headline_set = pd.DataFrame(headlines, columns=["text"])
        headline_set["trend_name"] = trend_name
        TweetCleaner = CleanSentences(headline_set)
        headlines_sentiment = TweetCleaner.generate_sentiment()
        headlines_closest_senti = []

        avg_senti_word = "neg"
        if average_sentiment > 0.5:
            avg_senti_word = "pos"
        elif average_sentiment == 0.5:
            avg_senti_word = "neu"

        # searches for headlines that match the twitter sentiment the closest
        for i in range(headlines_sentiment.shape[0]):
            headline_senti_word = "neg"
            if headlines_sentiment.loc[i, "avgSenti"] > 0.5:
                headline_senti_word = "pos"
            elif headlines_sentiment.loc[i, "avgSenti"] == 0.5:
                headline_senti_word = "neu"
            
            if (abs(headlines_sentiment.loc[i, "avgSenti"] - average_sentiment) < 0.2) and headline_senti_word == avg_senti_word:
                headlines_closest_senti.append(i)
        # if none of the headlines have a sentiment close to the twitter sentiment, add headlines with generally similar sentiment
        if len(headlines_closest_senti) == 0:
            print("No closely matched sentiment headlines found, searching for headlines with generally similar headlines..")
            for i in range(headlines_sentiment.shape[0]):
                if (headline_senti_word == avg_senti_word) or (abs(headlines_sentiment.loc[i, "avgSenti"] - average_sentiment) < 0.2):
                    headlines_closest_senti.append(i)
        return headlines_sentiment, headlines_closest_senti
    def text_generation(self, trend_name, headline_nouns, relevant_adjective,tweet_closest_sentiment_image):
        noun_list = ' '.join(headline_nouns[:4])
        token_list = "{} {}".format(noun_list, relevant_adjective) 
        # T5 used to generate sentence
        print("Words fed into sentence generator: {}".format(token_list))
        SentenceGen = T5SentenceGeneratorCG(token_list)
        SentenceGen.generate()
        generated_sentence = SentenceGen.get_sentence()
        print("sentence generated: {}".format(generated_sentence))

        # GPT-2 used to generate large article
        ArticleGen = GPT2ArticleGenerator(generated_sentence.capitalize())
        ArticleGen.generate()
        generated_article = ArticleGen.get_article()
        generated_article = generated_article.replace("<|endoftext|>", "")

        print("{} Article {}: {} ".format(trend_name, len(self.names) + 1, generated_article))
        self.names.append(trend_name)
        self.urls.append(self.unique_trends_df.loc[self.trends_passed, "url"])
        self.article_titles.append(generated_sentence.capitalize())
        self.articles.append(generated_article)
        self.imgURLs.append(tweet_closest_sentiment_image)
    def trend_skipper(self, headline_nouns):
        for i in range(len(self.trends)- self.trends_passed):
            if self.trends[i+ self.trends_passed] in headline_nouns and i not in self.trends_to_skip:
                print("will skip trend {} to prevent similar articles".format(self.trends[i + self.trends_passed]))
                self.trends_to_skip.append(i)    
    def process_trends(self):
        self.trends_passed = 0
        finished = False
        # until 5 trend articles are obtained or the end of the trend list is reached
        while finished == False:
            trend_name = self.trends[self.trends_passed]
            if self.trends_passed in self.trends_to_skip:
                print("Skipping {}".format(trend_name))
            else: 
                #process initial tweet set
                average_sentiment, tweet_closest_sentiment_image = self.process_initial_tweets(trend_name, "-filter:retweets")
                if tweet_closest_sentiment_image == "invalid_trend":
                    print("No tweets found, skipping trend")
                else:
                    final_set_large_tweets = self.process_larger_tweet_set(trend_name)
                    nouns, tweet_adjectives  = self.find_clean_common_words(final_set_large_tweets)
                    if len(tweet_adjectives) == 0:
                        print("No appropriate adjectives found, skipping trend")
                    else:
                        # find most appropriate adjective
                        relevant_adjective = self.find_adjective(trend_name, average_sentiment, tweet_adjectives)        
                        #small cleaning
                        for i in range(len(nouns)):
                            nouns[i] =  re.sub('[^a-zA-Z0-9 \n\.]', '', nouns[i])
                        # removes empty strings and short nouns
                        nouns = [x for x in nouns if x]
                        nouns = [x for x in nouns if len(x)>=3]
                        nouns_to_remove = []
                        # remove links if cleaner regex hasnt
                        for n in range(len(nouns)):
                            start = nouns[n][0:4]
                            if start == "http":
                                nouns_to_remove.append(n)
                            else:
                                nouns[n] = nouns[n].title()
                        for i in sorted(nouns_to_remove, reverse = True):
                            del nouns[i]
                        # same format, drop duplicates
                        nouns = list(dict.fromkeys(nouns))

                        # if enough nouns are found from twitter
                        if len(nouns) > 3:
                            # search for articles from last few hours
                            # tried to get most relevant articles by searching in the last two hours
                            # uses top four most common nouns to search
                            headlines = self.find_headlines(nouns)
                        # if not enough headlines have been found, skip trend
                        if len(headlines) < 3:
                            print("Can't find enough articles on topic, skipping trend {}".format(trend_name))
                        else:
                            # keeps headlines with close sentiment
                            headlines_sentiment ,headlines_closest_senti = self.find_sentiment_headlines(trend_name, average_sentiment, headlines)
                            if len(headlines_closest_senti) == 0:
                                print("No headlines with remotely similar sentiment found, skipping Trend with name {}".format(trend_name))
                            else:
                                # peforms extra cleaning on headline nouns
                                headline_nouns = self.clean_headline_nouns(headlines_closest_senti, headlines_sentiment)
                                # tries to see if trend is very similar to others by looking at the nouns in the articles
                                self.trend_skipper(headline_nouns)
                                if len(headline_nouns) <= 3:
                                    print("Not enough nouns, skipping Trend with name {}".format(trend_name))
                                else:
                                    self.text_generation(trend_name,headline_nouns,relevant_adjective, tweet_closest_sentiment_image)
                    
            self.trends_passed = self.trends_passed + 1
            # check if enough articles have been generated or if the end of the trend list has been reached
            if len(self.names) == 5:
                finished = True
            if self.trends_passed >= len(self.trends):
                finished = True
        return (self.names, self.urls, self.article_titles, self.articles, self.imgURLs)
trends= []
final_json = {}
vectoriser = load('tf_idf_vec.joblib')
LRmodel = load('LRSentiment.joblib')
TP = TrendProcessor()
names, urls, article_titles, articles, imgURLs = TP.process_trends()
for i in range(len(names)):
    trends.append({"name": names[i], "url":urls[i],"title": article_titles[i], "article":articles[i],"imgURL":imgURLs[i]})
result = {"trends":trends}
print(json.dumps(result))
with open("../website/output.json", "w") as outfile:
    json.dump(result, outfile)
