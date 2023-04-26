import os
import sys
import pandas as pd
import nltk
from nltk.stem import *
import gensim
import pickle
import spacy
import numpy as np
import PIL
import PIL.Image
import PIL.ImageFont
import PIL.ImageOps
import PIL.ImageDraw

class MyLDA:
    nlp = None
    lem = WordNetLemmatizer()
    stopwords = None

    def text_preprocessing_df(self, df):
        nltk.download('punkt')
        nltk.download('wordnet')
        nltk.download('stopwords')
        nltk.download('omw-1.4')
        
        # stopwords
        stopwords = nltk.corpus.stopwords.words('english')
        stopwords.extend(['from', 'subject', 're', 'edu', 'use'])
        
        corpus = []
        for text in df['text']:
            words = [w for w in nltk.tokenize.word_tokenize(text) if (w not in stopwords)]
#            words = [self.lem.lemmatize(w) for w in words if len(w)>3]
            corpus.append(words)

        corpus = self.lemmatization(corpus);        
        return corpus

    def text_preprocessing_texts(self, texts):
        nltk.download('punkt')
        nltk.download('wordnet')
        nltk.download('stopwords')
        nltk.download('omw-1.4')
        
        # stopwords
        stopwords = nltk.corpus.stopwords.words('english')
        stopwords.extend(['from', 'subject', 're', 'edu', 'use'])
        
        corpus = []
        for text in texts:
            words = [w for w in nltk.tokenize.word_tokenize(text) if (w not in stopwords)]
#            words = [self.lem.lemmatize(w) for w in words if len(w)>3]
            corpus.append(words)

        corpus = self.lemmatization(corpus);        
        return corpus

    def lemmatization(self, texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        texts_out = []
        nlp = spacy.load("en_core_web_sm")
        for text in texts:
            if len(text)<4:
                continue
            doc = nlp(" ".join(text)) 
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out

    def do_LDA_analysis_texts(self, texts, num_topics, feature, corpus, bow_corpus, dic):
        if len(texts) > 0:
            corpus = self.text_preprocessing_texts(texts)

            bigram = gensim.models.Phrases(corpus, min_count=5, threshold=100)
            trigram = gensim.models.Phrases(bigram[corpus], threshold=100)
            bigram_mod = gensim.models.phrases.Phraser(bigram)
            trigram_mod = gensim.models.phrases.Phraser(trigram)        
            
            bigram_texts = [bigram_mod[doc] for doc in corpus]
            trigram_texts = [trigram_mod[bigram_mod[doc]] for doc in corpus]

            corpus = self.lemmatization(bigram_texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])  

            dic = gensim.corpora.Dictionary(corpus)
            
            bow_corpus = [dic.doc2bow(doc) for doc in corpus]
#       else:
#           corpus, bow_corpus, dic, num_topics MUST be valid

        num_passes=10
        num_workers=32

        # count the words
        words = {}
        for wordList in corpus:
            for t in wordList:
                if t not in words:
                    words[t] = 0
                words[t] = words[t] + 1
        num_words = len(words.keys())

        maxWordCount = 0
        for k in words.keys():
            if words[k] > maxWordCount:
                maxWordCount = words[k]

        # count

        lda_model = gensim.models.LdaMulticore(bow_corpus, 
                                               num_topics=num_topics, 
                                               id2word=dic, 
                                               passes=num_passes, 
                                               workers=num_workers, 
                                               per_word_topics=True)
        for idx, topic in lda_model.print_topics(num_words=3):
            print('Topic: {} \nWords: {}'.format(idx, topic))

        # coherence rubric
        #    0.3 is bad
        #    0.4 is low
        #    0.55 is okay
        #    0.65 might be as good as it is going to get
        #    0.7 is nice
        #    0.8 is unlikely and
        #    0.9 is probably wrong

        from gensim.models import CoherenceModel

        # instantiate topic coherence model
        cm = CoherenceModel(model=lda_model, corpus=bow_corpus, texts=corpus, coherence='c_v')

        # get topic coherence score
        coherence_lda = cm.get_coherence()

        print('\n*** Coherency score: ', coherence_lda)

        if coherence_lda >= 0.65 and coherence_lda <= 0.7:
            print('*** Coherence score is acceptable')
        else:
            import matplotlib.pyplot as plt

            topics = []
            score = []
            bestTopicScore = coherence_lda
            topicCount = -1

            # 2 to n topics only
            for i in range(2, num_topics, 1):
                lda = gensim.models.LdaMulticore(corpus=bow_corpus, 
                                                 id2word=dic, 
                                                 iterations=10, 
                                                 num_topics=i, 
                                                 workers=num_workers, 
                                                 passes=num_passes, 
                                                 random_state=42, 
                                                 per_word_topics=True)
                cm = CoherenceModel(model=lda, corpus=bow_corpus, texts=corpus, coherence='c_v')
                topics.append(i)
                topicScore = cm.get_coherence()
                score.append(topicScore)
                
                print('Topic count: ', i, ' Score: ', topicScore)

                if topicScore > bestTopicScore and topicScore <= 0.7:
                    bestTopicScore = topicScore
                    topicCount = i

            plt.plot(topics, score)
            plt.xlabel('# of topics')
            plt.ylabel('Coherence Score')
            plt.savefig(feature + '_coherence_score.png')

            print('Best score: ', bestTopicScore)
            if bestTopicScore <= coherence_lda:
                print('Cannot find a topic with better coherence score')
                topicCount = num_topics
            else:
                print('Best number of topics: ', topicCount)
                print('Retrying ...')

                lda_model = gensim.models.LdaMulticore(bow_corpus, 
                                                       num_topics=topicCount, 
                                                       id2word=dic, 
                                                       passes=num_passes, 
                                                       workers=num_workers, 
                                                       per_word_topics=True)

                for idx, topic in lda.print_topics(num_words=3):
                    print('Topic: {}\nWords: {}'.format(idx, topic))

            print('*** Visualizing ...')

            # *** topics
            df_topic_keywords = self.format_topics_sentences(lda_model, bow_corpus, corpus)
            df_topic = df_topic_keywords.reset_index()
            df_topic.columns = ['DocNo', 'Topic', 'Contribution', 'Keywords', 'Text']
            topTopics = df_topic.head(topicCount)
            with open(feature + '_topics.txt', 'w') as f:
                dfStr = topTopics.to_string(header=True, index=True)
                f.write(dfStr)

            # *** sorted topics
            pd.options.display.max_colwidth = 100
            df_topics_sorted = pd.DataFrame()
            df_topics_grpd = df_topic_keywords.groupby('Topic')
            for i, grp in df_topics_grpd:
                df_topics_sorted = pd.concat([df_topics_sorted, 
                                              grp.sort_values(['Contribution'], 
                                              ascending=False).head(1)], 
                                              axis=0)
            df_topics_sorted.reset_index(drop=True, inplace=True)
            df_topics_sorted.columns = ['Topic', 'Contribution', 'Keywords', 'Text']
            topTopics = df_topics_sorted.head(topicCount)

            with open(feature + '_topics_sorted.txt', 'w') as f:
                dfStr = topTopics.to_string(header=True, index=True)
                f.write(dfStr)

            # *** word distribution
            doc_lens = [len(d) for d in df_topic.Text]
            maxLen = max(doc_lens)
            minLen = min(doc_lens)
            x = maxLen * 0.8
            y = 13

            plt.figure()
            plt.hist(doc_lens, histtype='stepfilled')
            plt.gca().set(ylabel='Number of Documents', xlabel='Document Word Count')
            plt.xticks([x for x in range(minLen, maxLen, int((maxLen-minLen)/10))])
            plt.yticks([y for y in range(1,y)])
            plt.tick_params(axis='x', labelsize=8, labelrotation=75)
            plt.text(x, y*0.86, "  Mean : " + str(round(np.mean(doc_lens))), family='monospace')
            plt.text(x, y*0.82, "Median : " + str(round(np.median(doc_lens))), family='monospace')
            plt.text(x, y*0.78, " Stdev : " + str(round(np.std(doc_lens))), family='monospace')
            plt.text(x, y*0.74, " 1%ile : " + str(round(np.quantile(doc_lens, q=0.01))), family='monospace')
            plt.text(x, y*0.70, "99%ile : " + str(round(np.quantile(doc_lens, q=0.99))), family='monospace')
            plt.title('Distribution Word Counts', fontdict=dict(size=18))
            plt.savefig(feature + "_dist_word_counts.png")

            # *** dominant word
            import seaborn as sns
            import matplotlib.colors as mcolors

            import math
            num_rows = int(math.sqrt(topicCount))
            num_cols = int(topicCount/num_rows)

            fig, axes = plt.subplots(num_rows, num_cols, figsize=(16,14), dpi=160, sharex=True, sharey=True)

            for i, ax in enumerate(axes.flatten()):    
                df_topic_sub = df_topic.loc[df_topic.Topic==i, :]
                doc_lens = [len(d) for d in df_topic_sub.Text]
                ax.hist(doc_lens)
                ax.tick_params(axis='y')
#                sns.kdeplot(doc_lens, color="black", shade=False, ax=ax.twinx())
                ax.set(xlabel='Document Word Count')
                ax.set_ylabel('Number of Documents')
                topicNum = '--'
                if i in df_topic.Topic.values:
                    topicNum = str(i)
                ax.set_title('Topic: ' + topicNum, fontdict=dict(size=16))

            fig.tight_layout()
            fig.subplots_adjust(top=0.90)
            plt.xticks([x for x in range(minLen, maxLen, int((maxLen-minLen)/10))])
            fig.suptitle('Distribution of Document Word Counts by Dominant Topic', fontsize=22)
            plt.savefig(feature + "_dist_word_dominant.png")
        
            # *** word cloud
            from matplotlib import pyplot as plt
            from wordcloud import WordCloud, STOPWORDS
            import matplotlib.colors as mcolors

            cloud = WordCloud(stopwords=self.stopwords,
                              background_color='black',
                              width=4000,
                              height=3000)
            topics = lda_model.show_topics(num_topics=topicCount, formatted=False)

            text = ""
            for i in range(len(topics)):
                for t in topics[i][1]:
                    text = text + ' ' + t[0]
            cloud.generate(text)
            plt.figure()
            plt.imshow(cloud, interpolation='bilinear')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(feature + "_wordcloud.png")

            # *** word importance
            out = []
            maxWeight = 0.0
            for i, topic in topics:
                for word, weight in topic:
                    if weight > maxWeight:
                        maxWeight = weight
                    out.append([word, i , weight])
            df = pd.DataFrame(out, columns=['word', 'topic_id', 'weight'])        

            maxWeight = maxWeight * 1.25
            num_rows = int(math.sqrt(topicCount))
            num_cols = int(topicCount/num_rows)
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(16,10), sharey=True, dpi=160)
            for i, ax in enumerate(axes.flatten()):
                ax.bar(x='word', height="weight", data=df.loc[df.topic_id==i, :], width=0.2, label='Weights')
                ax.set_ylim(0, maxWeight);
                topicNum = '--'
                if i in df_topic.Topic.values:
                    topicNum = str(i)
                ax.set_title('Topic: ' + topicNum, fontsize=16)
                ax.tick_params(axis='y', left=False)
                ax.set_xticklabels(df.loc[df.topic_id==i, 'word'], rotation=30, horizontalalignment= 'right')
                ax.legend(loc='upper right')
            fig.tight_layout(w_pad=1)    
            fig.suptitle('Word Count and Importance of Topic Keywords', fontsize=22, y=1.05)    
            plt.savefig(feature + "_word_importance.png")

            # *** LDA vis
            import pyLDAvis
            import pyLDAvis.gensim_models

            vis = pyLDAvis.gensim_models.prepare(lda_model, bow_corpus, dic, sort_topics=False)
            pyLDAvis.save_html(vis, feature + '_topics.html')

    def format_topics_sentences(self, lda_model, corpus, texts):
        # Init output
        topics_df = pd.DataFrame()

        # Get main topic in each document
        for i, row_list in enumerate(lda_model[corpus]):
            row = row_list[0] if lda_model.per_word_topics else row_list            
            # print(row)
            row = sorted(row, key=lambda x: (x[1]), reverse=True)
            # Get the Topic, Contribution and Keywords for each document
            for j, (topic_num, prop_topic) in enumerate(row):
                if j == 0:  # => dominant topic
                    wp = lda_model.show_topic(topic_num)
                    topic_keywords = ", ".join([word for word, prop in wp])
                    df = pd.DataFrame({'Topic' : topic_num, 
                                       'Contribution' : round(prop_topic,4), 
                                       'Keywords' : topic_keywords}, index=[i])
                    topics_df = pd.concat([topics_df, df])
                else:
                    break

        topics_df.columns = ['Topic', 'Contribution', 'Keywords']
#        print(topics_df)

        # Add original text to the end of the output
        contents = pd.Series(texts)
        topics_df = pd.concat([topics_df, contents], axis=1)
        return(topics_df)
    
