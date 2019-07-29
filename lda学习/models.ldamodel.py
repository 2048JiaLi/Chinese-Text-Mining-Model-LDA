#https://www.jianshu.com/p/74ec7d5f6821

'''Usage examples'''
'''Train an LDA model using a Gensim corpus'''
from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel
# Create a corpus from a list of texts
common_dictionary = Dictionary(common_texts)
#print(common_dictionary)
common_corpus = [common_dictionary.doc2bow(text) for text in common_texts]

# Train the model on the corpus.
lda = LdaModel(common_corpus, num_topics=10)
print(lda)
print(lda.inference(common_corpus))

'''Save a model to disk, or reload a pre-trained model'''
'''
from gensim.test.utils import datapath

#Save model to disk
temp_file = datapath('model')
lda.save(temp_file)

#Load a potentially pretrained model from disk.
lada = LdaModel.load(temp_file)
'''

'''Query, the model using new, unseen documents'''
# Create a new corpus, made of previously unseen documents.
other_texts = [['computer', 'time', 'graph'],
                ['survey', 'response', 'eps'],
                ['human', 'system', 'computer']
                ]

other_corpus = [common_dictionary.doc2bow(text) for text in other_texts]
unseen_doc = other_corpus[0]
vector = lda[unseen_doc]  # get topic probability distribution for a document
#print(vector)

'''Update the model by incrementally training on the new corpus'''
lda.update(other_corpus)
vector = lda[unseen_doc]

'''A lot of parameters can be tuned to optimize training for your specific case'''
lda = LdaModel(common_corpus, num_topics=50, alpha='auto', eval_every=5)  # learn asymmetric alpha from data

#class gensim.models.ldamodel.LdaModel(corpus=None, num_topics=100, id2word=None,
#                                        distributed=False, chunksize=2000, passes=1, update_every=1, 
#                                        alpha='symmetric', eta=None, decay=0.5, offset=1.0, eval_every=10, 
#                                        iterations=50, gamma_threshold=0.001, minimum_probability=0.01, random_state=None, 
#                                       ns_conf=None, minimum_phi_value=0.01, per_word_topics=False, callbacks=None, dtype=<type 'numpy.float32'>)¶