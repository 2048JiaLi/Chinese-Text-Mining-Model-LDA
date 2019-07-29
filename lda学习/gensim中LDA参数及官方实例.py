'''
在实际的运用中，LDA可以直接从gensim调，主要的一些参数有如下几个：

corpus：语料数据，需要包含单词id与词频
num_topics：我们需要生成的主题个数（重点调节）
id2word：是一种id到单词的映射（gensim也有包生成）
passes：遍历文本的次数，遍历越多越准备
alpha：主题分布的先验
eta：词分布的先验
'''

'''接下来，我们实战一把，直接用其官方的示例'''
from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel

# Create a corpus from a list of texts
common_dictionary = Dictionary(common_texts)
common_corpus = [common_dictionary.doc2bow(text) for text in common_texts]

#Train the model on the corpus
lda = LdaModel(common_corpus,num_topics=10)

'''一步步拆解来看，首先common_texts是list形式，里面的每一个元素都可以认为是一篇文档也是list结构：'''
print(type(common_texts))
print(common_texts[0])
'''第二步，doc2bow这个方法用于将文本转化为词袋形式，看一个官方的示例大家应该就能明白了，'''
from gensim.corpora import Dictionary
dct = Dictionary(["máma mele maso".split(), "ema má máma".split()])
print(dct.doc2bow(["this","is","máma"]))
print(dct.doc2bow(["this", "is", "máma"], return_missing=True))
'''初始化的时候对每一个词都会生成一个id，新的文本进去的时候，返回该文本每一个词的id，和对应的频数，对于那些不存在原词典的，可以控制是否返回。
此时生成的corpus就相当于是LDA训练模型的输入了，让我们检查一下：'''
print(common_corpus[0])
# human单词的id为0，且在第一个文档中只出现了一次
'''最后一步，我们只需调用LDA模型即可，这里指定了10个主题。'''
from gensim.models import LdaModel
lda = LdaModel(common_corpus, num_topics=10)
'''让我们检查一下结果（还有很多种方法大家可以看文档），比如我们想看第一个主题由哪些单词构成：'''
print(lda.print_topic(1, topn=2))
'''可以看出第一个模型的词分布，9号10号占比较大（这里topn控制了输出的单词个数，对应的单词可以通过之前生成dict找出）
我们还可以对刚才生成的lda模型用新语料去进行更新，'''
'''
# 能更新全部参数
lda.update(other_corpus)
#还能单独更新主题分布， 输入为之前的参数，其中rho指学习率
lda.update_alpha(gammat, rho)
#还能单独更新词分布
lda.update_eta(lambdat, rho)
'''