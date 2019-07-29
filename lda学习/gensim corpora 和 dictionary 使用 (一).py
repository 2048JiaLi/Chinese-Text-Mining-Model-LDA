#https://blog.csdn.net/qq_19707521/article/details/79174533
from gensim import corpora
from gensim.corpora.dictionary import Dictionary
from collections import  defaultdict

# 停用词
stoplist=set('for a of the and to in'.split())#字符串以空格分离
#{'a', 'and', 'for', 'in', 'of', 'the', 'to'}

# 数据
documents = ["Human machine interface for lab abc computer applications",
             "A survey of user opinion of computer system response time",
             "The EPS user interface management system",
             "System and human system engineering testing of EPS",
             "Relation of user perceived response time to error measurement",
             "The generation of random binary unordered trees",
             "The intersection graph of paths in trees",
             "Graph minors IV Widths of trees and well quasi ordering",
             "Graph minors A survey"]

# 在数据中过滤掉停用词
texts=[ [ word  for word in document.lower().split() if word not in stoplist ] for document in documents]
#[['human', 'machine', 'interface', 'lab', 'abc', 'computer', 'applications'],
# ['survey', 'user', 'opinion', 'computer', 'system', 'response', 'time'],
# ['eps', 'user', 'interface', 'management', 'system'],
# ['system', 'human', 'system', 'engineering', 'testing', 'eps'],
# ['relation', 'user', 'perceived', 'response', 'time', 'error', 'measurement'],
# ['generation', 'random', 'binary', 'unordered', 'trees'],
# ['intersection', 'graph', 'paths', 'trees'],
# ['graph', 'minors', 'iv', 'widths', 'trees', 'well', 'quasi', 'ordering'],
# ['graph', 'minors', 'survey']]

# 创建统计字典，次数统计
frequency=defaultdict(int)
for text in texts:
    for token in text:
        frequency[token]+=1
'''
>>>frequency
defaultdict(int,
            {'human': 2,
             'machine': 1,
             'interface': 2,
             'lab': 1,
             'abc': 1,
             'computer': 2,
             'applications': 1,
             'survey': 2,
             'user': 3,
             'opinion': 1,
             'system': 4,
             'response': 2,
             'time': 2,
             'eps': 2,
             'management': 1,
             'engineering': 1,
             'testing': 1,
             'relation': 1,
             'perceived': 1,
             'error': 1,
             'measurement': 1,
             'generation': 1,
             'random': 1,
             'binary': 1,
             'unordered': 1,
             'trees': 3,
             'intersection': 1,
             'graph': 3,
             'paths': 1,
             'minors': 2,
             'iv': 1,
             'widths': 1,
             'well': 1,
             'quasi': 1,
             'ordering': 1})
'''

#去掉只出现一次的单词
texts=[ [ token for token in text if frequency[token] > 1 ] for text in texts]
'''
>>>texts
[['human', 'interface', 'computer'],
 ['survey', 'user', 'computer', 'system', 'response', 'time'],
 ['eps', 'user', 'interface', 'system'],
 ['system', 'human', 'system', 'eps'],
 ['user', 'response', 'time'],
 ['trees'],
 ['graph', 'trees'],
 ['graph', 'minors', 'trees'],
 ['graph', 'minors', 'survey']]
'''

# 将文档存入字典
dct = Dictionary(texts)
'''
>>>dct.id2token
{0: 'computer',
 1: 'human',
 2: 'interface',
 3: 'response',
 4: 'survey',
 5: 'system',
 6: 'time',
 7: 'user',
 8: 'eps',
 9: 'trees',
 10: 'graph',
 11: 'minors'}

>>>dct.dfs 词频
{1: 2, 2: 2, 0: 2, 4: 2, 7: 3, 5: 3, 3: 2, 6: 2, 8: 2, 9: 3, 10: 3, 11: 2}
'''

corpus = [dct.doc2bow(text) for text in texts]
#语料库 也可以这样把文档写入 allow_update=True,默认是 False
'''
>>>dct.doc2bow(['human', 'interface', 'computer'])
>>>[(0, 1), (1, 1), (2, 1)]
'''
'''对应texts与字典dct
>>>corpus
[[(0, 1), (1, 1), (2, 1)],
 [(0, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1)],
 [(2, 1), (5, 1), (7, 1), (8, 1)],
 [(1, 1), (5, 2), (8, 1)],
 [(3, 1), (6, 1), (7, 1)],
 [(9, 1)],
 [(9, 1), (10, 1)],
 [(9, 1), (10, 1), (11, 1)],
 [(4, 1), (10, 1), (11, 1)]]
'''