### preprocess
#tweet-preprocessor: https://github.com/s/preprocessor
import numba
import numpy.random
import preprocessor as p
import time
import pandas as pd
import numpy as np
import re
import pysbd
from sentence_transformers import SentenceTransformer
import pickle
# PANDAS WILL DROP ROWS SILENTLY!!!  We need to set quoting=csv.QUOTE_NONE
import bz2
import csv
from tqdm import tqdm
from umap import UMAP
from hdbscan import HDBSCAN
import re
import itertools
from dateutil import parser as dateparser
import os

homedir = os.getenv("HOME")
os.chdir(homedir+"/OneDrive/manuscripts/psyop/data/")

fourChan = pd.read_csv(homedir+"4plebs_2024.tsv",sep="\t", header=None, quoting=csv.QUOTE_NONE)
fourChan = fourChan.rename(columns={0: "DATE", 1: "COUNTRY", 2: "MESSAGE"})
facebook = pd.read_csv(homedir+"oldtonew_psyop_meta/2024-01-14-09-15-26-NZDT-search-csv-export.csv", sep="\t", keep_default_na=False)
facebook['UNIXTIME'] = [round(dateparser.parse(item).timestamp()) for item in facebook['Post Created']]
insta = pd.read_csv(homedir+"oldtonew_psyop_meta/2024-01-14-09-16-36-NZDT-search-csv-export.csv", sep="\t", keep_default_na=False)
insta['UNIXTIME'] = [round(dateparser.parse(item).timestamp()) for item in insta['Post Created']]
telegram = pd.read_csv(homedir+"telegram.tsv",sep="\t", header=None, quoting=csv.QUOTE_NONE)
telegram = telegram.rename(columns={0: "USERNAME", 1: "DATE", 2: "FORWARDED_FROM_ID", 3: "MESSAGE"})

# crowdTangle has an annoying bookend issue with multiple quotes plus the quoted csv
def metaQuoteParse(text):
	text = [item.strip() for item in text]
	text = [re.sub(r'^[\"\']+|[\"\']+$', '', item) for item in text]
	text = [re.sub(r'[\t\r\n]', ' ', item) for item in text]
	return text


facebook['ALLPOSTTEXT'] = [item.strip() for item in list(map(lambda a,b,c,d: '%s %s %s %s' % (a,b,c,d), metaQuoteParse(facebook['Message']), metaQuoteParse(facebook['Description']),metaQuoteParse(facebook['Link Text']),metaQuoteParse(facebook['Image Text'])))]
insta['ALLPOSTTEXT'] = [item.strip() for item in list(map(lambda a,b,c: '%s %s %s' % (a,b,c), metaQuoteParse(insta['Title']), metaQuoteParse(insta['Description']),metaQuoteParse(insta['Image Text'])))]

combo = pd.DataFrame()
combo['DATE'] = list(itertools.chain(fourChan['DATE'],facebook['UNIXTIME'],insta['UNIXTIME'],telegram['DATE']))
combo['TEXT'] = list(itertools.chain(fourChan['MESSAGE'],facebook['ALLPOSTTEXT'],insta['ALLPOSTTEXT'],telegram['MESSAGE']))
combo['PLATFORM'] = list(itertools.chain(list(itertools.repeat("4chan", fourChan.shape[0])),
										list(itertools.repeat("Facebook", facebook.shape[0])),
										list(itertools.repeat("Instagram", insta.shape[0])),
										list(itertools.repeat("Telegram", telegram.shape[0]))))

combo.to_csv(homedir+'all_psyop_posts_combo.tsv', index=False, sep='\t', header=False, quoting=csv.QUOTE_NONE)

seg = pysbd.Segmenter(language="en", clean=False)
# in my experience HDBSCAN separates hashtags into separate clusters anyway
p.set_options(p.OPT.URL, p.OPT.MENTION, p.OPT.EMOJI, p.OPT.SMILEY, p.OPT.HASHTAG)
uniqueSentences = {}
allSentences = []
indices = []
# I'm sure there are way faster ways to do this, but whatever.
# I'm also forced to str() the text, as some fields are being returned as floats by python's Pandas, I assume.
for index in tqdm(combo.index):
	text = str(combo['TEXT'][index]).strip()
	# funky issue in TGDataset vs. preprocessor
	if ((combo['DATE'][index] == "1639495324") or (combo['DATE'][index] == "1639494448")):
		text = re.sub("bloghttps", "blog https", text)
		text = re.sub(".html...", ".html ..", text)
		text = re.sub(":https", ": https", text)
		text = re.sub("\"https", "\" https", text)
	# remove all escaped newlines in TGDataset
	text = re.sub(r'(\\n|\\t|\\r)', ' ', text)
	# remove unicode characters
	text = re.sub(r'<U\+[a-zA-Z0-9]{4,8}>', '', text)
	# remove unicode characters
	text = re.sub(r'\\[0-9]{3}', '', text)
	text = re.sub(r'\|>', ' ', text)
	text = re.sub(r'\|', ' ', text)
	text = re.sub(r'\>\>[0-9]{1,}', ' ', text)
	text = re.sub(r'\>[0-9]{1,}', ' ', text)
	text = p.clean(text)
	sentences = seg.segment(text)
	if len(sentences)==0:
		uniqueSentences.update({"": index})
		allSentences.append("")
		indices.append(index)
	else:
		for sentence in sentences:
			sentence = str(sentence).strip()
			uniqueSentences.update({sentence: index})
			allSentences.append(sentence)
			indices.append(index)

npUniqueSentences = pd.DataFrame.from_dict(uniqueSentences, orient='index')
npUniqueSentences.index.name = 'Sentence'
npUniqueSentences.reset_index(inplace=True)
npUniqueSentences.to_csv(homedir+'uniques_psyop.tsv', index=False, sep='\t', header=False, quoting=csv.QUOTE_NONE)

pd.DataFrame({'index':indices,'sentence':allSentences}).to_csv(homedir+'all_sentences_psyop.tsv', index=False, sep='\t', quoting=csv.QUOTE_NONE)

text = pd.read_csv(homedir+'uniques_psyop.tsv', keep_default_na=False, sep='\t', header=None,quoting=csv.QUOTE_NONE)
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
embedding_model = sentence_model.encode(text[0], normalize_embeddings=False, show_progress_bar=True)
with bz2.BZ2File(homedir+'embedding_psyop.pbz2', 'wb') as pkl:
	pickle.dump(embedding_model, pkl)

##


def send2UMAP(embedding_file,umap_file):
	with bz2.BZ2File(embedding_file, 'rb') as pkl:
		embedding_model = pickle.load(pkl)
	@numba.njit()
	def set_random_njit():
		np.random.seed(42)
	umap_model = UMAP(n_components=5, random_state=42, min_dist=0.0, spread=0.5, metric='cosine', verbose=True)
	set_random_njit()
	umap_model.fit(embedding_model)
	umap_embedding = umap_model.embedding_
	pd.DataFrame(umap_embedding).to_csv(umap_file, sep="\t", index=False, header=False)




def send2OpenTSNE(embedding_file,tsne_file):
	import openTSNE
	import csv
	umapped = pd.read_csv(embedding_file, keep_default_na=False, sep='\t', header=None, quoting=csv.QUOTE_NONE)
	dims = np.array(umapped)
	# https://www-nature-com.ezproxy.waikato.ac.nz/articles/s41467-019-13056-x
	if len(dims)>10000:
		# we must downsample to make init computationally feasible
		np.random.seed(42)
		shuffle = np.random.permutation(list(range(dims.shape[0])))
		reverse = np.argsort(shuffle)
		dims_shuffle = dims[shuffle]
		print("Downsampling.  Confirming unshuffle function works... " + str(np.array_equal(dims,dims_shuffle[reverse])))
		shuffle_sample = dims_shuffle[:25000]
		shuffle_rest = dims_shuffle[25000:]
		high_perplex = len(shuffle_sample) / 100
		print("30 and " + str(high_perplex))
		affinities_multiscale_mixture = openTSNE.affinity.Multiscale(shuffle_sample, perplexities=[30, high_perplex],
		                                                             metric="cosine", n_jobs=-1, random_state=42,
		                                                             verbose=True)
		init = openTSNE.initialization.pca(shuffle_sample, random_state=42, verbose=True)
		tsne_model = openTSNE.TSNE(learning_rate=len(shuffle_sample) / 12, n_jobs=8, verbose=True)
		sample_embedding_multiscale = tsne_model.fit(affinities=affinities_multiscale_mixture, initialization=init)
		rest_embedding_multiscale = sample_embedding_multiscale.prepare_partial(shuffle_rest)
		embedding_multiscale = np.vstack((sample_embedding_multiscale, rest_embedding_multiscale))[reverse]
	else:
		high_perplex = len(dims) / 100
		print("30 and " + str(high_perplex))
		affinities_multiscale_mixture = openTSNE.affinity.Multiscale(dims, perplexities=[30, (high_perplex)],metric="cosine", n_jobs=-1, random_state=42,verbose=True)
		init = openTSNE.initialization.pca(dims, random_state=42, verbose=True)
		print("Learning rate: " + str(len(dims) / 12))
		tsne_model = openTSNE.TSNE(learning_rate=len(dims) / 12, n_jobs=8, verbose=True)
		embedding_multiscale = tsne_model.fit(affinities=affinities_multiscale_mixture, initialization=init)
	pd.DataFrame(embedding_multiscale).to_csv(tsne_file, header=False, index=False, sep='\t',quoting=csv.QUOTE_NONE)



def send2HDBSCAN(umap_file,cluster_file,min_cluster_percentage):
	import csv
	umapped = pd.read_csv(umap_file,keep_default_na=False, sep='\t', header=None, quoting=csv.QUOTE_NONE)
	dims = np.array(umapped)
	# github.com/scikit-learn-contribi/hdbscan/issues/69
	#from sklearn.preprocessing import normalize
	#dims = normalize(dims, norm='l2')
	min_cluster_size = round(len(dims)*min_cluster_percentage)
	print("Min cluster size: " + str(min_cluster_size))
	#hdbscan_model = HDBSCAN(min_samples=2,cluster_selection_epsilon=cluster_selection_epsilon,min_cluster_size=min_cluster_size,cluster_selection_method='eom',prediction_data=True)
	hdbscan_model = HDBSCAN(min_samples=500,min_cluster_size=min_cluster_size, cluster_selection_method='eom', prediction_data=False)
	hdbscan_model.fit(dims)
	clusters = hdbscan_model.labels_
	#probs = hdbscan_model.probabilities_
	uniques = np.unique(np.array(clusters), return_counts=True)
	print("Distribution: " + str(uniques))
	print("Uniques: " + str(len(uniques[0])))
	print("-1 %: " + str(uniques[1][0]/len(clusters)))
	# convert array into dataframe
	#pd.DataFrame({'clusters': clusters, 'probs': probs}).to_csv(cluster_file, index=False, sep='\t',quoting=csv.QUOTE_NONE)
	pd.DataFrame({'clusters': clusters}).to_csv(cluster_file, index=False, sep='\t',quoting=csv.QUOTE_NONE)


def send2MATPLOT(dims2_file,clusters_file):
	import matplotlib.pyplot as plt
	import matplotlib.cm as cm
	dims = pd.read_csv(dims2_file, keep_default_na=False, sep='\t', header=None, quoting=csv.QUOTE_NONE)
	#from sklearn.preprocessing import normalize
	#dims = normalize(dims, norm='l2')
	print(dims.shape)
	clusters = pd.read_csv(clusters_file, keep_default_na=False, sep='\t', quoting=csv.QUOTE_NONE)
	plt.close()
	plt.scatter(dims[0], dims[1], c=clusters['clusters'], alpha=0.5, marker='.', s=1, cmap=cm.tab20)
	plt.show()


def send2BERTopic(uniques, embeddings, clusters):
	from bertopic import BERTopic
	from sklearn.feature_extraction.text import CountVectorizer
	import spacy
	# spacy.cli.download('en_core_web_sm')
	spacy.load('en_core_web_sm')
	from bertopic.representation import PartOfSpeech
	from bertopic.vectorizers import ClassTfidfTransformer
	from sentence_transformers import SentenceTransformer
	from bertopic.dimensionality import BaseDimensionalityReduction
	from bertopic.cluster import BaseCluster
	uniques = pd.read_csv(uniques, keep_default_na=False, sep='\t', header=None, quoting=csv.QUOTE_NONE)[0]
	embeddings = np.array(pd.read_csv(embeddings, keep_default_na=False, sep='\t', header=None, quoting=csv.QUOTE_NONE))
	clusters = pd.read_csv(clusters, keep_default_na=False, sep='\t', quoting=csv.QUOTE_NONE)['clusters']
	aspect_model1 = PartOfSpeech("en_core_web_sm")
	representation_models = [aspect_model1]
	vectorizer_model = CountVectorizer(stop_words="english", ngram_range=(1, 1))
	ctfidf_model = ClassTfidfTransformer(bm25_weighting=True, reduce_frequent_words=True)
	embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
	empty_reduction_model = BaseDimensionalityReduction()
	empty_cluster_model = BaseCluster()
	# Fit BERTopic without constructing embeddings
	topic_model = BERTopic(
		embedding_model=embedding_model,
		umap_model=empty_reduction_model,
		hdbscan_model=empty_cluster_model,
		vectorizer_model=vectorizer_model,
		ctfidf_model=ctfidf_model,
		representation_model=representation_models,
		verbose=True
	).fit(uniques, embeddings=embeddings, y=clusters)
	print("Customizing representative docs.")
	#representative_docs is screwed up for a number of reasons (pysbd doesn't properly parse " '; it tends to give long answers, etc.) let's fix that:
	revised_docs = pd.DataFrame({'Document': uniques, 'Topic': topic_model.topics_})
	revised_docs = revised_docs[~revised_docs.Document.str.contains("\"")]
	revised_docs = revised_docs[~revised_docs.Document.str.contains("\'")]
	revised_docs = revised_docs[revised_docs.Document.str.len() > 15]
	revised_docs = revised_docs[revised_docs.Document.str.len() < 100]
	revised_docs = revised_docs.reset_index(drop=True)
	# sample 5% of the data and give 10 sentences.
	tmprepr_docs_mappings, tmprepr_docs, tmprepr_docs_indices, tmprepr_docs_ids = topic_model._extract_representative_docs(c_tf_idf=topic_model.c_tf_idf_,
	                                                      documents=revised_docs,
	                                                      topics=topic_model.topic_representations_,
	                                                      nr_samples=round(revised_docs.shape[0]*0.05),
	                                                      nr_repr_docs=10)
	topic_model.representative_docs_= tmprepr_docs_mappings
	return topic_model

def send2BERTopicREVISEDDOCS(uniques, embeddings, clusters,revised_docs):
	from bertopic import BERTopic
	from sklearn.feature_extraction.text import CountVectorizer
	import spacy
	# spacy.cli.download('en_core_web_sm')
	spacy.load('en_core_web_sm')
	from bertopic.representation import PartOfSpeech
	from bertopic.vectorizers import ClassTfidfTransformer
	from sentence_transformers import SentenceTransformer
	from bertopic.dimensionality import BaseDimensionalityReduction
	from bertopic.cluster import BaseCluster
	uniques = pd.read_csv(uniques, keep_default_na=False, sep='\t', header=None, quoting=csv.QUOTE_NONE)[0]
	embeddings = np.array(pd.read_csv(embeddings, keep_default_na=False, sep='\t', header=None, quoting=csv.QUOTE_NONE))
	clusters = pd.read_csv(clusters, keep_default_na=False, sep='\t', quoting=csv.QUOTE_NONE)['clusters']
	aspect_model1 = PartOfSpeech("en_core_web_sm")
	representation_models = [aspect_model1]
	vectorizer_model = CountVectorizer(stop_words="english", ngram_range=(1, 1))
	ctfidf_model = ClassTfidfTransformer(bm25_weighting=True, reduce_frequent_words=True)
	embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
	empty_reduction_model = BaseDimensionalityReduction()
	empty_cluster_model = BaseCluster()
	# Fit BERTopic without constructing embeddings
	topic_model = BERTopic(
		embedding_model=embedding_model,
		umap_model=empty_reduction_model,
		hdbscan_model=empty_cluster_model,
		vectorizer_model=vectorizer_model,
		ctfidf_model=ctfidf_model,
		representation_model=representation_models,
		verbose=True
	).fit(uniques, embeddings=embeddings, y=clusters)
	tmprepr_docs_mappings, tmprepr_docs, tmprepr_docs_indices, tmprepr_docs_ids = topic_model._extract_representative_docs(c_tf_idf=topic_model.c_tf_idf_,
	                                                      documents=revised_docs,
	                                                      topics=topic_model.topic_representations_,
	                                                      nr_samples=round(revised_docs.shape[0]*0.05),
	                                                      nr_repr_docs=10)
	topic_model.representative_docs_= tmprepr_docs_mappings
	return topic_model

def resultsFromBERT2HDBSCANtopics(topic_model,hdbscan_file):
	import re
	map_BERTandHDBSCAN_topic_numbers = pd.DataFrame({'BERT': np.array(topic_model.topics_),
	                                                 'HDBSCAN': np.array(pd.read_csv(hdbscan_file,
	                                                                                 keep_default_na=False, sep='\t',
	                                                                                 quoting=csv.QUOTE_NONE)[
		                                                                     'clusters'])}).drop_duplicates().set_index(
	'BERT')['HDBSCAN'].to_dict()
	results = topic_model.get_topic_info()
	# let's repopulate representative docs...
	for i in range(len(results['Topic'])):
		results['Topic'][i] = map_BERTandHDBSCAN_topic_numbers.get(results['Topic'][i])
		results['Name'][i] = re.sub("^[0-9]*_",str(results['Topic'][i])+"_",results['Name'][i])
	return results



def results2browser(results):
	import os, tempfile
	resultsHTML = pd.DataFrame(results).to_html(classes=["table-bordered", "table-striped", "table-hover"])
	tmp = tempfile.NamedTemporaryFile(delete=False,suffix=".html")
	try:
		print(tmp.name)
		tmp.write(resultsHTML.encode('ascii'))
		os.system("firefox " + tmp.name)
		time.sleep(2)
		tmp.close()
	finally:
		os.unlink(tmp.name)
	return

def results2csv(results,results_file):
	results.to_csv(results_file,index=False, sep='\t')

send2UMAP(homedir+"embedding_psyop.pbz2",homedir+"umap_psyop.tsv")
send2OpenTSNE(homedir+"umap_psyop.tsv",homedir+"tsne_psyop.tsv")

min_cluster_percentage = 0.005
send2HDBSCAN(homedir+"tsne_psyop.tsv",homedir+"hdbscan_psyop"+str(min_cluster_percentage)+"_mcp.tsv",min_cluster_percentage)
#send2MATPLOT(homedir+"tsne_psyop.tsv",homedir+"hdbscan_psyop0.01_mcp.tsv")
psyop_BERTopic = send2BERTopic(homedir+"uniques_psyop.tsv",homedir+"tsne_psyop.tsv",homedir+"hdbscan_psyop0.005_mcp.tsv")
results2browser(resultsFromBERT2HDBSCANtopics(psyop_BERTopic,homedir+"hdbscan_psyop0.005_mcp.tsv"))
results2csv(resultsFromBERT2HDBSCANtopics(psyop_BERTopic,homedir+"hdbscan_psyop0.005_mcp.tsv"),homedir+"results_hdbscan_psyop0.005_mcp.tsv")

uniques = pd.read_csv(homedir+"uniques_psyop.tsv", keep_default_na=False, sep='\t', header=None, quoting=csv.QUOTE_NONE)
all_posts = pd.read_csv(homedir+"all_psyop_posts_combo.tsv", keep_default_na=False, sep='\t', header=None, quoting=csv.QUOTE_NONE)
uniques_platform = [all_posts[2][index] for index in uniques[1]]


# create revised docs for each platform
revised_docs = pd.DataFrame({'Document': uniques[0], 'Topic': psyop_BERTopic.topics_, 'Platform': uniques_platform})
revised_docs = revised_docs[~revised_docs.Document.str.contains("\"")]
revised_docs = revised_docs[~revised_docs.Document.str.contains("\'")]
revised_docs = revised_docs[revised_docs.Document.str.len() > 15]
revised_docs = revised_docs[revised_docs.Document.str.len() < 100]
revised_docs = revised_docs.reset_index(drop=True)
# sample 5% of the data and give 10 sentences.
revised_docs_4chan = revised_docs[revised_docs['Platform']=='4chan'].reset_index(drop=True)
revised_docs_meta = revised_docs[revised_docs['Platform']!='4chan']
revised_docs_meta = revised_docs_meta[revised_docs_meta['Platform']!='Telegram'].reset_index(drop=True)
revised_docs_telegram = revised_docs[revised_docs['Platform']=='Telegram'].reset_index(drop=True)

psyop_BERTopic_4chan = send2BERTopicREVISEDDOCS(homedir+"uniques_psyop.tsv",homedir+"tsne_psyop.tsv",homedir+"hdbscan_psyop0.005_mcp.tsv",revised_docs_4chan)
psyop_BERTopic_telegram = send2BERTopicREVISEDDOCS(homedir+"uniques_psyop.tsv",homedir+"tsne_psyop.tsv",homedir+"hdbscan_psyop0.005_mcp.tsv",revised_docs_telegram)
psyop_BERTopic_meta = send2BERTopicREVISEDDOCS(homedir+"uniques_psyop.tsv",homedir+"tsne_psyop.tsv",homedir+"hdbscan_psyop0.005_mcp.tsv",revised_docs_meta)

modified_results = resultsFromBERT2HDBSCANtopics(psyop_BERTopic,homedir+"hdbscan_psyop0.005_mcp.tsv")

modified_results['Rep_Docs_4chan'] = resultsFromBERT2HDBSCANtopics(psyop_BERTopic_4chan,homedir+"hdbscan_psyop0.005_mcp.tsv")['Representative_Docs']
modified_results['Rep_Docs_meta'] = resultsFromBERT2HDBSCANtopics(psyop_BERTopic_meta,homedir+"hdbscan_psyop0.005_mcp.tsv")['Representative_Docs']
modified_results['Rep_Docs_telegram'] = resultsFromBERT2HDBSCANtopics(psyop_BERTopic_telegram,homedir+"hdbscan_psyop0.005_mcp.tsv")['Representative_Docs']
modified_results = modified_results.drop('Representative_Docs', axis=1)


results2csv(modified_results,homedir+"results_MULTIPLEREPDOCS_hdbscan_psyop0.005_mcp.tsv")

results2browser(modified_results)

