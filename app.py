"""
Description
Training app for intro to corpus linguistics


"""
# Core Pkgs
import streamlit as st
import os
from operator import itemgetter

# NLP Pkgs
import spacy
from spacy import displacy
import nltk
nltk.download('punkt')
from nltk.tokenize import TreebankWordTokenizer as twt
from nltk.util import ngrams

#dealing with NLTk std.out
from io import StringIO
import sys

# GIF support
from streamlit_lottie import st_lottie
from streamlit_lottie import st_lottie_spinner
import requests

def load_lottieurl(url: str):
	r = requests.get(url)
	if r.status_code != 200:
		return none
	return r.json()


# Set tabs up
tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs(["Concordancer", "Frequencies", "POS Tagging", "Name Entity Recognition", "Keywords", "Collocations and N-grams"])

# Function for frequency list TAB0
def concordancer(corpus, searchTerm):
	tmp = sys.stdout
	my_result = StringIO()
	sys.stdout = my_result
	tokenizedText = nltk.word_tokenize(corpus)
	concText = nltk.Text(tokenizedText)
	concText.concordance(searchTerm, lines= 50)
	sys.stdout = tmp
	return my_result.getvalue()


# Function for frequency list TAB1
def get_freqy(my_text):
	my_text = my_text.lower()
	tokens = nltk.word_tokenize(my_text)
	freqDist = nltk.FreqDist(tokens)
	return freqDist.most_common(50)

@st.cache_resource
def text_analyzer(my_text):
	nlp = spacy.load('en_core_web_sm')
	docx = nlp(my_text)
	# tokens = [ token.text for token in docx]
	allData = [('"Token":{},\n"Lemma":{}'.format(token.text,token.lemma_))for token in docx ]
	return allData

# Function for POS visualization TAB2
def get_pos(text):
	# Tokenize text and pos tag each token
	tokens = twt().tokenize(text)
	tags = nltk.pos_tag(tokens, tagset = "universal")
	return tags

# Function For Extracting Entities TAB3
@st.cache_resource
def entity_analyzer(my_text):
	nlp = spacy.load('en_core_web_sm')
	docx = nlp(my_text)
	tokens = [ token.text for token in docx]
	entities = [(entity.text,entity.label_)for entity in docx.ents]
	allData = ['"Entities":{}'.format(entities)]
	return allData


# Keyword extraction done in place TAB4

# Function for collocations and N-grams TAB5
def ngram_analyzer(my_text, num):
	n_grams = ngrams(nltk.word_tokenize(my_text), num)
	return [ ' '.join(grams) for grams in n_grams]


def main():
	""" web interface """

	# Title
	with st.sidebar:
		computer = load_lottieurl("https://assets6.lottiefiles.com/packages/lf20_mymcy4zr.json")
		st_lottie(computer)
	
	st.sidebar.title("A First Glance at Corpus Methods")
	st.sidebar.subheader("A sampler of different corpus linguistic approaches to text")

	# Concordancer
	with tab0:
		st.subheader("Concordancer")
		st.info("Concordancing allows us to see a word of our choice in its original context. The resulting lines of text are sometimes referred to as 'Keyword in Context Lines', or KWIC Lines!")
		message = st.text_area("Enter Text","Type Here.")
		query = st.text_area("Enter Query","Type Query.")
		if st.button("Create Concordance"):
			st.text("Concordance lines:")
			st.text(concordancer(message, query))


	# Frequencies
	with tab1:
		st.subheader("See the word frequencies of your text")
		st.info("The frequencies of words in our text can tell us a lot about what is going on! We can use a frequency list to look for recurring clues.")
		message = st.text_area("Enter Text","Type Here..")
		if st.button("Get Frequencies"):
			st.text("Using NLTK Tokenizer...")
			results = get_freqy(message)
			st.json(results)
		st.subheader("Tokenize and Lemmatize your Text")
		st.info("When we want to talk about the frequencies of words in a text, we often talk about 'Types' and 'Tokens'. Tokens are the individual words of a text, while types are the unique words.")
		st.info("We can also use Lemmatization when we want to know what is going on in a text. Lemmas are the 'dictionary', or base, forms of words. When we 'lemmatize' our text, we collect the base form of each word, which grants our frequency lists a different perspective.")
		message = st.text_area("Enter Text","Type Here.....")
		if st.button("Tokenize"):
			st.text("Using Spacy tokenizer...")
			nlp_result = text_analyzer(message)
			st.json(nlp_result)

	# POS tagging
	with tab2:
		st.subheader("See the Parts-of-Speech in your text")
		st.info("Part-of-Speech tagging, or POS-tagging, is a great way of getting a new perspective on a text. The tags tell us which wordclasses are found in a text, and can be used to get overviews of the features of language in large materials.")
		message = st.text_area("Enter Text","Type Here...")
		if st.button("Tag for POS"):
			st.text("Using NLTK Tagger...")
			results = get_pos(message)
			st.success(results)


	# Entity Extraction
	with tab3:
		st.subheader("Identify Entities in your text")
		st.info("Named Entity Recognition is used to see what kinds of entities are found in a text, and it can help us figure out central areas, characters and concepts of our text. The tags look similar to the POS-tags, but are used to signal very different information.")
		message = st.text_area("Enter Text","Type Here....")
		if st.button("Extract"):
			entity_result = entity_analyzer(message)
			st.json(entity_result)


	#Keyword extraction
	with tab4:
		st.subheader("Get the Keywords of your Text")
		st.info("Which are the most important words of a text? We often talk about 'Keywords' when trying to define a text by its contents, but there are many different approaches to actually figuring out the 'key' words. This tool uses a trained dataset from Spacy, but there also manual approaches.")
		message = st.text_area("Enter Text","Type Here......")
		if st.button("Get Keywords"):
			st.text("Using Spacy Keyword Extractor...")
			nlp = spacy.load('en_core_web_sm')
			docx = nlp(message)
			summary_result = docx.ents
			st.success(summary_result)


	#Collocation and N-grams
	with tab5:
		st.subheader("Get the Collocations of your Text")
		st.info("'We should know a word by the company it keeps' is one of those quotes every linguistics student will hear from a lecturer at least once during their studies. The reason for this is that the context of a word is very important for formulating our understanding of it. From a corpus linguistics perspective we can look for words that often appear together in text in order to get an idea of the word groups in our specific text. If this seems dull, ask your teacher about connotation!")
		st.info("N-grams are simply groups of words defined by the number of words included. This tool allows us to look for bigrams (2 words), trigrams (3 words), and quadgrams(4 words).")
		message = st.text_area("Enter Text","Type Here.......")
		
		if st.button("Get Bigrams"):
			st.text("Using NLTK N-gram Extractor...")
			results = ngram_analyzer(message, 2)
			st.json(results)
		if st.button("Get Trigrams"):
			st.text("Using NLTK N-gram Extractor...")
			results = ngram_analyzer(message, 3)
			st.json(results)
		if st.button("Get Quadgrams"):
			st.text("Using NLTK N-gram Extractor...")
			results = ngram_analyzer(message, 4)
			st.json(results)

	st.sidebar.subheader("About the App")
	st.sidebar.info("This is a training app intended to introduce you to the basics of corpus linguistics. Use your own choice of text to see how the different methods work in practice!")
	st.sidebar.subheader("Contact")
	st.sidebar.text("Daniel Ihrmark")
	st.sidebar.text("(daniel.o.sundberg@lnu.se)")




if __name__ == '__main__':
	main()
