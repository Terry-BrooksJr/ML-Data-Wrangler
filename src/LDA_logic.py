import en_core_web_lg
import gensim.corpora as corpora
import matplotlib.pyplot as plt
import nltk
import pyLDAvis.gensim_models
import seaborn as sns
import spacy
from gensim.corpora.dictionary import Dictionary
from gensim.models import CoherenceModel, LdaMulticore
from gensim.models.ldamodel import LdaModel
from loguru import logger
from nltk.corpus import stopwords

nltk.download("stopwords")
sns.set_theme()


class LatentDirichletAllocator:
    def __init__(self, corpus: str, num_of_topics: int) -> None:
        self.corpus = ""
        self.documents = ""
        self._tokens = []
        self.id2word = ""
        self.num_of_topics = num_of_topics
        self.prelemma_corpus = corpus
        self.topics = []
        self.coherence_values = []

    def get_lda_model(
        self, iterations: int, workers: int, passes: int, num_of_topics: int = 0
    ) -> LdaModel:
        if num_of_topics is None:
            num_of_topics = self.num_of_topics
        return LdaMulticore(
            corpus=self.corpus,
            id2word=self.id2word,
            iterations=iterations,
            num_topics=num_of_topics,
            workers=workers,
            passes=passes,
            random_state=100,
        )

    def data_preprocessed(self):
        nlp = en_core_web_lg.load()
        stop_words = stopwords.words("english")
        removal = [
            "ADV",
            "PRON",
            "CCONJ",
            "PUNCT",
            "PART",
            "DET",
            "ADP",
            "SPACE",
            "NUM",
            "SYM",
        ]
        logger.info(
            "Configured SpaCy Model and NLTK Stopwords...Initiating Data Cleanse and Dictonary Creation "
        )
        try:
            return self.data_reshaped(nlp, removal)
        except Exception:
            logger.exception("Failed to Preprocess Data")
            return False

    def data_reshaped(self, nlp, removal):
        try:
            for comment in nlp.pipe(self.prelemma_corpus):
                proj_tok = [
                    comment.lemma_.lower()
                    for comment in self.prelemma_corpus
                    if comment.pos_ not in removal
                    or comment.lemma_.lower() not in stopwords
                    and not comment.is_stop
                    and comment.is_alpha
                ]
                self._tokens.append(proj_tok)
            self.id2word = Dictionary(self._tokens)
            self.id2word.filter_extremes(no_below=5, no_above=0.5, keep_n=1000)
            self.corpus = [self.GDictionary.doc2bow(doc) for doc in self._tokens]
            logger.success("Successfully PreProcessed Data")
            return True
        except Exception:
            logger.exception("Unable to Massage Data")
            return False

    def model_trained(self, passes: int, iterations: int, workers: int = 4) -> list:
        try:
            for i in range(1, 20):
                lda_model = self.get_lda_model(50, 4, 10, i)
                cm = CoherenceModel(
                    model=lda_model,
                    corpus=self.corpus,
                    dictionary=self.id2word,
                    coherence="c_v",
                )
                self.topics.append(i)
                self.coherence_values.append(cm.get_coherence())
            logger.success("Successfully Trained Model")
            return True
        except Exception as e:
            logger.exception("Failed to  Train Model")
            return False

    def vizualize_results(self):
        try:
            lda_display = pyLDAvis.gensim_models.prepare(
                self.get_lda_model(), self.corpus, self.id2word
            )
            _ = plt.plot(self.topics, self.coherence_values)
            _ = plt.xlabel("Number of Topics")
            _ = plt.ylabel("Coherence")
            plt.show()
            return lda_display
        except Exception:
            logger.exception("Failed to Vizualize Results")
            return None

    def get_top_5_topic(self):
        return self.topics[:4]
