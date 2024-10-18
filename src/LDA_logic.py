import random
from typing import List

import en_core_web_lg
import matplotlib.pyplot as plt
import nltk
import pyLDAvis.gensim_models
import seaborn as sns
from gensim.corpora.dictionary import Dictionary as MappingDictionary
from gensim.models import CoherenceModel, LdaMulticore
from gensim.models.ldamodel import LdaModel
from loguru import logger
from nltk.corpus import stopwords
from numpy.random import RandomState
from spacy.tokens import Doc
from spacy.language import Language
from wrangler import DataWrangler
nltk.download("stopwords")
sns.set_theme()


class LatentDirichletAllocator:
    """
     A class for performing Latent Dirichlet Allocation (LDA) on a text corpus.

    This class handles the preprocessing of text data, training of the LDA model,
    and visualization of the results. It provides methods to manage topics and
    coherence values, enabling effective topic modeling on the provided corpus.
    """

    def __init__(self, num_of_topics: int) -> None:
        """
        Initializes the LatentDirichletAllocator with a corpus and number of topics.

        Args:
            corpus: The text corpus to be analyzed.
            num_of_topics: The number of topics to be identified by the LDA model.
        """
        self.corpus: List[Doc] = []
        self.documents: str = ""
        self._tokens: List[Doc] = []
        self.id2word: MappingDictionary = ""
        self.num_of_topics: int = num_of_topics
        self.prelemma_corpus: str = None

        self.topics: List[str] = []
        self.coherence_values: List[float] = []
    def __getitem__(self, item):  
         return self.LatentDirichletAllocator[item]
    def _generate_random_state(self) -> RandomState:
        """
        Generates a random state for model training.

        Returns:
            A RandomState object initialized with a random seed.
        """
        random_seed = random.randint(1, (4294967296 - 1))
        return RandomState(random_seed)

    def get_lda_model(
        self, iterations: int, workers: int, passes: int, num_of_topics: int = 0
    ) -> LdaModel:
        """Retrieves the LDA model based on the specified parameters.

        Args:
            iterations: The number of iterations for the LDA model.
            workers: The number of worker threads to use.
            passes: The number of passes through the corpus.
            num_of_topics: The number of topics to model (defaults to instance's value).

        Returns:
            An LdaModel instance configured with the provided parameters.
        """
        if num_of_topics is None:
            num_of_topics = self.num_of_topics
        elif workers is None:
            workers = 4
        return LdaMulticore(
            corpus=self.corpus,
            id2word=self.id2word,
            iterations=iterations,
            num_topics=num_of_topics,
            workers=workers,
            passes=passes,
            random_state=self._generate_random_state(),
        )

    def data_preprocessed(self, wranglerInstance:DataWrangler=None) -> bool:
        """
        Prepares the data for further processing by configuring the necessary NLP model and stopwords.

        This function initializes the SpaCy language model and defines a list of parts of speech to remove during data cleansing. It then calls the `data_reshaped` method to perform the actual data reshaping and tokenization, handling any exceptions that may occur during the process.

        Args:
            wranglerInstance (DataWrangler, optional): An instance of DataWrangler used for data handling, defaults to None.

        Returns:
            bool: True if the data was successfully preprocessed, False otherwise.

        Raises:
            Exception: Logs an error if data preprocessing fails.
        """
        nlp = en_core_web_lg.load()
        stop_words: List[str] = stopwords.words("english")
        removal = [
            "ADV",
            "PRON",
            "PUNCT",
            "PART",
            "DET",
            "ADP",
            "SPACE",
            "NUM",
            "SYM",
        ]
        logger.info(
            "Configured SpaCy Model and NLTK Stopwords...Initiating Data Cleanse and Dictonary Creation"
        )
        try:
            return self.data_reshaped(nlp, removal, wranglerInstance)
        except Exception:
            logger.exception("Failed to Preprocess Data")
            return False

    def data_reshaped(self, nlp:Language, removal:List[str],wranglerInstance:DataWrangler ) -> bool:
        """
        Preprocesses the data by reshaping the corpus and generating tokens from the input text.

        This function takes a natural language processing model and a list of parts of speech to remove, processes the prelemma corpus to extract lemmatized tokens, and constructs a bag-of-words representation. It handles the case where the prelemma corpus is empty by attempting to regenerate it using a provided DataWrangler instance.

        Args:
            nlp (Language): The natural language processing model used for tokenization and lemmatization.
            removal (List[str]): A list of part-of-speech tags to be excluded from the tokenization process.
            wranglerInstance (DataWrangler): An instance of DataWrangler used to regenerate the corpus if necessary.

        Returns:
            bool: True if the data was successfully pre-processed, False otherwise.

        Raises:
            Exception: Logs an error if data processing fails.
        """
        try: 
            if self.prelemma_corpus is not None:
                for comment in nlp.pipe(self.prelemma_corpus):
                    proj_tok = [
                        token.lemma_.lower()
                        for token in comment
                        if (
                            token.pos_ not in removal
                            and token.lemma_.lower() not in stopwords
                            and not token.is_stop
                            and token.is_alpha
                        )
                    ]
                    self._tokens.append(proj_tok)
            else:
                logger.error("Prelemma Corpus is Empty...attempting to regenerate Corpus...")
                self.prelemma_corpus = wranglerInstance.create_corpus()
            logger.debug(f"Token Length:{len(self._tokens)}")
            self.id2word = MappingDictionary(self._tokens)
            self.id2word.filter_extremes(no_below=5, no_above=0.5, keep_n=1000)
            logger.debug(f"Pre-Lemma Corpus Length:{len(self.prelemma_corpus)}")
            self.corpus = [self.id2word.doc2bow(doc) for doc in self._tokens]

            logger.success("Successfully Pre-Processed Data")
            return True
        except Exception as e:
            logger.exception("Unable to Massage Data")
            return False

    def model_trained(
        self, iterations: int, workers: int, passes: int, num_of_topics: int
    ) -> bool:
        """Trains the LDA model and evaluates coherence for a range of topic counts.

        This method iteratively trains the LDA model for a specified number of
        topics and records the coherence values for each topic count. It helps
        in determining the optimal number of topics for the model based on
        coherence scores.

        Args:
            iterations: The number of iterations for the LDA model training.
            workers: The number of worker threads to use during training.
            passes: The number of passes through the corpus during training.
            num_of_topics: The number of topics to evaluate during training.

        Returns:
            True if the model is successfully trained and coherence values are
            recorded, False otherwise.
        """
        try:
            for i in range(1, 20):
                lda_model = self.get_lda_model(
                    iterations=50, workers=4, passes=10, num_of_topics=i
                )
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

    def visualize_results(self):
        """Visualizes the results of the LDA model and its coherence values.

        This method prepares and displays the visualization of the LDA model
        using the pyLDAvis library, along with a plot of coherence values
        against the number of topics. It provides insights into the model's
        performance and helps in understanding the topic distribution.

        Returns:
            The visualization object if successful, None otherwise.

        Raises:
            Exception: If there is an error during the visualization process.
        """
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
            logger.exception("Failed to Visualize Results")
            return None

    def get_top_5_topic(self):
        """Retrieves the top five topics identified by the LDA model.

        This method returns a list containing the five most significant topics
        based on the training results. It is useful for quickly accessing the
        primary topics derived from the analysis.

        Returns:
            A list of the top five topics.
        """
        return self.topics[:5]
