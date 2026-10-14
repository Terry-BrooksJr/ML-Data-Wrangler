import os
import pathlib
import random
from typing import Callable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # Plots are saved to files; no GUI backend in worker threads.

import matplotlib.pyplot as plt
import nltk
import pyLDAvis
import pyLDAvis.gensim_models
import seaborn as sns
from gensim.corpora.dictionary import Dictionary as MappingDictionary
from gensim.models import CoherenceModel
from gensim.models.ldamodel import LdaModel
from loguru import logger
from nltk.corpus import stopwords
from numpy.random import RandomState
from PyQt5.QtCore import pyqtSignal, QMutex, QObject
from spacy.language import Language

from wrangler import DataWrangler

try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords", quiet=True)
sns.set_theme()

# spaCy parts of speech that carry no topical meaning.
REMOVED_POS = ["ADV", "PRON", "PUNCT", "PART", "DET", "ADP", "SPACE", "NUM", "SYM"]
# Dictionary.filter_extremes settings; recorded with each saved run.
FILTER_NO_BELOW = 5
FILTER_NO_ABOVE = 0.5
FILTER_KEEP_N = 5000


def tokenize(
    nlp: Language,
    stopwords: List[str],
    documents: List[str],
    progress: Optional[Callable[[int], None]] = None,
) -> List[List[str]]:
    """Lemmatizes documents into token lists, dropping stopwords and REMOVED_POS.

    Used for training and for assigning topics with a saved model, so both see
    the same preprocessing.

    Args:
        progress: Called with percent complete (0-100) after each document.
    """
    stopword_set = set(stopwords)
    tokens: List[List[str]] = []
    total = len(documents) or 1
    # Lemmas only need the tagger, so skip the slower parser and NER.
    with nlp.select_pipes(disable=[p for p in ("parser", "ner") if p in nlp.pipe_names]):
        for i, document in enumerate(nlp.pipe(documents)):
            tokens.append(
                [
                    token.lemma_.lower()
                    for token in document
                    if (
                        token.pos_ not in REMOVED_POS
                        and token.lemma_.lower() not in stopword_set
                        and not token.is_stop
                        and token.is_alpha
                    )
                ]
            )
            if progress:
                progress(int((i + 1) / total * 100))
    return tokens


class LatentDirichletAllocator(QObject):
    """
    A class for performing Latent Dirichlet Allocation (LDA) on a text corpus.

    This class handles the preprocessing of text data, training of the LDA model,
    and visualization of the results. It provides methods to manage topics and
    coherence values, enabling effective topic modeling on the provided corpus.

    Signals:
        preprocess_finished: Emitted when the corpus has been tokenized.
        preprocess_progress: Emitted with preprocessing progress (0-100).
        train_finished: Emitted when training is complete.
        train_progress: Emitted with training progress (0-100).
        error: Emitted with an error message if an error occurs.
        worker_status: Emitted with the current status of the worker.
    """

    preprocess_finished = pyqtSignal()
    preprocess_progress = pyqtSignal(int)
    train_finished = pyqtSignal()
    train_progress = pyqtSignal(int)
    _mutex = QMutex()
    error = pyqtSignal(str)
    worker_status = pyqtSignal(str)

    def __init__(self, num_of_topics: int, seed: Optional[int] = None) -> None:
        """
        Initializes the LatentDirichletAllocator.

        Args:
            num_of_topics: The highest number of topics to evaluate when training.
            seed: Optional random seed, for reproducible models.
        """
        super().__init__()
        self.corpus: List[List[Tuple[int, int]]] = []
        self._tokens: List[List[str]] = []
        self.id2word: Optional[MappingDictionary] = None
        self.num_of_topics: int = num_of_topics
        self.seed: int = seed if seed is not None else random.randint(1, 2**32 - 1)
        # One document per ticket, and the Zendesk ID of each, in the same order.
        self.prelemma_corpus: Optional[List[str]] = None
        self.document_ids: List[str] = []
        # The settings of the last training run, recorded with saved runs.
        self.settings: dict = {}

        self.topic_counts: List[int] = []
        self.coherence_values: List[float] = []
        self.lda_model: Optional[LdaModel] = None
        self.best_num_topics: Optional[int] = None

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, name: str, value) -> None:
        return setattr(self, name, value)

    @staticmethod
    def validate_inputs(number_of_topics, iterations, passes) -> Tuple[bool, str]:
        """
        Validates the user inputs for model training parameters.

        Args:
            number_of_topics (str): The highest number of topics to evaluate.
            iterations (str): The number of iterations for the training process.
            passes (str): The number of passes for the training process.

        Returns:
            tuple: A boolean indicating validity and an error message.
        """
        values = [str(number_of_topics), str(iterations), str(passes)]
        if not all(map(str.isdigit, values)):
            return False, "All inputs must be integers."
        if int(number_of_topics) < 2:
            return False, "Number of topics must be at least 2."
        if int(passes) >= 20 or int(iterations) >= 200:
            return False, "Passes should be < 20 and iterations < 200."
        return True, ""

    def get_lda_model(
        self,
        iterations: int = 50,
        passes: int = 10,
        num_of_topics: Optional[int] = None,
    ) -> LdaModel:
        """Trains an LDA model with the specified parameters.

        Runs in a single process: on this corpus it is faster than
        LdaMulticore, and on macOS each extra process re-imports app.py.

        Args:
            iterations: The number of iterations for the LDA model.
            passes: The number of passes through the corpus.
            num_of_topics: The number of topics to model (defaults to instance's value).

        Returns:
            An LdaModel instance configured with the provided parameters.
        """
        return LdaModel(
            corpus=self.corpus,
            id2word=self.id2word,
            iterations=iterations,
            num_topics=num_of_topics or self.num_of_topics,
            passes=passes,
            random_state=RandomState(self.seed),
        )

    def process_corpus(
        self, nlp: Language, stopwords: List[str], wranglerInstance: DataWrangler
    ) -> bool:
        """
        Lemmatizes each ticket document and builds the bag-of-words corpus.

        If no documents have been set, they are generated from the DataWrangler.

        Args:
            nlp (Language): The spaCy model used for tokenization and lemmatization.
            stopwords (List[str]): Words to exclude from the tokens.
            wranglerInstance (DataWrangler): Used to generate the documents if needed.

        Returns:
            bool: True if the data was successfully pre-processed, False otherwise.
        """
        logger.info(
            "Configured SpaCy Model and NLTK Stopwords...Initiating Data Cleanse and Dictonary Creation"
        )
        self.worker_status.emit("Configured SpaCy Model..")
        try:
            if not self.prelemma_corpus:
                logger.info(
                    "Prelemma Corpus is empty. Regenerating Corpus using DataWrangler"
                )
                self.prelemma_corpus = wranglerInstance.ticket_documents()
                self.document_ids = wranglerInstance.ticket_ids()

            self.worker_status.emit("Lemmatizing Corpus...")
            self._tokens = tokenize(nlp, stopwords, self.prelemma_corpus, self.preprocess_progress.emit)
            self.worker_status.emit("Lemmatization Completed")
            logger.debug(f"Token Length:{len(self._tokens)}")

            self.worker_status.emit(
                f"Preparing Mapping Dictonary for {len(self._tokens)} documents... "
            )
            self.id2word = MappingDictionary(self._tokens)
            self.worker_status.emit("Finally, Filtering Out Extremes...")
            self.id2word.filter_extremes(
                no_below=FILTER_NO_BELOW, no_above=FILTER_NO_ABOVE, keep_n=FILTER_KEEP_N
            )
            self.corpus = [self.id2word.doc2bow(doc) for doc in self._tokens]
            logger.debug(
                f"Pre-Lemma Corpus Length:{len(self.prelemma_corpus)} \n Mapping Dict: {self.id2word} \n Post Processing Corpus: {len(self.corpus)}"
            )
            logger.success("Successfully Processed Corpus")
            self.preprocess_finished.emit()
            return True
        except Exception as e:
            logger.exception("FAILED to Process Data")
            self.error.emit(f"Failed to process data: {e}")
            return False

    def model_trained(
        self, iterations: int, passes: int, num_of_topics: int
    ) -> bool:
        """Trains LDA models for 2..num_of_topics topics and keeps the most coherent.

        Args:
            iterations: The number of iterations for the LDA model training.
            passes: The number of passes through the corpus during training.
            num_of_topics: The highest number of topics to evaluate.

        Returns:
            True if the models are trained and coherence values are recorded,
            False otherwise.
        """
        try:
            self.topic_counts, self.coherence_values = [], []
            self.lda_model, self.best_num_topics = None, None
            self.settings = {
                "max_topics": num_of_topics,
                "iterations": iterations,
                "passes": passes,
                "seed": self.seed,
                "filter_extremes": {
                    "no_below": FILTER_NO_BELOW,
                    "no_above": FILTER_NO_ABOVE,
                    "keep_n": FILTER_KEEP_N,
                },
            }
            topic_range = range(2, num_of_topics + 1)
            for i, topic_count in enumerate(topic_range):
                self.worker_status.emit(f"Training model with {topic_count} topics...")
                lda_model = self.get_lda_model(
                    iterations=iterations,
                    passes=passes,
                    num_of_topics=topic_count,
                )
                coherence = CoherenceModel(
                    model=lda_model,
                    texts=self._tokens,
                    dictionary=self.id2word,
                    coherence="c_v",
                    processes=1,  # c_v otherwise starts a pool of processes
                ).get_coherence()
                logger.info(f"{topic_count} topics: coherence {coherence:.4f}")
                self.topic_counts.append(topic_count)
                self.coherence_values.append(coherence)
                if coherence == max(self.coherence_values):
                    self.lda_model, self.best_num_topics = lda_model, topic_count
                self.train_progress.emit(int((i + 1) / len(topic_range) * 100))

            logger.success(
                f"Successfully Trained Model. Best: {self.best_num_topics} topics"
            )
            self.train_finished.emit()
            return True
        except Exception as e:
            logger.exception("Failed to  Train Model")
            self.error.emit(f"Failed to train model: {e}")
            return False

    def run(
        self,
        nlp: Language,
        stopwords: List[str],
        wranglerInstance: DataWrangler,
        iterations: int,
        passes: int,
        num_of_topics: int,
    ) -> bool:
        """Preprocesses the corpus and then trains the models, for use in a worker thread."""
        return self.process_corpus(nlp, stopwords, wranglerInstance) and self.model_trained(
            iterations=iterations,
            passes=passes,
            num_of_topics=num_of_topics,
        )

    def get_topics(self, num_words: int = 10) -> List[Tuple[int, float, List[str]]]:
        """Returns the best model's topics, most prevalent first.

        Returns:
            A list of (topic id, share of the corpus, top words) tuples.
        """
        if self.lda_model is None:
            return []
        weights = [0.0] * self.lda_model.num_topics
        for bow in self.corpus:
            for topic_id, probability in self.lda_model.get_document_topics(bow):
                weights[topic_id] += probability
        total = sum(weights) or 1.0
        topics = [
            (
                topic_id,
                weights[topic_id] / total,
                [word for word, _ in self.lda_model.show_topic(topic_id, topn=num_words)],
            )
            for topic_id in range(self.lda_model.num_topics)
        ]
        return sorted(topics, key=lambda topic: topic[1], reverse=True)

    def get_top_5_topic(self) -> List[str]:
        """Retrieves the five most prevalent topics of the best model.

        Returns:
            A list of up to five topics, each as a string of its top words.
        """
        return [
            f"{share:.0%} of corpus: {', '.join(words)}"
            for _, share, words in self.get_topics()[:5]
        ]

    def visualize_results(
        self, output_dir: Optional[pathlib.Path] = None
    ) -> Tuple[Optional[str], Optional[str]]:
        """Saves the pyLDAvis chart and the coherence plot for the best model.

        Args:
            output_dir: Where to write the files; each saved run passes its own
                directory so charts aren't overwritten. Defaults to the
                current directory.

        Returns:
            Paths to the saved LDA chart (HTML) and coherence plot (PNG), or
            (None, None) if visualization fails.
        """
        output_dir = output_dir if output_dir is not None else pathlib.Path.cwd()
        try:
            chart_path = os.path.join(output_dir, "lda_model.html")
            lda_display = pyLDAvis.gensim_models.prepare(
                self.lda_model, self.corpus, self.id2word
            )
            pyLDAvis.save_html(lda_display, chart_path)

            plot_path = os.path.join(output_dir, "coherence.png")
            plt.figure()
            plt.plot(self.topic_counts, self.coherence_values, marker="o")
            plt.xlabel("Number of Topics")
            plt.ylabel("Coherence")
            plt.savefig(plot_path, bbox_inches="tight")
            plt.close()
            return chart_path, plot_path
        except Exception:
            logger.exception("Failed to Visualize Results")
            return None, None
