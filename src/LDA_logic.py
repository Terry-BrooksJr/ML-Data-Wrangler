import random
from typing import List, Tuple, Union, Callable
from PyQt5.QtWidgets import QMessageBox
from tqdm_loggable.auto import tqdm
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
import threading
from concurrent.futures import ThreadPoolExecutor
from wrangler import DataWrangler
import gradio as gr

nltk.download("stopwords")
sns.set_theme()


class LatentDirichletAllocator:
    """
     A class for performing Latent Dirichlet Allocation (LDA) on a text corpus.

    This class handles the preprocessing of text data, training of the LDA model,
    and visualization of the results. It provides methods to manage topics and
    coherence values, enabling effective topic modeling on the provided corpus.
    """

    def __init__(
        self,
        num_of_topics: int,
        prelemma_corpus: str = None,
    ) -> None:
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
        self.prelemma_corpus: str = prelemma_corpus
        self.passes: int = None
        self.iterations = None
        self.topics: List[str] = []
        self.coherence_values: List[float] = []

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, name: str, value) -> None:
        return setattr(self, name, value)

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

    class LDAModelWorker:
        """
        A class for managing the training of a Latent Dirichlet Allocation (LDA) model.
        This class handles input validation, data preprocessing, and model training, allowing for asynchronous operations through callbacks.

        Attributes:
            _tokens (list): A list to store tokenized data.
            prelemma_corpus (list): The corpus before lemmatization.
            id2word (MappingDictionary): A mapping from word IDs to words.
            corpus (list): The bag-of-words representation of the corpus.
            topics (list): A list to store the number of topics evaluated.
            coherence_values (list): A list to store coherence values for each topic count.
        """

        def __init__(self, on_status: Callable[[str], None]):
            """
            Initializes the LDAModelWorker with empty attributes for tokens, corpus, and model evaluation metrics.
            """
            self.lock = threading.Lock()
            self.on_status = on_status  # A callback to report status or progress
            self.executor = ThreadPoolExecutor(max_workers=4)

        def _validate_inputs(self, number_of_topics, iterations, passes):
            """
            Validates the user inputs for model training parameters.
            This method checks that the inputs are integers and within acceptable ranges.

            Args:
                number_of_topics (str): The number of topics to be used in the model.
                iterations (str): The number of iterations for the training process.
                passes (str): The number of passes for the training process.

            Returns:
                tuple: A tuple containing a boolean indicating validity and an error message.
            """
            if not all(map(str.isdigit, [number_of_topics, iterations, passes])):
                return False, "All inputs must be integers."
            if int(passes) >= 20 or int(iterations) >= 200:
                return False, "Passes should be < 20 and iterations < 200."
            return True, ""

        def train_model(
            self, passes: int, iterations: int, number_of_topics: int, callback=None
        ):
            """
            Trains the model using the specified parameters for topics, iterations, and passes.
            This method validates the input values and initiates the training process in a separate thread.

            Args:
                passes (int): The number of passes for the training process.
                iterations (int): The number of iterations for the training process.
                number_of_topics (int): The number of topics to be used in the model.
                callback (function, optional): A callback function to handle success or error messages.

            Returns:
                None
            """
            logger.info("Starting the Model Training..")

            valid, error_message = self._validate_inputs(
                number_of_topics, iterations, passes
            )
            if not valid:
                logger.error(error_message)
                if callback:
                    callback(error=error_message)
                return

            def train():
                if self.model_trained(
                    iterations=iterations,
                    workers=4,
                    passes=passes,
                    num_of_topics=number_of_topics,
                ):
                    logger.success("Model successfully trained!")
                    if callback:
                        callback(success=True)

            thread = threading.Thread(target=train)
            thread.start()

        def process_corpus(
            self,
            nlp,
            stopwords,
            wrangler_instance,
            progress=gr.Progress(track_tqdm=True),
        ):
            """
            Pre-processes the data by reshaping the corpus and generating tokens from the input text.
            """
            logger.info(
                "Configured SpaCy Model and NLTK Stopwords...Initiating Data Cleanse and Dictionary Creation"
            )

            try:
                if self.prelemma_corpus:
                    logger.info("Lemmatized Corpus...")
                    unprocessed_corpus = nlp.pipe(self.prelemma_corpus)
                    for i, comment in tqdm(enumerate(unprocessed_corpus)):
                        proj_tok = [
                            token.lemma_.lower()
                            for token in comment
                            if (
                                token.pos_
                                not in [
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
                                and token.lemma_.lower() not in stopwords
                                and not token.is_stop
                                and token.is_alpha
                            )
                        ]
                        self._tokens.append(proj_tok)

                    logger.info("Lemmatization Completed")
                else:
                    logger.info(
                        "Prelemma Corpus is empty. Regenerating Corpus using DataWrangler"
                    )
                    self.prelemma_corpus = wrangler_instance.create_corpus()

                logger.debug(f"Token Length:{len(self._tokens)}")
                self.id2word = MappingDictionary(self._tokens)
                self.id2word.filter_extremes(no_below=5, no_above=0.5, keep_n=5000)
                self.corpus = [self.id2word.doc2bow(doc) for doc in self._tokens]

                logger.success("Successfully Processed Corpus")

            except Exception as e:
                logger.exception("FAILED to Process Data")

        def preprocess_input_data(self, nlp, wrangler_instance=None, callback=None):
            """
            Prepares the data for further processing.
            This method calls the process_corpus function and handles any exceptions that may occur.

            Args:
                nlp: The natural language processing model used for tokenization and lemmatization.
                wrangler_instance: An instance of DataWrangler used to regenerate the corpus if necessary.
                callback (function, optional): A callback function to handle progress updates or errors.

            Returns:
                None
            """
            try:
                self.process_corpus(
                    nlp, stopwords, wrangler_instance, callback=callback
                )
            except Exception as e:
                logger.exception("Failed to Preprocess Data")
                if callback:
                    callback(error=str(e))

        def model_trained(self, callback=None):
            def train(progress=gr.Progress(track_tqdm=True)):
                """
                Trains the LDA model concurrently for a specified range of topic counts.
                This function utilizes a thread pool to train multiple models simultaneously and collects their coherence values.

                Args:
                    workers (int): The number of worker threads to use during training.
                    num_of_topics (int): The number of topics to evaluate during training.
                    iterations (int): The number of iterations for the LDA model training.
                    passes (int): The number of passes through the corpus during training.
                    callback (function, optional): A callback function to handle progress updates or errors.

                Returns:
                    None

                Raises:
                    Exception: Logs an error if the training process fails.
                """
                try:
                    with self.executor as executor:
                        futures = []
                        for i in tqdm(range(1, self.num_topics + 1)):
                            futures.append(
                                executor.submit(
                                    self._train_single_model,
                                    i,
                                    self.iterations,
                                    self.passes,
                                )
                            )

                        for future in futures:
                            result = future.result()  # Blocks until the result is ready
                            self.topics.append(result["topics"])
                            self.coherence_values.append(result["coherence"])
                            if callback:
                                callback(progress=result["topics"])

                    if callback:
                        callback(success=True)
                    logger.success("Successfully Trained Model")
                except Exception as e:
                    logger.exception("Failed to Train Model")
                    if callback:
                        callback(error=str(e))

            thread = threading.Thread(target=train)
            thread.start()

        def _train_single_model(
            self,
            i,
            iterations,
            passes,
        ):
            lda_model = self.get_lda_model(
                iterations=iterations, workers=4, passes=passes, num_of_topics=i
            )
            """
            Trains a single LDA model for a specified number of topics and evaluates its coherence.
            This method is called by model_trained to train individual models concurrently.

            Args:
                i (int): The number of topics for the model.
                iterations (int): The number of iterations for the LDA model training.
                passes (int): The number of passes through the corpus during training.

            Returns:
                dict: A dictionary containing the number of topics and the corresponding coherence value.
            """
            cm = CoherenceModel(
                model=lda_model,
                corpus=self.corpus,
                dictionary=self.id2word,
                coherence="c_v",
            )
            return {"topics": i, "coherence": cm.get_coherence()}
