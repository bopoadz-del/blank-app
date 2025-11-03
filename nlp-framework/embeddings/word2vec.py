"""
Word2Vec Embeddings
Train and use Word2Vec word embeddings
"""

import numpy as np
from typing import List, Optional, Union, Tuple
import logging

try:
    from gensim.models import Word2Vec
    from gensim.models.callbacks import CallbackAny2Vec
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False
    logging.warning("gensim not available. Install with: pip install gensim")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EpochLogger(CallbackAny2Vec):
    """Callback to log training progress"""

    def __init__(self):
        self.epoch = 0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        logger.info(f"Epoch {self.epoch}: loss = {loss:.4f}")
        self.epoch += 1


class Word2VecEmbeddings:
    """
    Word2Vec Word Embeddings

    Learns distributed representations of words from text.
    Words with similar context have similar vectors.

    Two algorithms:
    - Skip-gram (sg=1): Predicts context from target word. Better for rare words.
    - CBOW (sg=0): Predicts target word from context. Faster, better for frequent words.

    Best for:
    - Finding semantically similar words
    - Word analogies (king - man + woman = queen)
    - Feature extraction for downstream tasks
    - Understanding word relationships
    - Transfer learning for NLP

    Applications:
    - Text classification
    - Sentiment analysis
    - Named entity recognition
    - Document clustering
    - Recommendation systems
    """

    def __init__(
        self,
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 5,
        sg: int = 0,
        hs: int = 0,
        negative: int = 5,
        epochs: int = 10,
        workers: int = 4,
        seed: int = 42
    ):
        """
        Initialize Word2Vec model

        Args:
            vector_size: Dimensionality of word vectors
            window: Maximum distance between current and predicted word
            min_count: Minimum word frequency (ignore words with freq < min_count)
            sg: Training algorithm (0=CBOW, 1=Skip-gram)
            hs: Use hierarchical softmax (1) or negative sampling (0)
            negative: Number of negative samples (0=no negative sampling)
            epochs: Number of training epochs
            workers: Number of worker threads
            seed: Random seed
        """
        if not GENSIM_AVAILABLE:
            raise ImportError("gensim not installed. Install with: pip install gensim")

        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.sg = sg
        self.hs = hs
        self.negative = negative
        self.epochs = epochs
        self.workers = workers
        self.seed = seed

        self.model = None
        self.trained = False

    def train(self, sentences: List[List[str]], verbose: bool = True):
        """
        Train Word2Vec model

        Args:
            sentences: List of tokenized sentences (each sentence is a list of words)
            verbose: Show training progress

        Example:
            sentences = [
                ['i', 'love', 'machine', 'learning'],
                ['word', 'embeddings', 'are', 'powerful'],
                ['word2vec', 'is', 'great']
            ]
        """
        logger.info("=" * 70)
        logger.info("Training Word2Vec")
        logger.info("=" * 70)
        logger.info(f"Algorithm: {'Skip-gram' if self.sg else 'CBOW'}")
        logger.info(f"Vector size: {self.vector_size}")
        logger.info(f"Window: {self.window}")
        logger.info(f"Min count: {self.min_count}")
        logger.info(f"Epochs: {self.epochs}")
        logger.info(f"Training on {len(sentences)} sentences...")

        # Create model
        self.model = Word2Vec(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            sg=self.sg,
            hs=self.hs,
            negative=self.negative,
            epochs=self.epochs,
            workers=self.workers,
            seed=self.seed,
            compute_loss=verbose,
            callbacks=[EpochLogger()] if verbose else []
        )

        self.trained = True

        logger.info("Training complete!")
        logger.info(f"Vocabulary size: {len(self.model.wv)}")
        logger.info("=" * 70)

        return self

    def get_vector(self, word: str) -> Optional[np.ndarray]:
        """
        Get vector for a word

        Args:
            word: Word to get vector for

        Returns:
            Word vector or None if word not in vocabulary
        """
        if not self.trained:
            raise ValueError("Model not trained. Call train() first.")

        try:
            return self.model.wv[word]
        except KeyError:
            logger.warning(f"Word '{word}' not in vocabulary")
            return None

    def get_vectors(self, words: List[str]) -> np.ndarray:
        """
        Get vectors for multiple words

        Args:
            words: List of words

        Returns:
            Array of word vectors (shape: [n_words, vector_size])
        """
        vectors = []
        for word in words:
            vec = self.get_vector(word)
            if vec is not None:
                vectors.append(vec)
            else:
                # Use zero vector for OOV words
                vectors.append(np.zeros(self.vector_size))

        return np.array(vectors)

    def most_similar(
        self,
        positive: Optional[Union[str, List[str]]] = None,
        negative: Optional[Union[str, List[str]]] = None,
        topn: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Find most similar words

        Args:
            positive: Words to contribute positively
            negative: Words to contribute negatively
            topn: Number of results to return

        Returns:
            List of (word, similarity_score) tuples

        Examples:
            # Find similar words
            model.most_similar('king', topn=5)

            # Word arithmetic
            model.most_similar(positive=['king', 'woman'], negative=['man'])
            # Expected result: queen
        """
        if not self.trained:
            raise ValueError("Model not trained. Call train() first.")

        # Convert strings to lists
        if isinstance(positive, str):
            positive = [positive]
        if isinstance(negative, str):
            negative = [negative]

        try:
            return self.model.wv.most_similar(
                positive=positive or [],
                negative=negative or [],
                topn=topn
            )
        except KeyError as e:
            logger.error(f"Word not in vocabulary: {e}")
            return []

    def similarity(self, word1: str, word2: str) -> float:
        """
        Calculate cosine similarity between two words

        Args:
            word1: First word
            word2: Second word

        Returns:
            Similarity score (0-1)
        """
        if not self.trained:
            raise ValueError("Model not trained. Call train() first.")

        try:
            return float(self.model.wv.similarity(word1, word2))
        except KeyError as e:
            logger.error(f"Word not in vocabulary: {e}")
            return 0.0

    def doesnt_match(self, words: List[str]) -> str:
        """
        Find the word that doesn't match the others

        Args:
            words: List of words

        Returns:
            Word that doesn't match

        Example:
            model.doesnt_match(['breakfast', 'lunch', 'dinner', 'car'])
            # Expected: 'car'
        """
        if not self.trained:
            raise ValueError("Model not trained. Call train() first.")

        try:
            return self.model.wv.doesnt_match(words)
        except KeyError as e:
            logger.error(f"Word not in vocabulary: {e}")
            return ""

    def get_mean_vector(self, words: List[str]) -> np.ndarray:
        """
        Get mean vector of multiple words (document embedding)

        Args:
            words: List of words

        Returns:
            Mean vector
        """
        vectors = self.get_vectors(words)
        return np.mean(vectors, axis=0)

    def get_vocab_size(self) -> int:
        """Get vocabulary size"""
        if not self.trained:
            return 0
        return len(self.model.wv)

    def get_vocab_words(self) -> List[str]:
        """Get all vocabulary words"""
        if not self.trained:
            return []
        return list(self.model.wv.index_to_key)

    def contains_word(self, word: str) -> bool:
        """Check if word is in vocabulary"""
        if not self.trained:
            return False
        return word in self.model.wv

    def save(self, path: str):
        """
        Save Word2Vec model

        Args:
            path: File path to save model
        """
        if not self.trained:
            raise ValueError("Model not trained. Call train() first.")

        self.model.save(path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """
        Load Word2Vec model

        Args:
            path: File path to load model from
        """
        if not GENSIM_AVAILABLE:
            raise ImportError("gensim not installed. Install with: pip install gensim")

        self.model = Word2Vec.load(path)
        self.trained = True

        # Update parameters from loaded model
        self.vector_size = self.model.wv.vector_size
        self.window = self.model.window
        self.min_count = self.model.min_count
        self.sg = self.model.sg
        self.epochs = self.model.epochs

        logger.info(f"Model loaded from {path}")
        logger.info(f"Vocabulary size: {len(self.model.wv)}")

        return self

    def update_training(self, new_sentences: List[List[str]], epochs: int = 5):
        """
        Continue training with new sentences (online learning)

        Args:
            new_sentences: New tokenized sentences
            epochs: Number of additional epochs
        """
        if not self.trained:
            raise ValueError("Model not trained. Call train() first.")

        logger.info(f"Updating model with {len(new_sentences)} new sentences...")

        # Build vocabulary for new sentences
        self.model.build_vocab(new_sentences, update=True)

        # Continue training
        self.model.train(
            new_sentences,
            total_examples=len(new_sentences),
            epochs=epochs
        )

        logger.info(f"Model updated. New vocabulary size: {len(self.model.wv)}")

        return self


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("Word2Vec Embeddings Test")
    print("=" * 70)

    # Sample sentences (tokenized)
    sentences = [
        ['machine', 'learning', 'is', 'amazing'],
        ['deep', 'learning', 'is', 'powerful'],
        ['natural', 'language', 'processing', 'is', 'interesting'],
        ['word', 'embeddings', 'are', 'useful'],
        ['word2vec', 'creates', 'word', 'vectors'],
        ['neural', 'networks', 'learn', 'patterns'],
        ['data', 'science', 'requires', 'machine', 'learning'],
        ['artificial', 'intelligence', 'is', 'transforming', 'industries'],
        ['embeddings', 'capture', 'semantic', 'meaning'],
        ['skip', 'gram', 'and', 'cbow', 'are', 'word2vec', 'algorithms'],
        ['the', 'king', 'is', 'a', 'male', 'ruler'],
        ['the', 'queen', 'is', 'a', 'female', 'ruler'],
        ['man', 'and', 'woman', 'are', 'humans'],
        ['boy', 'and', 'girl', 'are', 'children'],
        ['father', 'and', 'mother', 'are', 'parents']
    ]

    # Test CBOW
    print("\n1. CBOW Model")
    print("-" * 70)

    cbow = Word2VecEmbeddings(
        vector_size=50,
        window=3,
        min_count=1,
        sg=0,  # CBOW
        epochs=50,
        workers=4
    )

    cbow.train(sentences, verbose=False)

    print(f"Vocabulary size: {cbow.get_vocab_size()}")
    print(f"Vector size: {cbow.vector_size}")

    # Test word vector
    word = 'learning'
    vector = cbow.get_vector(word)
    if vector is not None:
        print(f"\nVector for '{word}' (first 10 dims): {vector[:10]}")

    # Test similarity
    print("\n2. Word Similarity")
    print("-" * 70)
    pairs = [
        ('machine', 'learning'),
        ('king', 'queen'),
        ('man', 'woman'),
        ('neural', 'networks')
    ]

    for word1, word2 in pairs:
        sim = cbow.similarity(word1, word2)
        print(f"Similarity('{word1}', '{word2}'): {sim:.4f}")

    # Test most similar
    print("\n3. Most Similar Words")
    print("-" * 70)

    test_words = ['learning', 'word2vec', 'king']
    for word in test_words:
        if cbow.contains_word(word):
            similar = cbow.most_similar(word, topn=3)
            print(f"\nMost similar to '{word}':")
            for sim_word, score in similar:
                print(f"  {sim_word}: {score:.4f}")

    # Test word arithmetic
    print("\n4. Word Arithmetic")
    print("-" * 70)

    # king - man + woman = ?
    if all(cbow.contains_word(w) for w in ['king', 'man', 'woman']):
        result = cbow.most_similar(
            positive=['king', 'woman'],
            negative=['man'],
            topn=3
        )
        print("king - man + woman =")
        for word, score in result:
            print(f"  {word}: {score:.4f}")

    # Test Skip-gram
    print("\n5. Skip-gram Model")
    print("-" * 70)

    skipgram = Word2VecEmbeddings(
        vector_size=50,
        window=3,
        min_count=1,
        sg=1,  # Skip-gram
        epochs=50,
        workers=4
    )

    skipgram.train(sentences, verbose=False)
    print(f"Vocabulary size: {skipgram.get_vocab_size()}")

    # Test document embedding (mean of word vectors)
    print("\n6. Document Embedding")
    print("-" * 70)

    doc = ['machine', 'learning', 'is', 'powerful']
    doc_vector = cbow.get_mean_vector(doc)
    print(f"Document: {' '.join(doc)}")
    print(f"Document vector (first 10 dims): {doc_vector[:10]}")

    print("\n" + "=" * 70)
    print("Word2Vec embeddings tested successfully!")
