"""
NLP Framework
Complete Natural Language Processing toolkit
"""

__version__ = "1.0.0"

# Import main components for easy access
try:
    from .preprocessing.text_processor import TextProcessor, SpacyProcessor
    from .preprocessing.tokenizer import WordTokenizer, BPETokenizer, CharacterTokenizer
except ImportError:
    pass

try:
    from .embeddings.word2vec import Word2VecEmbeddings
    from .embeddings.bert_embeddings import BERTEmbeddings, SentenceTransformer
except ImportError:
    pass

try:
    from .models.text_classifier import TextCNN, BiLSTM, BERTClassifier
    from .models.ner import BiLSTM_CRF, BERT_NER, NER_TAG_SCHEMES
    from .models.sentiment_analyzer import SentimentAnalyzer, LexiconBasedSentiment
except ImportError:
    pass

__all__ = [
    # Preprocessing
    'TextProcessor',
    'SpacyProcessor',
    'WordTokenizer',
    'BPETokenizer',
    'CharacterTokenizer',

    # Embeddings
    'Word2VecEmbeddings',
    'BERTEmbeddings',
    'SentenceTransformer',

    # Models
    'TextCNN',
    'BiLSTM',
    'BERTClassifier',
    'BiLSTM_CRF',
    'BERT_NER',
    'SentimentAnalyzer',
    'LexiconBasedSentiment',

    # Constants
    'NER_TAG_SCHEMES',
]
