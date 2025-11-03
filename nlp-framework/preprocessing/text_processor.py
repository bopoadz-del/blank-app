"""
Text Preprocessing Utilities
Cleaning, normalization, and preprocessing for NLP tasks
"""

import re
import string
from typing import List, Optional, Union
import logging

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer, PorterStemmer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logging.warning("NLTK not available. Install with: pip install nltk")

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logging.warning("spaCy not available. Install with: pip install spacy")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextProcessor:
    """
    Comprehensive text preprocessing pipeline

    Features:
    - Lowercasing
    - Punctuation removal
    - Stopword removal
    - Lemmatization/Stemming
    - HTML/URL removal
    - Number handling
    - Special character removal
    """

    def __init__(
        self,
        lowercase: bool = True,
        remove_punctuation: bool = False,
        remove_stopwords: bool = False,
        remove_numbers: bool = False,
        remove_html: bool = True,
        remove_urls: bool = True,
        remove_emails: bool = True,
        lemmatize: bool = False,
        stem: bool = False,
        language: str = 'english',
        custom_stopwords: Optional[List[str]] = None
    ):
        """
        Initialize text processor

        Args:
            lowercase: Convert text to lowercase
            remove_punctuation: Remove punctuation marks
            remove_stopwords: Remove stopwords
            remove_numbers: Remove numbers
            remove_html: Remove HTML tags
            remove_urls: Remove URLs
            remove_emails: Remove email addresses
            lemmatize: Apply lemmatization
            stem: Apply stemming
            language: Language for stopwords
            custom_stopwords: Additional stopwords to remove
        """
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        self.remove_numbers = remove_numbers
        self.remove_html = remove_html
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.lemmatize = lemmatize
        self.stem = stem
        self.language = language

        # Initialize stopwords
        if self.remove_stopwords:
            if NLTK_AVAILABLE:
                try:
                    self.stopwords = set(stopwords.words(language))
                except LookupError:
                    logger.warning(f"Stopwords for {language} not found. Downloading...")
                    nltk.download('stopwords')
                    self.stopwords = set(stopwords.words(language))
            else:
                self.stopwords = set()

            if custom_stopwords:
                self.stopwords.update(custom_stopwords)

        # Initialize lemmatizer/stemmer
        if self.lemmatize and NLTK_AVAILABLE:
            try:
                self.lemmatizer = WordNetLemmatizer()
            except LookupError:
                nltk.download('wordnet')
                nltk.download('omw-1.4')
                self.lemmatizer = WordNetLemmatizer()

        if self.stem and NLTK_AVAILABLE:
            self.stemmer = PorterStemmer()

    def process(self, text: str) -> str:
        """
        Process a single text

        Args:
            text: Input text

        Returns:
            Processed text
        """
        if not text:
            return ""

        # Remove HTML tags
        if self.remove_html:
            text = self._remove_html(text)

        # Remove URLs
        if self.remove_urls:
            text = self._remove_urls(text)

        # Remove emails
        if self.remove_emails:
            text = self._remove_emails(text)

        # Lowercase
        if self.lowercase:
            text = text.lower()

        # Remove numbers
        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)

        # Remove punctuation
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))

        # Tokenize for word-level operations
        if self.remove_stopwords or self.lemmatize or self.stem:
            if NLTK_AVAILABLE:
                tokens = word_tokenize(text)
            else:
                tokens = text.split()

            # Remove stopwords
            if self.remove_stopwords:
                tokens = [t for t in tokens if t.lower() not in self.stopwords]

            # Lemmatize
            if self.lemmatize and hasattr(self, 'lemmatizer'):
                tokens = [self.lemmatizer.lemmatize(t) for t in tokens]

            # Stem
            if self.stem and hasattr(self, 'stemmer'):
                tokens = [self.stemmer.stem(t) for t in tokens]

            text = ' '.join(tokens)

        # Clean extra whitespace
        text = ' '.join(text.split())

        return text

    def process_batch(self, texts: List[str]) -> List[str]:
        """
        Process a batch of texts

        Args:
            texts: List of input texts

        Returns:
            List of processed texts
        """
        return [self.process(text) for text in texts]

    def _remove_html(self, text: str) -> str:
        """Remove HTML tags"""
        clean = re.compile('<.*?>')
        return re.sub(clean, '', text)

    def _remove_urls(self, text: str) -> str:
        """Remove URLs"""
        pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return re.sub(pattern, '', text)

    def _remove_emails(self, text: str) -> str:
        """Remove email addresses"""
        pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return re.sub(pattern, '', text)


class SpacyProcessor:
    """
    Text processor using spaCy
    Provides POS tagging, NER, dependency parsing
    """

    def __init__(self, model: str = 'en_core_web_sm'):
        """
        Initialize spaCy processor

        Args:
            model: spaCy model name
        """
        if not SPACY_AVAILABLE:
            raise ImportError("spaCy not installed. Install with: pip install spacy")

        try:
            self.nlp = spacy.load(model)
        except OSError:
            logger.warning(f"Model {model} not found. Downloading...")
            import subprocess
            subprocess.run(['python', '-m', 'spacy', 'download', model])
            self.nlp = spacy.load(model)

    def process(self, text: str) -> dict:
        """
        Process text with spaCy

        Args:
            text: Input text

        Returns:
            Dictionary with tokens, pos_tags, entities, etc.
        """
        doc = self.nlp(text)

        return {
            'tokens': [token.text for token in doc],
            'lemmas': [token.lemma_ for token in doc],
            'pos_tags': [token.pos_ for token in doc],
            'entities': [(ent.text, ent.label_) for ent in doc.ents],
            'noun_chunks': [chunk.text for chunk in doc.noun_chunks]
        }

    def extract_entities(self, text: str) -> List[tuple]:
        """Extract named entities"""
        doc = self.nlp(text)
        return [(ent.text, ent.label_) for ent in doc.ents]

    def get_pos_tags(self, text: str) -> List[tuple]:
        """Get POS tags"""
        doc = self.nlp(text)
        return [(token.text, token.pos_) for token in doc]


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("Text Preprocessing Test")
    print("=" * 70)

    sample_text = """
    This is a SAMPLE text with HTML <b>tags</b>, URLs http://example.com,
    emails test@example.com, and numbers 12345. It also has punctuation!!!
    """

    # Test basic processor
    print("\n1. Basic Text Processor")
    print("-" * 70)
    processor = TextProcessor(
        lowercase=True,
        remove_punctuation=True,
        remove_stopwords=True,
        remove_html=True,
        remove_urls=True,
        remove_emails=True,
        lemmatize=True
    )

    cleaned = processor.process(sample_text)
    print(f"Original: {sample_text.strip()}")
    print(f"Cleaned: {cleaned}")

    # Test batch processing
    print("\n2. Batch Processing")
    print("-" * 70)
    texts = [
        "First text with some content.",
        "Second text with different words.",
        "Third text for testing purposes."
    ]

    cleaned_batch = processor.process_batch(texts)
    for i, (original, cleaned) in enumerate(zip(texts, cleaned_batch), 1):
        print(f"{i}. {original} -> {cleaned}")

    # Test spaCy processor (if available)
    if SPACY_AVAILABLE:
        print("\n3. spaCy Processor")
        print("-" * 70)
        try:
            spacy_processor = SpacyProcessor()
            result = spacy_processor.process("Apple Inc. is looking at buying U.K. startup for $1 billion")
            print(f"Entities: {result['entities']}")
            print(f"POS tags: {result['pos_tags'][:5]}...")
        except Exception as e:
            print(f"spaCy processing failed: {e}")

    print("\n" + "=" * 70)
    print("Text preprocessing tested successfully!")
