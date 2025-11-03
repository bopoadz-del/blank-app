"""
Tokenization Utilities
Word-level, subword (BPE), and character-level tokenization
"""

import re
from typing import List, Optional, Union, Dict
from collections import Counter
import logging
import json

try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize, wordpunct_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logging.warning("NLTK not available. Install with: pip install nltk")

try:
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from tokenizers.processors import TemplateProcessing
    TOKENIZERS_AVAILABLE = True
except ImportError:
    TOKENIZERS_AVAILABLE = False
    logging.warning("tokenizers not available. Install with: pip install tokenizers")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WordTokenizer:
    """
    Word-level tokenization

    Splits text into words using various strategies:
    - Whitespace splitting
    - NLTK word tokenizer (better handling of punctuation)
    - Regex-based tokenization
    - Custom pattern matching

    Best for:
    - Traditional NLP tasks
    - Bag-of-words models
    - TF-IDF vectorization
    - Simple text processing
    """

    def __init__(
        self,
        method: str = 'nltk',
        lowercase: bool = True,
        remove_punctuation: bool = False,
        min_token_length: int = 1,
        max_vocab_size: Optional[int] = None,
        special_tokens: Optional[List[str]] = None
    ):
        """
        Initialize word tokenizer

        Args:
            method: Tokenization method ('nltk', 'whitespace', 'wordpunct', 'regex')
            lowercase: Convert to lowercase
            remove_punctuation: Remove punctuation from tokens
            min_token_length: Minimum token length
            max_vocab_size: Maximum vocabulary size (None = unlimited)
            special_tokens: Special tokens to preserve (e.g., ['<PAD>', '<UNK>'])
        """
        self.method = method
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.min_token_length = min_token_length
        self.max_vocab_size = max_vocab_size

        self.special_tokens = special_tokens or ['<PAD>', '<UNK>', '<SOS>', '<EOS>']
        self.vocab = {}
        self.inverse_vocab = {}
        self.token_counts = Counter()
        self.fitted = False

        if method == 'nltk' and not NLTK_AVAILABLE:
            logger.warning("NLTK not available, falling back to whitespace tokenization")
            self.method = 'whitespace'

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize a single text

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        if not text:
            return []

        # Apply method-specific tokenization
        if self.method == 'nltk':
            try:
                tokens = word_tokenize(text)
            except LookupError:
                nltk.download('punkt')
                tokens = word_tokenize(text)
        elif self.method == 'wordpunct':
            if NLTK_AVAILABLE:
                tokens = wordpunct_tokenize(text)
            else:
                tokens = text.split()
        elif self.method == 'regex':
            # Use regex to find word characters
            tokens = re.findall(r'\b\w+\b', text)
        else:  # whitespace
            tokens = text.split()

        # Apply filters
        if self.lowercase:
            tokens = [t.lower() for t in tokens]

        if self.remove_punctuation:
            tokens = [re.sub(r'[^\w\s]', '', t) for t in tokens]
            tokens = [t for t in tokens if t]  # Remove empty strings

        if self.min_token_length > 1:
            tokens = [t for t in tokens if len(t) >= self.min_token_length]

        return tokens

    def tokenize_batch(self, texts: List[str]) -> List[List[str]]:
        """Tokenize a batch of texts"""
        return [self.tokenize(text) for text in texts]

    def fit(self, texts: List[str]):
        """
        Build vocabulary from texts

        Args:
            texts: List of training texts
        """
        logger.info(f"Building vocabulary from {len(texts)} texts...")

        # Collect all tokens
        for text in texts:
            tokens = self.tokenize(text)
            self.token_counts.update(tokens)

        # Add special tokens
        for i, token in enumerate(self.special_tokens):
            self.vocab[token] = i

        # Add most common tokens
        most_common = self.token_counts.most_common(self.max_vocab_size)

        for token, count in most_common:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)

        # Create inverse mapping
        self.inverse_vocab = {idx: token for token, idx in self.vocab.items()}

        self.fitted = True
        logger.info(f"Vocabulary built. Size: {len(self.vocab)}")
        logger.info(f"Most common tokens: {self.token_counts.most_common(10)}")

        return self

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to token IDs

        Args:
            text: Input text
            add_special_tokens: Add <SOS> and <EOS> tokens

        Returns:
            List of token IDs
        """
        if not self.fitted:
            raise ValueError("Tokenizer not fitted. Call fit() first.")

        tokens = self.tokenize(text)

        # Convert to IDs
        unk_id = self.vocab.get('<UNK>', 1)
        token_ids = [self.vocab.get(token, unk_id) for token in tokens]

        if add_special_tokens:
            sos_id = self.vocab.get('<SOS>', 2)
            eos_id = self.vocab.get('<EOS>', 3)
            token_ids = [sos_id] + token_ids + [eos_id]

        return token_ids

    def encode_batch(self, texts: List[str], add_special_tokens: bool = True) -> List[List[int]]:
        """Encode a batch of texts"""
        return [self.encode(text, add_special_tokens) for text in texts]

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs to text

        Args:
            token_ids: List of token IDs
            skip_special_tokens: Skip special tokens in output

        Returns:
            Decoded text
        """
        if not self.fitted:
            raise ValueError("Tokenizer not fitted. Call fit() first.")

        tokens = []
        for idx in token_ids:
            token = self.inverse_vocab.get(idx, '<UNK>')

            if skip_special_tokens and token in self.special_tokens:
                continue

            tokens.append(token)

        return ' '.join(tokens)

    def decode_batch(self, token_ids_batch: List[List[int]], skip_special_tokens: bool = True) -> List[str]:
        """Decode a batch of token IDs"""
        return [self.decode(token_ids, skip_special_tokens) for token_ids in token_ids_batch]

    def get_vocab_size(self) -> int:
        """Get vocabulary size"""
        return len(self.vocab)

    def save(self, path: str):
        """Save tokenizer to file"""
        data = {
            'vocab': self.vocab,
            'token_counts': dict(self.token_counts),
            'method': self.method,
            'lowercase': self.lowercase,
            'remove_punctuation': self.remove_punctuation,
            'min_token_length': self.min_token_length,
            'max_vocab_size': self.max_vocab_size,
            'special_tokens': self.special_tokens
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Tokenizer saved to {path}")

    def load(self, path: str):
        """Load tokenizer from file"""
        with open(path, 'r') as f:
            data = json.load(f)

        self.vocab = data['vocab']
        self.token_counts = Counter(data['token_counts'])
        self.method = data['method']
        self.lowercase = data['lowercase']
        self.remove_punctuation = data['remove_punctuation']
        self.min_token_length = data['min_token_length']
        self.max_vocab_size = data['max_vocab_size']
        self.special_tokens = data['special_tokens']

        self.inverse_vocab = {idx: token for token, idx in self.vocab.items()}
        self.fitted = True

        logger.info(f"Tokenizer loaded from {path}. Vocab size: {len(self.vocab)}")
        return self


class BPETokenizer:
    """
    Byte-Pair Encoding (BPE) Tokenizer

    Subword tokenization that learns merge rules from training data.
    Splits words into frequent subword units.

    Best for:
    - Handling rare/unknown words
    - Multilingual models
    - Character-rich languages
    - Modern transformers (BERT, GPT)

    Advantages:
    - Better handling of OOV words
    - Smaller vocabulary size
    - Better for morphologically rich languages
    """

    def __init__(
        self,
        vocab_size: int = 10000,
        min_frequency: int = 2,
        special_tokens: Optional[List[str]] = None
    ):
        """
        Initialize BPE tokenizer

        Args:
            vocab_size: Target vocabulary size
            min_frequency: Minimum frequency for tokens
            special_tokens: Special tokens (e.g., ['<PAD>', '<UNK>', '<SOS>', '<EOS>'])
        """
        if not TOKENIZERS_AVAILABLE:
            raise ImportError("tokenizers library not installed. Install with: pip install tokenizers")

        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.special_tokens = special_tokens or ['<PAD>', '<UNK>', '<SOS>', '<EOS>']

        # Create tokenizer with BPE model
        self.tokenizer = Tokenizer(models.BPE(unk_token='<UNK>'))
        self.tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

        self.fitted = False

    def fit(self, texts: List[str]):
        """
        Train BPE tokenizer on texts

        Args:
            texts: List of training texts
        """
        logger.info(f"Training BPE tokenizer on {len(texts)} texts...")
        logger.info(f"Target vocab size: {self.vocab_size}")

        # Create trainer
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=self.special_tokens
        )

        # Train
        self.tokenizer.train_from_iterator(texts, trainer=trainer)

        # Add post-processing for special tokens
        self.tokenizer.post_processor = TemplateProcessing(
            single="<SOS> $A <EOS>",
            pair="<SOS> $A <EOS> $B:1 <EOS>:1",
            special_tokens=[
                ("<SOS>", self.tokenizer.token_to_id("<SOS>")),
                ("<EOS>", self.tokenizer.token_to_id("<EOS>")),
            ],
        )

        self.fitted = True
        logger.info(f"BPE training complete. Vocab size: {self.tokenizer.get_vocab_size()}")

        return self

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to token IDs

        Args:
            text: Input text
            add_special_tokens: Add special tokens

        Returns:
            List of token IDs
        """
        if not self.fitted:
            raise ValueError("Tokenizer not fitted. Call fit() first.")

        encoding = self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
        return encoding.ids

    def encode_batch(self, texts: List[str], add_special_tokens: bool = True) -> List[List[int]]:
        """Encode a batch of texts"""
        encodings = self.tokenizer.encode_batch(texts, add_special_tokens=add_special_tokens)
        return [enc.ids for enc in encodings]

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs to text

        Args:
            token_ids: List of token IDs
            skip_special_tokens: Skip special tokens

        Returns:
            Decoded text
        """
        if not self.fitted:
            raise ValueError("Tokenizer not fitted. Call fit() first.")

        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def decode_batch(self, token_ids_batch: List[List[int]], skip_special_tokens: bool = True) -> List[str]:
        """Decode a batch of token IDs"""
        return [self.decode(ids, skip_special_tokens) for ids in token_ids_batch]

    def tokenize(self, text: str) -> List[str]:
        """Get token strings (not IDs)"""
        if not self.fitted:
            raise ValueError("Tokenizer not fitted. Call fit() first.")

        encoding = self.tokenizer.encode(text)
        return encoding.tokens

    def get_vocab_size(self) -> int:
        """Get vocabulary size"""
        return self.tokenizer.get_vocab_size()

    def save(self, path: str):
        """Save tokenizer to file"""
        self.tokenizer.save(path)
        logger.info(f"BPE tokenizer saved to {path}")

    def load(self, path: str):
        """Load tokenizer from file"""
        self.tokenizer = Tokenizer.from_file(path)
        self.fitted = True
        logger.info(f"BPE tokenizer loaded from {path}")
        return self


class CharacterTokenizer:
    """
    Character-level tokenization

    Treats each character as a separate token.

    Best for:
    - Very small datasets
    - Character-level language models
    - Handling any text without vocabulary issues
    - Text generation at character level

    Advantages:
    - No OOV problem
    - Very small vocabulary
    - Works for any language

    Disadvantages:
    - Long sequences
    - May lose word-level semantics
    """

    def __init__(
        self,
        lowercase: bool = True,
        special_tokens: Optional[List[str]] = None,
        max_vocab_size: Optional[int] = None
    ):
        """
        Initialize character tokenizer

        Args:
            lowercase: Convert to lowercase
            special_tokens: Special tokens
            max_vocab_size: Maximum vocabulary size (None = unlimited)
        """
        self.lowercase = lowercase
        self.special_tokens = special_tokens or ['<PAD>', '<UNK>', '<SOS>', '<EOS>']
        self.max_vocab_size = max_vocab_size

        self.vocab = {}
        self.inverse_vocab = {}
        self.char_counts = Counter()
        self.fitted = False

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into characters

        Args:
            text: Input text

        Returns:
            List of characters
        """
        if self.lowercase:
            text = text.lower()

        return list(text)

    def fit(self, texts: List[str]):
        """
        Build character vocabulary from texts

        Args:
            texts: List of training texts
        """
        logger.info(f"Building character vocabulary from {len(texts)} texts...")

        # Collect all characters
        for text in texts:
            chars = self.tokenize(text)
            self.char_counts.update(chars)

        # Add special tokens
        for i, token in enumerate(self.special_tokens):
            self.vocab[token] = i

        # Add most common characters
        most_common = self.char_counts.most_common(self.max_vocab_size)

        for char, count in most_common:
            if char not in self.vocab:
                self.vocab[char] = len(self.vocab)

        # Create inverse mapping
        self.inverse_vocab = {idx: char for char, idx in self.vocab.items()}

        self.fitted = True
        logger.info(f"Character vocabulary built. Size: {len(self.vocab)}")
        logger.info(f"Most common chars: {self.char_counts.most_common(20)}")

        return self

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to character IDs

        Args:
            text: Input text
            add_special_tokens: Add <SOS> and <EOS> tokens

        Returns:
            List of character IDs
        """
        if not self.fitted:
            raise ValueError("Tokenizer not fitted. Call fit() first.")

        chars = self.tokenize(text)

        # Convert to IDs
        unk_id = self.vocab.get('<UNK>', 1)
        char_ids = [self.vocab.get(char, unk_id) for char in chars]

        if add_special_tokens:
            sos_id = self.vocab.get('<SOS>', 2)
            eos_id = self.vocab.get('<EOS>', 3)
            char_ids = [sos_id] + char_ids + [eos_id]

        return char_ids

    def decode(self, char_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode character IDs to text

        Args:
            char_ids: List of character IDs
            skip_special_tokens: Skip special tokens

        Returns:
            Decoded text
        """
        if not self.fitted:
            raise ValueError("Tokenizer not fitted. Call fit() first.")

        chars = []
        for idx in char_ids:
            char = self.inverse_vocab.get(idx, '<UNK>')

            if skip_special_tokens and char in self.special_tokens:
                continue

            chars.append(char)

        return ''.join(chars)

    def get_vocab_size(self) -> int:
        """Get vocabulary size"""
        return len(self.vocab)

    def save(self, path: str):
        """Save tokenizer to file"""
        data = {
            'vocab': self.vocab,
            'char_counts': dict(self.char_counts),
            'lowercase': self.lowercase,
            'max_vocab_size': self.max_vocab_size,
            'special_tokens': self.special_tokens
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Character tokenizer saved to {path}")

    def load(self, path: str):
        """Load tokenizer from file"""
        with open(path, 'r') as f:
            data = json.load(f)

        self.vocab = data['vocab']
        self.char_counts = Counter(data['char_counts'])
        self.lowercase = data['lowercase']
        self.max_vocab_size = data['max_vocab_size']
        self.special_tokens = data['special_tokens']

        self.inverse_vocab = {idx: char for char, idx in self.vocab.items()}
        self.fitted = True

        logger.info(f"Character tokenizer loaded from {path}. Vocab size: {len(self.vocab)}")
        return self


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("Tokenization Test")
    print("=" * 70)

    sample_texts = [
        "This is a sample text for tokenization testing.",
        "Machine learning models require proper tokenization.",
        "Byte-pair encoding is great for handling rare words.",
        "Character-level tokenization works for any language!"
    ]

    # Test Word Tokenizer
    print("\n1. Word Tokenizer")
    print("-" * 70)

    word_tokenizer = WordTokenizer(method='nltk', lowercase=True, max_vocab_size=50)
    word_tokenizer.fit(sample_texts)

    test_text = "This is a new testing sample."
    tokens = word_tokenizer.tokenize(test_text)
    encoded = word_tokenizer.encode(test_text)
    decoded = word_tokenizer.decode(encoded)

    print(f"Text: {test_text}")
    print(f"Tokens: {tokens}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    print(f"Vocab size: {word_tokenizer.get_vocab_size()}")

    # Test BPE Tokenizer
    if TOKENIZERS_AVAILABLE:
        print("\n2. BPE Tokenizer")
        print("-" * 70)

        bpe_tokenizer = BPETokenizer(vocab_size=100, min_frequency=1)
        bpe_tokenizer.fit(sample_texts)

        tokens_bpe = bpe_tokenizer.tokenize(test_text)
        encoded_bpe = bpe_tokenizer.encode(test_text)
        decoded_bpe = bpe_tokenizer.decode(encoded_bpe)

        print(f"Text: {test_text}")
        print(f"Tokens: {tokens_bpe}")
        print(f"Encoded: {encoded_bpe}")
        print(f"Decoded: {decoded_bpe}")
        print(f"Vocab size: {bpe_tokenizer.get_vocab_size()}")
    else:
        print("\n2. BPE Tokenizer: Not available (tokenizers library not installed)")

    # Test Character Tokenizer
    print("\n3. Character Tokenizer")
    print("-" * 70)

    char_tokenizer = CharacterTokenizer(lowercase=True)
    char_tokenizer.fit(sample_texts)

    tokens_char = char_tokenizer.tokenize(test_text)
    encoded_char = char_tokenizer.encode(test_text)
    decoded_char = char_tokenizer.decode(encoded_char)

    print(f"Text: {test_text}")
    print(f"Tokens (first 20): {tokens_char[:20]}...")
    print(f"Encoded (first 20): {encoded_char[:20]}...")
    print(f"Decoded: {decoded_char}")
    print(f"Vocab size: {char_tokenizer.get_vocab_size()}")

    print("\n" + "=" * 70)
    print("Tokenization tested successfully!")
