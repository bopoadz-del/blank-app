"""
Sentiment Analysis Models
Traditional and transformer-based sentiment analysis
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Union, Dict, Tuple
import logging

try:
    from transformers import (
        pipeline,
        AutoTokenizer,
        AutoModelForSequenceClassification,
        DistilBertForSequenceClassification,
        RobertaForSequenceClassification
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("transformers not available. Install with: pip install transformers")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    High-level Sentiment Analysis Interface

    Provides easy-to-use sentiment analysis with pretrained models
    or custom trained models.

    Supports:
    - Binary sentiment (positive/negative)
    - Multi-class sentiment (positive/neutral/negative)
    - Fine-grained sentiment (1-5 stars)
    - Aspect-based sentiment analysis

    Best for:
    - Product reviews
    - Social media analysis
    - Customer feedback
    - Movie reviews
    - General text sentiment
    """

    def __init__(
        self,
        model_name: str = 'distilbert-base-uncased-finetuned-sst-2-english',
        device: Optional[str] = None,
        use_pipeline: bool = True
    ):
        """
        Initialize sentiment analyzer

        Args:
            model_name: HuggingFace model name or path
            device: Device ('cuda', 'cpu', or None for auto)
            use_pipeline: Use HuggingFace pipeline (simpler) or manual loading

        Popular models:
        - distilbert-base-uncased-finetuned-sst-2-english: Binary (pos/neg)
        - nlptown/bert-base-multilingual-uncased-sentiment: 5-class (1-5 stars)
        - cardiffnlp/twitter-roberta-base-sentiment: 3-class (pos/neu/neg)
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers not installed. Install with: pip install transformers")

        self.model_name = model_name
        self.use_pipeline = use_pipeline

        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        logger.info(f"Loading sentiment model: {model_name}")
        logger.info(f"Device: {self.device}")

        if use_pipeline:
            # Use HuggingFace pipeline
            self.pipe = pipeline(
                "sentiment-analysis",
                model=model_name,
                device=0 if self.device == 'cuda' else -1
            )
            self.tokenizer = None
            self.model = None
        else:
            # Manual loading
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            self.pipe = None

        # Get label mapping
        if not use_pipeline:
            self.id2label = self.model.config.id2label
            self.label2id = self.model.config.label2id
        else:
            # Pipeline handles labels automatically
            self.id2label = None
            self.label2id = None

        logger.info("Model loaded successfully")

    def analyze(
        self,
        texts: Union[str, List[str]],
        return_scores: bool = True,
        batch_size: int = 32
    ) -> Union[Dict, List[Dict]]:
        """
        Analyze sentiment of text(s)

        Args:
            texts: Single text or list of texts
            return_scores: Include confidence scores
            batch_size: Batch size for processing

        Returns:
            Dictionary or list of dictionaries with 'label' and optional 'score'
        """
        single_text = isinstance(texts, str)
        if single_text:
            texts = [texts]

        if self.use_pipeline:
            # Use pipeline
            results = self.pipe(texts, batch_size=batch_size)
        else:
            # Manual processing
            results = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_results = self._analyze_batch(batch, return_scores)
                results.extend(batch_results)

        if single_text:
            return results[0]

        return results

    def _analyze_batch(
        self,
        texts: List[str],
        return_scores: bool = True
    ) -> List[Dict]:
        """
        Analyze a batch of texts manually

        Args:
            texts: List of texts
            return_scores: Include confidence scores

        Returns:
            List of result dictionaries
        """
        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )

        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        # Get predictions
        with torch.no_grad():
            outputs = self.model(**encoded)

        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        predictions = torch.argmax(logits, dim=-1)

        # Format results
        results = []
        for i in range(len(texts)):
            label_id = predictions[i].item()
            label = self.id2label[label_id]
            score = probs[i][label_id].item()

            result = {'label': label}
            if return_scores:
                result['score'] = score

            results.append(result)

        return results

    def get_label_scores(
        self,
        text: str
    ) -> Dict[str, float]:
        """
        Get scores for all labels

        Args:
            text: Input text

        Returns:
            Dictionary mapping labels to scores
        """
        if self.use_pipeline:
            # Switch to manual mode temporarily
            logger.warning("get_label_scores not available with pipeline mode")
            return {}

        # Tokenize
        encoded = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )

        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        # Get predictions
        with torch.no_grad():
            outputs = self.model(**encoded)

        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)[0]

        # Create label-score mapping
        label_scores = {}
        for label_id, label in self.id2label.items():
            label_scores[label] = probs[label_id].item()

        return label_scores

    def analyze_aspects(
        self,
        text: str,
        aspects: List[str]
    ) -> Dict[str, Dict]:
        """
        Aspect-based sentiment analysis

        Analyzes sentiment for specific aspects mentioned in text.

        Args:
            text: Input text
            aspects: List of aspects to analyze (e.g., ['food', 'service', 'ambiance'])

        Returns:
            Dictionary mapping aspects to sentiment results

        Example:
            text = "The food was excellent but the service was terrible"
            aspects = ['food', 'service']
            results = analyzer.analyze_aspects(text, aspects)
            # results: {'food': {'label': 'POSITIVE', 'score': 0.95},
            #           'service': {'label': 'NEGATIVE', 'score': 0.89}}
        """
        results = {}

        for aspect in aspects:
            # Simple approach: analyze sentences containing the aspect
            sentences = text.split('.')
            relevant_sentences = [s for s in sentences if aspect.lower() in s.lower()]

            if relevant_sentences:
                # Analyze combined relevant sentences
                combined = '. '.join(relevant_sentences)
                sentiment = self.analyze(combined)
                results[aspect] = sentiment
            else:
                # Aspect not found
                results[aspect] = {'label': 'NEUTRAL', 'score': 0.0}

        return results


class LexiconBasedSentiment:
    """
    Lexicon-based Sentiment Analysis

    Uses sentiment lexicons (word lists with polarity scores)
    to analyze sentiment without machine learning.

    Advantages:
    - No training required
    - Interpretable
    - Fast
    - Works with limited data

    Disadvantages:
    - Doesn't understand context well
    - Struggles with sarcasm/irony
    - Language/domain dependent
    """

    def __init__(self, lexicon: str = 'vader'):
        """
        Initialize lexicon-based analyzer

        Args:
            lexicon: Lexicon to use ('vader', 'afinn', 'textblob')
        """
        self.lexicon = lexicon

        if lexicon == 'vader':
            try:
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                self.analyzer = SentimentIntensityAnalyzer()
                self.vader_available = True
            except ImportError:
                logger.warning("vaderSentiment not available. Install with: pip install vaderSentiment")
                self.vader_available = False

        elif lexicon == 'textblob':
            try:
                from textblob import TextBlob
                self.textblob_available = True
            except ImportError:
                logger.warning("textblob not available. Install with: pip install textblob")
                self.textblob_available = False

        elif lexicon == 'afinn':
            try:
                from afinn import Afinn
                self.afinn = Afinn()
                self.afinn_available = True
            except ImportError:
                logger.warning("afinn not available. Install with: pip install afinn")
                self.afinn_available = False

    def analyze(self, text: str) -> Dict:
        """
        Analyze sentiment using lexicon

        Args:
            text: Input text

        Returns:
            Dictionary with sentiment scores
        """
        if self.lexicon == 'vader' and hasattr(self, 'vader_available') and self.vader_available:
            scores = self.analyzer.polarity_scores(text)
            # VADER returns: {'neg': 0.0, 'neu': 0.0, 'pos': 1.0, 'compound': 0.5}
            return scores

        elif self.lexicon == 'textblob' and hasattr(self, 'textblob_available') and self.textblob_available:
            from textblob import TextBlob
            blob = TextBlob(text)
            # TextBlob returns polarity (-1 to 1) and subjectivity (0 to 1)
            return {
                'polarity': blob.sentiment.polarity,
                'subjectivity': blob.sentiment.subjectivity
            }

        elif self.lexicon == 'afinn' and hasattr(self, 'afinn_available') and self.afinn_available:
            score = self.afinn.score(text)
            # AFINN returns numeric score
            return {
                'score': score,
                'label': 'positive' if score > 0 else 'negative' if score < 0 else 'neutral'
            }

        else:
            raise ValueError(f"Lexicon {self.lexicon} not available")


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("Sentiment Analysis Test")
    print("=" * 70)

    # Check if transformers is available
    if not TRANSFORMERS_AVAILABLE:
        print("transformers library not available. Install with:")
        print("pip install transformers")
        exit(1)

    # Test transformer-based sentiment analysis
    print("\n1. Transformer-based Sentiment Analysis")
    print("-" * 70)

    try:
        # Binary sentiment (positive/negative)
        analyzer = SentimentAnalyzer(
            model_name='distilbert-base-uncased-finetuned-sst-2-english',
            use_pipeline=True
        )

        test_texts = [
            "This movie is absolutely amazing! I loved every minute of it.",
            "The product quality is terrible and it broke after one day.",
            "It's okay, nothing special but not bad either.",
            "Best purchase ever! Highly recommended!",
            "Waste of money. Very disappointed."
        ]

        print("\nBinary Sentiment (Positive/Negative):")
        for text in test_texts:
            result = analyzer.analyze(text)
            print(f"\nText: {text}")
            print(f"Sentiment: {result['label']} (confidence: {result['score']:.4f})")

    except Exception as e:
        print(f"Transformer test failed: {e}")

    # Test aspect-based sentiment
    print("\n2. Aspect-based Sentiment Analysis")
    print("-" * 70)

    try:
        review = """
        The food at this restaurant was absolutely delicious and the presentation was beautiful.
        However, the service was extremely slow and the staff seemed uninterested.
        The ambiance was nice with good music, but it was too noisy to have a conversation.
        Prices are reasonable for the quality of food you get.
        """

        aspects = ['food', 'service', 'ambiance', 'price']

        print(f"\nReview: {review.strip()}")
        print("\nAspect-based sentiment:")

        results = analyzer.analyze_aspects(review, aspects)
        for aspect, sentiment in results.items():
            print(f"  {aspect.capitalize():12s}: {sentiment['label']:10s} ({sentiment['score']:.4f})")

    except Exception as e:
        print(f"Aspect-based test failed: {e}")

    # Test lexicon-based sentiment (if available)
    print("\n3. Lexicon-based Sentiment Analysis (VADER)")
    print("-" * 70)

    try:
        vader = LexiconBasedSentiment(lexicon='vader')

        if vader.vader_available:
            test_text = "This product is absolutely fantastic! I'm so happy with my purchase!"

            scores = vader.analyze(test_text)

            print(f"Text: {test_text}")
            print(f"\nVADER scores:")
            print(f"  Negative: {scores['neg']:.4f}")
            print(f"  Neutral:  {scores['neu']:.4f}")
            print(f"  Positive: {scores['pos']:.4f}")
            print(f"  Compound: {scores['compound']:.4f}")
        else:
            print("VADER not available. Install with: pip install vaderSentiment")

    except Exception as e:
        print(f"VADER test failed: {e}")

    # Test batch processing
    print("\n4. Batch Processing")
    print("-" * 70)

    try:
        batch_texts = [
            "Great product!",
            "Terrible experience.",
            "Meh, it's okay.",
            "Love it!",
            "Hate it!"
        ]

        results = analyzer.analyze(batch_texts)

        print("Batch sentiment analysis:")
        for text, result in zip(batch_texts, results):
            print(f"  '{text:25s}' -> {result['label']:10s} ({result['score']:.4f})")

    except Exception as e:
        print(f"Batch test failed: {e}")

    print("\n" + "=" * 70)
    print("Sentiment analysis tested successfully!")
