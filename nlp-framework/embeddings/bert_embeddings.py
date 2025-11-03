"""
BERT Embeddings
Contextualized word embeddings using BERT and other transformers
"""

import torch
import numpy as np
from typing import List, Optional, Union, Tuple, Dict
import logging

try:
    from transformers import (
        AutoModel, AutoTokenizer,
        BertModel, BertTokenizer,
        RobertaModel, RobertaTokenizer,
        DistilBertModel, DistilBertTokenizer
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("transformers not available. Install with: pip install transformers")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BERTEmbeddings:
    """
    BERT-based Contextualized Embeddings

    Uses transformer models (BERT, RoBERTa, DistilBERT, etc.) to generate
    contextualized word embeddings where the same word can have different
    embeddings based on context.

    Key differences from Word2Vec:
    - Contextualized: Same word has different embeddings in different contexts
    - Bidirectional: Considers both left and right context
    - Pre-trained: Can use models trained on massive corpora
    - Subword tokenization: Better handling of rare/OOV words

    Supported models:
    - BERT: bert-base-uncased, bert-large-uncased
    - RoBERTa: roberta-base, roberta-large
    - DistilBERT: distilbert-base-uncased (faster, smaller)
    - Any HuggingFace transformer model

    Best for:
    - Transfer learning for NLP tasks
    - Semantic search
    - Text classification
    - Question answering
    - Named entity recognition
    - Sentence similarity
    """

    def __init__(
        self,
        model_name: str = 'bert-base-uncased',
        device: Optional[str] = None,
        max_length: int = 512,
        pooling: str = 'mean'
    ):
        """
        Initialize BERT embeddings

        Args:
            model_name: HuggingFace model name
            device: Device ('cuda', 'cpu', or None for auto)
            max_length: Maximum sequence length
            pooling: Pooling strategy ('mean', 'cls', 'max', 'mean_sqrt')
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers not installed. Install with: pip install transformers")

        self.model_name = model_name
        self.max_length = max_length
        self.pooling = pooling

        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        logger.info(f"Loading {model_name}...")
        logger.info(f"Device: {self.device}")

        # Load model and tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

        self.model.to(self.device)
        self.model.eval()

        # Get model dimensions
        self.embedding_dim = self.model.config.hidden_size

        logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = False,
        return_tokens: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, List[List[str]]]]:
        """
        Encode texts to embeddings

        Args:
            texts: Single text or list of texts
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            return_tokens: Also return tokenized texts

        Returns:
            Embeddings array (shape: [n_texts, embedding_dim])
            If return_tokens=True: (embeddings, tokens)
        """
        # Convert single text to list
        single_text = isinstance(texts, str)
        if single_text:
            texts = [texts]

        embeddings = []
        all_tokens = [] if return_tokens else None

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # Tokenize
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )

            # Move to device
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**encoded)

            # Apply pooling
            batch_embeddings = self._pool_embeddings(
                outputs.last_hidden_state,
                encoded['attention_mask']
            )

            embeddings.append(batch_embeddings.cpu().numpy())

            # Get tokens if requested
            if return_tokens:
                for j in range(len(batch_texts)):
                    tokens = self.tokenizer.convert_ids_to_tokens(
                        encoded['input_ids'][j]
                    )
                    all_tokens.append(tokens)

            if show_progress:
                logger.info(f"Processed {min(i + batch_size, len(texts))}/{len(texts)} texts")

        # Concatenate batches
        embeddings = np.concatenate(embeddings, axis=0)

        # Return single embedding if input was single text
        if single_text:
            embeddings = embeddings[0]
            if return_tokens:
                all_tokens = all_tokens[0]

        if return_tokens:
            return embeddings, all_tokens

        return embeddings

    def _pool_embeddings(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Pool token embeddings to get sentence embedding

        Args:
            hidden_states: Token embeddings [batch, seq_len, hidden_dim]
            attention_mask: Attention mask [batch, seq_len]

        Returns:
            Pooled embeddings [batch, hidden_dim]
        """
        if self.pooling == 'cls':
            # Use [CLS] token embedding
            return hidden_states[:, 0, :]

        elif self.pooling == 'mean':
            # Mean pooling over all tokens
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            return sum_embeddings / sum_mask

        elif self.pooling == 'max':
            # Max pooling over all tokens
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            hidden_states[mask_expanded == 0] = -1e9  # Set padding to large negative
            return torch.max(hidden_states, dim=1)[0]

        elif self.pooling == 'mean_sqrt':
            # Mean pooling with sqrt normalization (used by some sentence transformers)
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            return sum_embeddings / torch.sqrt(sum_mask)

        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")

    def get_token_embeddings(
        self,
        text: str,
        layer: int = -1
    ) -> Tuple[List[str], np.ndarray]:
        """
        Get embeddings for each token

        Args:
            text: Input text
            layer: Which layer to extract embeddings from (-1 = last layer)

        Returns:
            (tokens, embeddings) where embeddings shape is [n_tokens, embedding_dim]
        """
        # Tokenize
        encoded = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=self.max_length
        )

        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**encoded, output_hidden_states=True)

        # Get tokens
        tokens = self.tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])

        # Get embeddings from specified layer
        if layer == -1:
            embeddings = outputs.last_hidden_state[0]
        else:
            embeddings = outputs.hidden_states[layer][0]

        return tokens, embeddings.cpu().numpy()

    def similarity(self, text1: str, text2: str) -> float:
        """
        Calculate cosine similarity between two texts

        Args:
            text1: First text
            text2: Second text

        Returns:
            Cosine similarity (0-1)
        """
        embeddings = self.encode([text1, text2])

        # Cosine similarity
        emb1 = embeddings[0]
        emb2 = embeddings[1]

        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        similarity = dot_product / (norm1 * norm2)

        return float(similarity)

    def most_similar(
        self,
        query: str,
        candidates: List[str],
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Find most similar texts to query

        Args:
            query: Query text
            candidates: List of candidate texts
            top_k: Number of results to return

        Returns:
            List of (text, similarity_score) tuples
        """
        # Encode query
        query_emb = self.encode(query)

        # Encode candidates
        candidate_embs = self.encode(candidates, batch_size=32)

        # Calculate similarities
        similarities = []
        for i, cand_emb in enumerate(candidate_embs):
            dot_product = np.dot(query_emb, cand_emb)
            norm_q = np.linalg.norm(query_emb)
            norm_c = np.linalg.norm(cand_emb)
            sim = dot_product / (norm_q * norm_c)
            similarities.append((candidates[i], float(sim)))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]

    def get_embedding_dim(self) -> int:
        """Get embedding dimensionality"""
        return self.embedding_dim

    def get_model_info(self) -> Dict[str, any]:
        """Get model information"""
        return {
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'max_length': self.max_length,
            'pooling': self.pooling,
            'device': str(self.device),
            'num_parameters': sum(p.numel() for p in self.model.parameters())
        }


class SentenceTransformer:
    """
    Sentence Transformer wrapper
    Optimized models for sentence embeddings
    """

    def __init__(
        self,
        model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
        device: Optional[str] = None
    ):
        """
        Initialize Sentence Transformer

        Args:
            model_name: SentenceTransformer model name
            device: Device ('cuda', 'cpu', or None for auto)

        Popular models:
        - all-MiniLM-L6-v2: Fast and efficient (384 dim)
        - all-mpnet-base-v2: High quality (768 dim)
        - multi-qa-mpnet-base-dot-v1: Question answering
        """
        try:
            from sentence_transformers import SentenceTransformer as ST
            self.st_available = True
        except ImportError:
            logger.warning("sentence-transformers not available")
            logger.warning("Falling back to BERT embeddings")
            self.st_available = False

        if self.st_available:
            logger.info(f"Loading SentenceTransformer: {model_name}")
            self.model = ST(model_name, device=device)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
        else:
            # Fallback to BERT
            self.model = BERTEmbeddings(model_name=model_name, device=device)
            self.embedding_dim = self.model.embedding_dim

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = False
    ) -> np.ndarray:
        """Encode texts to embeddings"""
        if self.st_available:
            return self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True
            )
        else:
            return self.model.encode(texts, batch_size=batch_size, show_progress=show_progress)

    def similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between texts"""
        if self.st_available:
            emb1 = self.model.encode([text1])[0]
            emb2 = self.model.encode([text2])[0]
            return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
        else:
            return self.model.similarity(text1, text2)


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("BERT Embeddings Test")
    print("=" * 70)

    # Check if transformers is available
    if not TRANSFORMERS_AVAILABLE:
        print("transformers library not available. Install with:")
        print("pip install transformers")
        exit(1)

    # Test BERT embeddings
    print("\n1. BERT Embeddings (DistilBERT)")
    print("-" * 70)

    bert = BERTEmbeddings(
        model_name='distilbert-base-uncased',
        pooling='mean'
    )

    print(f"Model: {bert.model_name}")
    print(f"Embedding dimension: {bert.embedding_dim}")
    print(f"Device: {bert.device}")

    # Encode single text
    text = "Machine learning is transforming the world."
    embedding = bert.encode(text)
    print(f"\nText: {text}")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding (first 10 dims): {embedding[:10]}")

    # Encode batch
    print("\n2. Batch Encoding")
    print("-" * 70)

    texts = [
        "Natural language processing is fascinating.",
        "Deep learning models are powerful.",
        "Transformers revolutionized NLP.",
        "BERT provides contextualized embeddings."
    ]

    embeddings = bert.encode(texts, batch_size=2)
    print(f"Number of texts: {len(texts)}")
    print(f"Embeddings shape: {embeddings.shape}")

    # Test similarity
    print("\n3. Semantic Similarity")
    print("-" * 70)

    pairs = [
        ("I love machine learning", "Machine learning is great"),
        ("I love machine learning", "The weather is nice today"),
        ("Dog is an animal", "Cat is an animal"),
        ("Dog is an animal", "Python is a programming language")
    ]

    for text1, text2 in pairs:
        sim = bert.similarity(text1, text2)
        print(f"\nText 1: {text1}")
        print(f"Text 2: {text2}")
        print(f"Similarity: {sim:.4f}")

    # Test most similar
    print("\n4. Find Most Similar")
    print("-" * 70)

    query = "machine learning algorithms"
    candidates = [
        "deep learning neural networks",
        "natural language processing",
        "computer vision models",
        "random forest classifier",
        "cooking recipes",
        "sports news"
    ]

    results = bert.most_similar(query, candidates, top_k=3)
    print(f"Query: {query}")
    print("\nMost similar:")
    for text, score in results:
        print(f"  {text}: {score:.4f}")

    # Test token embeddings
    print("\n5. Token-level Embeddings")
    print("-" * 70)

    text = "BERT creates contextualized embeddings"
    tokens, token_embs = bert.get_token_embeddings(text)

    print(f"Text: {text}")
    print(f"Tokens: {tokens}")
    print(f"Token embeddings shape: {token_embs.shape}")
    print(f"First token '{tokens[0]}' embedding (first 5 dims): {token_embs[0][:5]}")

    # Test different pooling strategies
    print("\n6. Pooling Strategies")
    print("-" * 70)

    test_text = "Testing different pooling methods"
    for pooling in ['cls', 'mean', 'max']:
        bert_pool = BERTEmbeddings(
            model_name='distilbert-base-uncased',
            pooling=pooling
        )
        emb = bert_pool.encode(test_text)
        print(f"{pooling.upper():4s} pooling - first 5 dims: {emb[:5]}")

    print("\n" + "=" * 70)
    print("BERT embeddings tested successfully!")
