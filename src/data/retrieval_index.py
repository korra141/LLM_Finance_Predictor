"""
Retrieval index management for RAG (Retrieval-Augmented Generation).
Handles building and managing embeddings and vector indices for financial documents.
"""

import os
import pickle
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime
import json

# Vector store imports
import faiss
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)


class FinancialRetrievalIndex:
    """Manage retrieval index for financial documents."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.embedding_model = None
        self.vector_store = None
        self.index = None
        self.documents = []
        self.metadata = []
        
        # Initialize embedding model
        self._initialize_embedding_model()
        
        # Initialize vector store
        self._initialize_vector_store()
    
    def _initialize_embedding_model(self):
        """Initialize the embedding model."""
        model_name = self.config.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')
        self.embedding_model = SentenceTransformer(model_name)
        logger.info(f"Initialized embedding model: {model_name}")
    
    def _initialize_vector_store(self):
        """Initialize the vector store based on configuration."""
        vector_store_type = self.config.get('vector_store', 'faiss')
        
        if vector_store_type == 'faiss':
            self._initialize_faiss()
        elif vector_store_type == 'chroma':
            self._initialize_chroma()
        else:
            raise ValueError(f"Unsupported vector store type: {vector_store_type}")
    
    def _initialize_faiss(self):
        """Initialize FAISS vector store."""
        embedding_dim = self.config.get('embedding_dim', 384)
        index_type = self.config.get('index_type', 'IndexFlatIP')
        
        if index_type == 'IndexFlatIP':
            self.index = faiss.IndexFlatIP(embedding_dim)
        elif index_type == 'IndexFlatL2':
            self.index = faiss.IndexFlatL2(embedding_dim)
        elif index_type == 'IndexIVFFlat':
            quantizer = faiss.IndexFlatL2(embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, embedding_dim, 100)
        else:
            raise ValueError(f"Unsupported FAISS index type: {index_type}")
        
        logger.info(f"Initialized FAISS index: {index_type}")
    
    def _initialize_chroma(self):
        """Initialize ChromaDB vector store."""
        persist_directory = self.config.get('persist_directory', './data/retrieval_corpus/chroma_db')
        
        self.vector_store = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Create or get collection
        collection_name = self.config.get('collection_name', 'financial_documents')
        try:
            self.collection = self.vector_store.get_collection(collection_name)
        except:
            self.collection = self.vector_store.create_collection(
                name=collection_name,
                metadata={"description": "Financial documents for RAG"}
            )
        
        logger.info(f"Initialized ChromaDB collection: {collection_name}")
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """
        Add documents to the retrieval index.
        
        Args:
            documents: List of document dictionaries with 'text', 'metadata' keys
        """
        if not documents:
            logger.warning("No documents provided")
            return
        
        # Extract texts and metadata
        texts = [doc['text'] for doc in documents]
        metadata_list = [doc.get('metadata', {}) for doc in documents]
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        # Add to vector store
        vector_store_type = self.config.get('vector_store', 'faiss')
        
        if vector_store_type == 'faiss':
            self._add_to_faiss(embeddings, texts, metadata_list)
        elif vector_store_type == 'chroma':
            self._add_to_chroma(embeddings, texts, metadata_list)
        
        # Store documents and metadata
        self.documents.extend(texts)
        self.metadata.extend(metadata_list)
        
        logger.info(f"Added {len(documents)} documents to retrieval index")
    
    def _add_to_faiss(self, embeddings: np.ndarray, texts: List[str], metadata_list: List[Dict]):
        """Add embeddings to FAISS index."""
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to index
        self.index.add(embeddings.astype('float32'))
        
        # Store additional data
        if not hasattr(self, 'faiss_texts'):
            self.faiss_texts = []
            self.faiss_metadata = []
        
        self.faiss_texts.extend(texts)
        self.faiss_metadata.extend(metadata_list)
    
    def _add_to_chroma(self, embeddings: np.ndarray, texts: List[str], metadata_list: List[Dict]):
        """Add embeddings to ChromaDB."""
        # Prepare data for ChromaDB
        ids = [f"doc_{i}" for i in range(len(texts))]
        
        # Add to collection
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadata_list,
            ids=ids
        )
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for relevant documents.
        
        Args:
            query: Search query
            top_k: Number of results to return
        
        Returns:
            List of relevant documents with scores
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        
        vector_store_type = self.config.get('vector_store', 'faiss')
        
        if vector_store_type == 'faiss':
            return self._search_faiss(query_embedding, top_k)
        elif vector_store_type == 'chroma':
            return self._search_chroma(query, top_k)
        else:
            raise ValueError(f"Unsupported vector store type: {vector_store_type}")
    
    def _search_faiss(self, query_embedding: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        """Search FAISS index."""
        # Normalize query embedding
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        # Prepare results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.faiss_texts):
                results.append({
                    'text': self.faiss_texts[idx],
                    'metadata': self.faiss_metadata[idx],
                    'score': float(score)
                })
        
        return results
    
    def _search_chroma(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Search ChromaDB."""
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        # Prepare results
        formatted_results = []
        for i in range(len(results['documents'][0])):
            formatted_results.append({
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'score': results['distances'][0][i] if 'distances' in results else 0.0
            })
        
        return formatted_results
    
    def build_from_news_data(self, news_df: pd.DataFrame):
        """
        Build retrieval index from news data.
        
        Args:
            news_df: DataFrame with news data
        """
        documents = []
        
        for _, row in news_df.iterrows():
            # Combine title and content
            text = f"{row.get('title', '')} {row.get('content', '')}"
            
            # Create document
            doc = {
                'text': text,
                'metadata': {
                    'symbol': row.get('symbol'),
                    'date': str(row.get('date')),
                    'source': row.get('source'),
                    'sentiment': row.get('sentiment'),
                    'type': 'news'
                }
            }
            documents.append(doc)
        
        self.add_documents(documents)
        logger.info(f"Built retrieval index from {len(documents)} news articles")
    
    def build_from_fundamental_data(self, fundamental_df: pd.DataFrame):
        """
        Build retrieval index from fundamental data.
        
        Args:
            fundamental_df: DataFrame with fundamental data
        """
        documents = []
        
        for _, row in fundamental_df.iterrows():
            # Create text from fundamental metrics
            text = f"Symbol: {row.get('symbol', '')} "
            text += f"Sector: {row.get('sector', '')} "
            text += f"Industry: {row.get('industry', '')} "
            text += f"Market Cap: {row.get('market_cap', '')} "
            text += f"P/E Ratio: {row.get('pe_ratio', '')} "
            text += f"Revenue Growth: {row.get('revenue_growth', '')}"
            
            # Create document
            doc = {
                'text': text,
                'metadata': {
                    'symbol': row.get('symbol'),
                    'sector': row.get('sector'),
                    'industry': row.get('industry'),
                    'type': 'fundamental'
                }
            }
            documents.append(doc)
        
        self.add_documents(documents)
        logger.info(f"Built retrieval index from {len(documents)} fundamental records")
    
    def save_index(self, filepath: str):
        """
        Save the retrieval index to disk.
        
        Args:
            filepath: Path to save the index
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        if self.config.get('vector_store') == 'faiss':
            # Save FAISS index
            faiss.write_index(self.index, f"{filepath}.faiss")
            
            # Save additional data
            with open(f"{filepath}_data.pkl", 'wb') as f:
                pickle.dump({
                    'texts': self.faiss_texts,
                    'metadata': self.faiss_metadata,
                    'config': self.config
                }, f)
        
        elif self.config.get('vector_store') == 'chroma':
            # ChromaDB is already persistent
            with open(f"{filepath}_config.json", 'w') as f:
                json.dump(self.config, f, indent=2)
        
        logger.info(f"Saved retrieval index to {filepath}")
    
    def load_index(self, filepath: str):
        """
        Load the retrieval index from disk.
        
        Args:
            filepath: Path to load the index from
        """
        if self.config.get('vector_store') == 'faiss':
            # Load FAISS index
            self.index = faiss.read_index(f"{filepath}.faiss")
            
            # Load additional data
            with open(f"{filepath}_data.pkl", 'rb') as f:
                data = pickle.load(f)
                self.faiss_texts = data['texts']
                self.faiss_metadata = data['metadata']
                self.config.update(data['config'])
        
        elif self.config.get('vector_store') == 'chroma':
            # ChromaDB is already persistent, just load config
            with open(f"{filepath}_config.json", 'r') as f:
                config = json.load(f)
                self.config.update(config)
        
        logger.info(f"Loaded retrieval index from {filepath}")
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the retrieval index."""
        stats = {
            'total_documents': len(self.documents),
            'vector_store_type': self.config.get('vector_store'),
            'embedding_model': self.config.get('embedding_model'),
            'embedding_dim': self.config.get('embedding_dim')
        }
        
        if self.config.get('vector_store') == 'faiss':
            stats['index_size'] = self.index.ntotal
        elif self.config.get('vector_store') == 'chroma':
            stats['index_size'] = self.collection.count()
        
        return stats


if __name__ == "__main__":
    # Example usage
    config = {
        'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
        'vector_store': 'faiss',
        'embedding_dim': 384,
        'index_type': 'IndexFlatIP'
    }
    
    # Initialize retrieval index
    retrieval_index = FinancialRetrievalIndex(config)
    
    # Create sample documents
    sample_docs = [
        {
            'text': 'Apple Inc. reported strong quarterly earnings with revenue growth of 15%.',
            'metadata': {'symbol': 'AAPL', 'type': 'news', 'date': '2023-01-01'}
        },
        {
            'text': 'Microsoft Corporation announced new AI initiatives in cloud computing.',
            'metadata': {'symbol': 'MSFT', 'type': 'news', 'date': '2023-01-02'}
        }
    ]
    
    # Add documents
    retrieval_index.add_documents(sample_docs)
    
    # Search
    results = retrieval_index.search('Apple earnings', top_k=2)
    print("Search results:", results)
    
    # Get stats
    stats = retrieval_index.get_index_stats()
    print("Index stats:", stats)