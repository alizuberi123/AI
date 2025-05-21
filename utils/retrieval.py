import os
import numpy as np
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
import google.generativeai as genai
from operator import itemgetter

class HybridRetriever:
    """
    A hybrid retrieval system that combines vector search, BM25, and reranking
    for more accurate and comprehensive document retrieval.
    """
    
    def __init__(self, chroma_db, embedding_model=None):
        """
        Initialize the hybrid retriever with vector and sparse retrievers.
        
        Args:
            chroma_db: Initialized Chroma vector database
            embedding_model: Optional embedding model (will use default if None)
        """
        self.chroma_db = chroma_db
        self.embedding_model = embedding_model or HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Get all documents from Chroma to initialize BM25
        self.all_docs = self.chroma_db.get()
        if self.all_docs and 'documents' in self.all_docs and self.all_docs['documents']:
            self.bm25_docs = [Document(page_content=text, metadata=meta) 
                             for text, meta in zip(self.all_docs['documents'], 
                                                 self.all_docs['metadatas'])]
            self.bm25_retriever = BM25Retriever.from_documents(self.bm25_docs)
            self.bm25_retriever.k = 10  # Return top 10 results
        else:
            self.bm25_docs = []
            self.bm25_retriever = None
            
    def get_relevant_documents(self, query: str, k: int = 10, hybrid_alpha: float = 0.5, 
                              rerank: bool = True, diversity_bias: float = 0.3) -> List[Document]:
        """
        Retrieve documents using hybrid search.
        
        Args:
            query: User query
            k: Number of documents to retrieve
            hybrid_alpha: Weight for vector search (1-hybrid_alpha = weight for BM25)
            rerank: Whether to apply LLM reranking
            diversity_bias: How much to prioritize diverse sources (0-1)
            
        Returns:
            List of relevant documents
        """
        results = []
        
        # Vector search (semantic similarity)
        vector_docs = self.chroma_db.similarity_search(query, k=k)
        
        # BM25 search (keyword matching)
        if self.bm25_retriever:
            bm25_docs = self.bm25_retriever.get_relevant_documents(query)
        else:
            bm25_docs = []
            
        # Combine results with hybrid weighting
        doc_dict = {}
        
        # Score and add vector search results
        for i, doc in enumerate(vector_docs):
            doc_id = self._get_doc_id(doc)
            # Use reciprocal rank as score (highest rank = position 1 = score 1.0)
            score = hybrid_alpha * (1.0 / (i + 1))
            if doc_id in doc_dict:
                doc_dict[doc_id]["score"] += score
            else:
                doc_dict[doc_id] = {"doc": doc, "score": score, "sources": {doc.metadata.get("source", "unknown")}}
        
        # Score and add BM25 results
        for i, doc in enumerate(bm25_docs):
            doc_id = self._get_doc_id(doc)
            score = (1.0 - hybrid_alpha) * (1.0 / (i + 1))
            if doc_id in doc_dict:
                doc_dict[doc_id]["score"] += score
            else:
                doc_dict[doc_id] = {"doc": doc, "score": score, "sources": {doc.metadata.get("source", "unknown")}}
        
        # Apply diversity bias - boost scores of underrepresented sources
        if diversity_bias > 0:
            source_counts = {}
            for item in doc_dict.values():
                source = next(iter(item["sources"]))  # Get the first (only) source
                source_counts[source] = source_counts.get(source, 0) + 1
                
            # Apply diversity boost
            for doc_id, item in doc_dict.items():
                source = next(iter(item["sources"]))
                diversity_boost = diversity_bias * (1.0 / source_counts[source])
                item["score"] += diversity_boost
        
        # Sort by combined score
        sorted_results = sorted(doc_dict.values(), key=itemgetter("score"), reverse=True)
        combined_docs = [item["doc"] for item in sorted_results[:k]]
        
        # Apply reranking if enabled
        if rerank and combined_docs:
            return self.rerank_with_gemini(query, combined_docs)
        
        return combined_docs
        
    def _get_doc_id(self, doc: Document) -> str:
        """Generate a unique ID for a document based on content and metadata"""
        meta = doc.metadata
        content_hash = hash(doc.page_content[:100])  # Use start of content for hash
        source = meta.get("source", "unknown")
        page = meta.get("page", "0")
        chunk_id = meta.get("chunk_id", "0")
        return f"{source}_{page}_{chunk_id}_{content_hash}"
    
    def rerank_with_gemini(self, query: str, docs: List[Document]) -> List[Document]:
        """
        Use Gemini to rerank documents based on relevance to the query.
        
        Args:
            query: User query
            docs: List of documents to rerank
            
        Returns:
            Reranked list of documents
        """
        # Skip if no documents or only one document
        if not docs or len(docs) <= 1:
            return docs
            
        try:
            model = genai.GenerativeModel('models/gemini-1.5-pro-latest')
            formatted_chunks = "\n\n".join([
                f"[{i+1}] {c.page_content.strip()}"
                for i, c in enumerate(docs)
            ])
            
            prompt = f"""
You are evaluating document chunks for relevance to a user query.

User Query: {query}

Below are document chunks. Score each on a 0-10 scale for how relevant, comprehensive, and accurate it is for answering this query.
Prioritize chunks that directly address the query with concrete information over vague or tangential mentions.

Document Chunks:
{formatted_chunks}

Return a JSON object with chunk IDs and scores, like:
{{
    "ranked_ids": [3, 1, 5, 2, 4]
}}

Only include the JSON. No explanations or other text.
"""
            response = model.generate_content(prompt)
            result = response.text.strip()
            
            # Parse the JSON response
            import json
            import re
            
            # Clean the result - extract just the JSON part
            json_pattern = r'\{.*\}'
            json_match = re.search(json_pattern, result, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                try:
                    ranking = json.loads(json_str)
                    if "ranked_ids" in ranking:
                        # Convert 1-based to 0-based indexing
                        indices = [int(i) - 1 for i in ranking["ranked_ids"]]
                        # Filter valid indices
                        valid_indices = [i for i in indices if 0 <= i < len(docs)]
                        # Ensure we don't miss any documents
                        remaining = [i for i in range(len(docs)) if i not in valid_indices]
                        all_indices = valid_indices + remaining
                        # Return the reranked documents
                        return [docs[i] for i in all_indices]
                except json.JSONDecodeError:
                    pass
                    
        except Exception as e:
            print(f"Error in reranking: {e}")
            
        # Fallback to original order if reranking fails
        return docs
        
    def query_expansion(self, query: str, num_variants: int = 3) -> List[str]:
        """
        Generate query variations to improve recall.
        
        Args:
            query: Original user query
            num_variants: Number of alternative queries to generate
            
        Returns:
            List of expanded queries
        """
        try:
            model = genai.GenerativeModel('models/gemini-1.5-pro-latest')
            prompt = f"""
Given this search query: "{query}"

Generate {num_variants} alternative versions that:
1. Rephrase with different terminology
2. Expand with related concepts
3. Add technical specificity

Return ONLY a JSON array of strings with the alternative queries. Nothing else.
Example: ["first alternative", "second alternative", "third alternative"]
"""
            response = model.generate_content(prompt)
            result = response.text.strip()
            
            # Parse the JSON response
            import json
            import re
            
            # Try to extract a JSON array
            json_pattern = r'\[.*\]'
            json_match = re.search(json_pattern, result, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                try:
                    expanded_queries = json.loads(json_str)
                    if isinstance(expanded_queries, list) and expanded_queries:
                        return [query] + expanded_queries  # Original + expanded
                except json.JSONDecodeError:
                    pass
                    
        except Exception as e:
            print(f"Error in query expansion: {e}")
            
        # Fallback to just the original query
        return [query]
        
    def retrieval_with_expansion(self, query: str, k: int = 10) -> List[Document]:
        """
        Retrieve documents using query expansion for improved recall.
        
        Args:
            query: Original user query
            k: Number of documents to retrieve
            
        Returns:
            List of relevant documents
        """
        expanded_queries = self.query_expansion(query)
        
        all_docs = []
        doc_dict = {}  # To track unique documents
        
        # Retrieve documents for each query variant
        for q in expanded_queries:
            docs = self.get_relevant_documents(q, k=k//2)  # Get fewer docs per query
            
            # Track unique documents
            for doc in docs:
                doc_id = self._get_doc_id(doc)
                if doc_id not in doc_dict:
                    doc_dict[doc_id] = doc
                    all_docs.append(doc)
                    
            # Stop if we have enough documents
            if len(all_docs) >= k:
                break
                
        # Final reranking of combined results
        if all_docs:
            return self.rerank_with_gemini(query, all_docs[:k])
        
        return all_docs[:k] 