import os
import json
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
import google.generativeai as genai

class ConversationMemory:
    """
    Advanced conversation memory system that provides short and long-term memory
    capabilities for an RAG chatbot.
    """
    
    def __init__(self, rag_name: str, base_dir: str = "rag_sessions"):
        """
        Initialize the conversation memory system.
        
        Args:
            rag_name: Name of the RAG session
            base_dir: Base directory for RAG sessions
        """
        self.rag_name = rag_name
        self.base_dir = base_dir
        self.rag_path = os.path.join(base_dir, rag_name)
        self.memory_path = os.path.join(self.rag_path, "memory")
        self.history_path = os.path.join(self.rag_path, "chat_history.json")
        self.summary_path = os.path.join(self.memory_path, "conversation_summary.json")
        
        # Create memory directory if it doesn't exist
        os.makedirs(self.memory_path, exist_ok=True)
        
        # Load conversation history
        self.history = self._load_chat_history()
        
        # Load or initialize conversation summaries
        self.summaries = self._load_summaries()
        
        # In-memory cache for recent history
        self.recent_messages = self.history[-10:] if self.history else []
        
        # Track when a summary was last generated
        self.last_summary_timestamp = self._get_last_summary_timestamp()
        self.summary_frequency = 10  # Generate summary every 10 messages
    
    def _load_chat_history(self) -> List[Dict[str, Any]]:
        """Load the conversation history from file"""
        if os.path.exists(self.history_path):
            try:
                with open(self.history_path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                return []
        return []
    
    def _load_summaries(self) -> Dict[str, Any]:
        """Load conversation summaries"""
        if os.path.exists(self.summary_path):
            try:
                with open(self.summary_path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                return self._initialize_summaries()
        return self._initialize_summaries()
    
    def _initialize_summaries(self) -> Dict[str, Any]:
        """Initialize empty summaries structure"""
        return {
            "overall_summary": "",
            "topics": [],
            "key_insights": [],
            "last_updated": "",
            "message_count": 0
        }
    
    def _get_last_summary_timestamp(self) -> float:
        """Get the timestamp of the last summary generation"""
        if "last_updated" in self.summaries and self.summaries["last_updated"]:
            try:
                # Convert ISO format to timestamp
                dt = datetime.fromisoformat(self.summaries["last_updated"].replace('Z', '+00:00'))
                return dt.timestamp()
            except (ValueError, TypeError):
                pass
        return 0.0
    
    def add_message(self, question: str, answer: str, sources: List[Dict[str, Any]] = None) -> None:
        """
        Add a new message pair to the conversation history.
        
        Args:
            question: User's question
            answer: System's answer
            sources: List of source metadata used for the answer
        """
        # Create message object
        timestamp = datetime.now().isoformat()
        message = {
            "question": question,
            "answer": answer,
            "sources": sources or [],
            "timestamp": timestamp
        }
        
        # Add to history
        self.history.append(message)
        self.recent_messages.append(message)
        
        # Keep recent messages limited
        if len(self.recent_messages) > 15:
            self.recent_messages.pop(0)
        
        # Save history
        self._save_chat_history()
        
        # Update summary if needed
        self._maybe_update_summary()
    
    def _save_chat_history(self) -> None:
        """Save the conversation history to file"""
        try:
            with open(self.history_path, "w") as f:
                json.dump(self.history, f)
        except Exception as e:
            print(f"Error saving chat history: {e}")
    
    def _save_summaries(self) -> None:
        """Save the conversation summaries to file"""
        try:
            with open(self.summary_path, "w") as f:
                json.dump(self.summaries, f)
        except Exception as e:
            print(f"Error saving summaries: {e}")
    
    def _maybe_update_summary(self) -> None:
        """Update the conversation summary if enough new messages have been added"""
        current_count = len(self.history)
        previous_count = self.summaries.get("message_count", 0)
        
        # Update summary if enough new messages or enough time has passed
        if (current_count - previous_count >= self.summary_frequency or
            (time.time() - self.last_summary_timestamp > 3600)):  # 1 hour
            self._generate_summaries()
    
    def _generate_summaries(self) -> None:
        """Generate summaries of the conversation using Gemini"""
        try:
            # Only proceed if we have enough messages
            if len(self.history) < 3:
                return
                
            model = genai.GenerativeModel('models/gemini-1.5-pro-latest')
            
            # Format recent conversation for the model
            formatted_history = "\n\n".join([
                f"User: {msg['question']}\nAssistant: {msg['answer']}"
                for msg in self.history[-30:]  # Use last 30 messages at most
            ])
            
            # Generate overall summary
            summary_prompt = f"""
You are analyzing a conversation between a user and an AI assistant.
Summarize the main points of this conversation to help with long-term context.

The conversation:
{formatted_history}

Provide a JSON response with these fields:
1. "overall_summary": A concise paragraph (3-5 sentences) summarizing the overall conversation
2. "topics": Array of 3-5 main topics discussed
3. "key_insights": Array of 3-5 important facts or insights from the conversation

Response format:
{{
  "overall_summary": "...",
  "topics": ["topic1", "topic2", ...],
  "key_insights": ["insight1", "insight2", ...]
}}

Only provide the JSON, no other text.
"""
            response = model.generate_content(summary_prompt)
            result = response.text.strip()
            
            # Parse the JSON response
            import json
            import re
            
            # Extract JSON
            json_pattern = r'\{.*\}'
            json_match = re.search(json_pattern, result, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                try:
                    summary_data = json.loads(json_str)
                    
                    # Update summaries
                    self.summaries["overall_summary"] = summary_data.get("overall_summary", self.summaries["overall_summary"])
                    
                    # Add new topics while keeping track of old ones
                    existing_topics = set(self.summaries["topics"])
                    new_topics = [t for t in summary_data.get("topics", []) if t not in existing_topics]
                    self.summaries["topics"] = self.summaries["topics"] + new_topics
                    
                    # Similarly for insights
                    existing_insights = set(self.summaries["key_insights"])
                    new_insights = [i for i in summary_data.get("key_insights", []) if i not in existing_insights]
                    self.summaries["key_insights"] = self.summaries["key_insights"] + new_insights
                    
                    # Update metadata
                    self.summaries["last_updated"] = datetime.now().isoformat()
                    self.summaries["message_count"] = len(self.history)
                    
                    # Save updated summaries
                    self._save_summaries()
                    
                    # Update timestamp
                    self.last_summary_timestamp = time.time()
                    
                except json.JSONDecodeError:
                    print("Error parsing summary JSON")
                    
        except Exception as e:
            print(f"Error generating summaries: {e}")
    
    def get_relevant_history(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve the most relevant previous exchanges for the current query.
        
        Args:
            query: Current user query
            k: Number of relevant exchanges to retrieve
            
        Returns:
            List of relevant conversation exchanges
        """
        if not self.history:
            return []
            
        try:
            model = genai.GenerativeModel('models/gemini-1.5-pro-latest')
            
            # Format history for relevance ranking
            formatted_history = "\n\n".join([
                f"EXCHANGE {i}:\nUser: {msg['question']}\nAssistant: {msg['answer']}"
                for i, msg in enumerate(self.history[-20:], 1)  # Use last 20 messages at most
            ])
            
            prompt = f"""
Given this new user query:
"{query}"

Identify which of the previous conversation exchanges are most relevant to this query.
Choose up to {min(k, len(self.history))} exchanges that contain information that would help answer this query.

Previous conversation:
{formatted_history}

Return ONLY a JSON array with the numbers of the relevant exchanges, sorted by relevance (most relevant first).
Example: [3, 7, 1]

If no exchanges are particularly relevant, return an empty array.
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
                    relevant_indices = json.loads(json_str)
                    if isinstance(relevant_indices, list) and relevant_indices:
                        # Adjust indices to match history (1-based to 0-based)
                        # And ensure they're within the range of the last 20 messages
                        start_idx = max(0, len(self.history) - 20)
                        relevant_exchanges = []
                        
                        for idx in relevant_indices:
                            # Convert from 1-based to 0-based and adjust
                            adjusted_idx = start_idx + idx - 1
                            if 0 <= adjusted_idx < len(self.history):
                                relevant_exchanges.append(self.history[adjusted_idx])
                                
                        return relevant_exchanges[:k]
                except json.JSONDecodeError:
                    pass
                    
        except Exception as e:
            print(f"Error finding relevant history: {e}")
            
        # Fallback to most recent exchanges
        return self.recent_messages[-k:]
    
    def get_contextual_summary(self, query: str) -> str:
        """
        Generate a contextual summary based on the query and conversation history.
        
        Args:
            query: Current user query
            
        Returns:
            A contextual summary for enhancing the current query
        """
        if not self.history or not self.summaries["overall_summary"]:
            return ""
            
        try:
            # Get relevant history
            relevant_history = self.get_relevant_history(query, k=3)
            
            # Extract summaries
            overall_summary = self.summaries["overall_summary"]
            topics = self.summaries["topics"][:5]  # Limit to top 5
            insights = self.summaries["key_insights"][:5]  # Limit to top 5
            
            model = genai.GenerativeModel('models/gemini-1.5-pro-latest')
            
            # Format relevant history
            formatted_relevant = "\n\n".join([
                f"User: {msg['question']}\nAssistant: {msg['answer']}"
                for msg in relevant_history
            ])
            
            prompt = f"""
Create a concise contextual memory for an AI assistant answering this new query:
"{query}"

Previous conversation summary:
{overall_summary}

Main topics discussed: {', '.join(topics)}

Key insights: {', '.join(insights)}

Most relevant previous exchanges:
{formatted_relevant}

Based on this history, create a brief contextual summary (2-3 sentences) focused ONLY on information directly relevant to this new query.
Include only information that would be helpful for answering the query.
If the conversation history doesn't contain relevant information, respond with an empty string.
"""
            response = model.generate_content(prompt)
            context = response.text.strip()
            
            # Remove quotation marks if present
            if context.startswith('"') and context.endswith('"'):
                context = context[1:-1]
                
            return context
            
        except Exception as e:
            print(f"Error generating contextual summary: {e}")
            return ""
    
    def create_thread(self, name: str, start_idx: int = None, end_idx: int = None) -> str:
        """
        Create a new conversation thread (fork) from the current conversation.
        
        Args:
            name: Name for the new thread
            start_idx: Starting message index (optional)
            end_idx: Ending message index (optional)
            
        Returns:
            New thread name if successful, empty string otherwise
        """
        if not self.history:
            return ""
            
        # Ensure name is valid as a directory
        import re
        safe_name = re.sub(r'[^\w\-\.]', '_', name)
        new_thread_name = f"{self.rag_name}_thread_{safe_name}"
        new_thread_path = os.path.join(self.base_dir, new_thread_name)
        
        # Check if thread already exists
        if os.path.exists(new_thread_path):
            return ""
            
        try:
            # Create thread directories
            os.makedirs(new_thread_path, exist_ok=True)
            os.makedirs(os.path.join(new_thread_path, "chroma_db"), exist_ok=True)
            os.makedirs(os.path.join(new_thread_path, "files"), exist_ok=True)
            os.makedirs(os.path.join(new_thread_path, "memory"), exist_ok=True)
            
            # Copy selected history or all history
            if start_idx is not None and end_idx is not None:
                thread_history = self.history[start_idx:end_idx+1]
            else:
                thread_history = self.history.copy()
                
            # Save thread history
            with open(os.path.join(new_thread_path, "chat_history.json"), "w") as f:
                json.dump(thread_history, f)
                
            # Generate initial summary for the thread
            thread_memory = ConversationMemory(new_thread_name, self.base_dir)
            thread_memory._generate_summaries()
            
            return new_thread_name
            
        except Exception as e:
            print(f"Error creating thread: {e}")
            return ""
    
    def search_history(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search the conversation history for messages related to the query.
        
        Args:
            query: Search query
            k: Maximum number of results to return
            
        Returns:
            List of matching messages
        """
        if not self.history:
            return []
            
        try:
            model = genai.GenerativeModel('models/gemini-1.5-pro-latest')
            
            # Format history for search
            formatted_history = "\n\n".join([
                f"EXCHANGE {i}:\nUser: {msg['question']}\nAssistant: {msg['answer']}"
                for i, msg in enumerate(self.history, 1)
            ])
            
            prompt = f"""
Search this conversation history for exchanges matching this search query:
"{query}"

Conversation history:
{formatted_history}

Return ONLY a JSON array with the numbers of the matching exchanges, sorted by relevance.
Example: [3, 7, 1]

If no exchanges match the query, return an empty array.
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
                    match_indices = json.loads(json_str)
                    if isinstance(match_indices, list) and match_indices:
                        # Convert from 1-based to 0-based
                        matches = []
                        for idx in match_indices:
                            adjusted_idx = idx - 1
                            if 0 <= adjusted_idx < len(self.history):
                                matches.append(self.history[adjusted_idx])
                        return matches[:k]
                except json.JSONDecodeError:
                    pass
                    
        except Exception as e:
            print(f"Error searching history: {e}")
            
        # Fallback to simple keyword matching
        matches = []
        query_lower = query.lower()
        for msg in self.history:
            if (query_lower in msg["question"].lower() or 
                query_lower in msg["answer"].lower()):
                matches.append(msg)
                if len(matches) >= k:
                    break
                    
        return matches 