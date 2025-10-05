# nlp_processor.py
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

class NLPQueryProcessor:
    """Advanced NLP processor for converting natural language to formal DSL commands."""
    
    def __init__(self):
        try:
            self.stemmer = PorterStemmer()
            self.stop_words = set(stopwords.words('english'))
        except Exception as e:
            print(f"Warning: NLTK data not fully available: {e}")
            # Fallback to basic functionality
            self.stemmer = None
            self.stop_words = set()
        
        # Enhanced training data with more variations
        self.training_data = [
            # System Commands - CPU
            ("show me the cpu usage", "SHOW CPU"),
            ("what is the processor usage", "SHOW CPU"),
            ("how much cpu is being used", "SHOW CPU"),
            ("display cpu utilization", "SHOW CPU"),
            ("check processor load", "SHOW CPU"),
            ("cpu percentage", "SHOW CPU"),
            ("processor performance", "SHOW CPU"),
            ("system cpu status", "SHOW CPU"),
            
            # System Commands - Memory
            ("show memory usage", "SHOW MEMORY"),
            ("how much ram is used", "SHOW MEMORY"),
            ("display memory status", "SHOW MEMORY"),
            ("check ram usage", "SHOW MEMORY"),
            ("memory consumption", "SHOW MEMORY"),
            ("available memory", "SHOW MEMORY"),
            ("system memory", "SHOW MEMORY"),
            ("ram status", "SHOW MEMORY"),
            
            # System Commands - Tasks/Processes
            ("show running processes", "SHOW TASKS"),
            ("list all tasks", "SHOW TASKS"),
            ("what processes are running", "SHOW TASKS"),
            ("display active applications", "SHOW TASKS"),
            ("running programs", "SHOW TASKS"),
            ("active processes", "SHOW TASKS"),
            ("system tasks", "SHOW TASKS"),
            ("process list", "SHOW TASKS"),
            
            # Database Commands - Table Operations
            ("show all tables", "DB SELECT name FROM sqlite_master WHERE type='table';"),
            ("list tables", "DB SELECT name FROM sqlite_master WHERE type='table';"),
            ("what tables exist", "DB SELECT name FROM sqlite_master WHERE type='table';"),
            ("display database tables", "DB SELECT name FROM sqlite_master WHERE type='table';"),
            
            # Database Commands - Generic SQL
            ("select all from users", "DB SELECT * FROM users;"),
            ("get all records from logs", "DB SELECT * FROM logs;"),
            ("show me everything in the database", "DB SELECT name FROM sqlite_master WHERE type='table';"),
            ("create a table called users", "DB CREATE TABLE users (id INTEGER, name TEXT);"),
            ("insert data into logs", "DB INSERT INTO logs VALUES (1, 'Sample log');"),
        ]
        
        # Intent patterns for better classification
        self.intent_patterns = {
            'system_cpu': [
                r'\b(cpu|processor|processing)\b.*\b(usage|utilization|load|performance|status)\b',
                r'\b(show|display|check|get)\b.*\b(cpu|processor)\b',
                r'\b(how much|what is|what\'s)\b.*\b(cpu|processor)\b'
            ],
            'system_memory': [
                r'\b(memory|ram|mem)\b.*\b(usage|status|consumption|available)\b',
                r'\b(show|display|check|get)\b.*\b(memory|ram)\b',
                r'\b(how much|what is|what\'s)\b.*\b(memory|ram)\b'
            ],
            'system_tasks': [
                r'\b(process|task|program|application|app)\b.*\b(running|active|list)\b',
                r'\b(show|display|list|get)\b.*\b(process|task|program)\b',
                r'\b(what|which)\b.*\b(process|task|program)\b.*\b(running|active)\b'
            ],
            'database_tables': [
                r'\b(show|list|display|get)\b.*\btable\b',
                r'\b(what|which)\b.*\btable\b.*\b(exist|available)\b',
                r'\btable\b.*\b(list|show|display)\b'
            ],
            'database_select': [
                r'\b(select|get|show|display)\b.*\bfrom\b',
                r'\b(all|everything)\b.*\bfrom\b',
                r'\bselect\b.*\*'
            ],
            'database_create': [
                r'\bcreate\b.*\btable\b',
                r'\bmake\b.*\btable\b',
                r'\bnew\b.*\btable\b'
            ],
            'database_insert': [
                r'\binsert\b.*\binto\b',
                r'\badd\b.*\bto\b.*\btable\b',
                r'\bput\b.*\bin\b.*\btable\b'
            ]
        }
        
        # Initialize TF-IDF vectorizer
        try:
            self.vectorizer = TfidfVectorizer(
                lowercase=True,
                stop_words='english',
                ngram_range=(1, 3),
                max_features=1000
            )
            
            # Train the model
            self._train_model()
            self.initialized = True
        except Exception as e:
            print(f"Warning: TF-IDF initialization failed: {e}")
            self.vectorizer = None
            self.initialized = False
    
    def _train_model(self):
        """Train the NLP model with the training data."""
        try:
            self.queries = [query for query, _ in self.training_data]
            self.commands = [command for _, command in self.training_data]
            
            # Fit TF-IDF vectorizer
            self.query_vectors = self.vectorizer.fit_transform(self.queries)
        except Exception as e:
            print(f"Warning: Model training failed: {e}")
            self.queries = []
            self.commands = []
            self.query_vectors = None
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for better matching."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters except important ones
        text = re.sub(r'[^\w\s\*\(\)=<>!]', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract entities like table names, column names, etc."""
        entities = {
            'tables': [],
            'columns': [],
            'values': [],
            'numbers': []
        }
        
        # Extract table names (common patterns)
        # Extract table names (common patterns)
        table_patterns = [
            r'\bfrom\s+(\w+)',
            r'\binto\s+(\w+)',
            r'\btable\s+(\w+)',
            r'\bcalled\s+(\w+)',          # "called logs" → captures "logs"
            r'\bnamed\s+(\w+)',           # "named users" → captures "users"
            r'\bcreate\s+table\s+(\w+)',   # "create table logs" → captures "logs"
            r'\bmake\s+table\s+(\w+)'      # "make table users" → captures "users"
        ]
        
        for pattern in table_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities['tables'].extend(matches)
        
        # Extract column names
        column_patterns = [
            r'\bselect\s+([\w\s,\*]+)\s+from',
            r'\(([^)]+)\)'  # Content within parentheses
        ]
                # Extract columns from "with X and Y" patterns
        with_pattern = r'\bwith\s+([^,;]+?)(?:\s+and\s+([^,;]+?))?(?:\s+and\s+([^,;]+?))?\b'
        with_matches = re.findall(with_pattern, text, re.IGNORECASE)
        for match_tuple in with_matches:
            for col in match_tuple:
                if col.strip():
                    entities['columns'].append(col.strip())
                    
        for pattern in column_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if match != '*':
                    cols = [col.strip() for col in match.split(',')]
                    entities['columns'].extend(cols)
        
        # Extract numbers
        entities['numbers'] = re.findall(r'\b\d+(?:\.\d+)?\b', text)
        
        # Extract quoted strings as values
        entities['values'] = re.findall(r"'([^']*)'|\"([^\"]*)\"", text)
        entities['values'] = [v[0] or v[1] for v in entities['values']]
        
        return entities
    
    def _classify_intent(self, text: str) -> Optional[str]:
        """Classify the intent of the natural language query."""
        preprocessed = self._preprocess_text(text)
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, preprocessed, re.IGNORECASE):
                    return intent
        
        return None
    
    def _similarity_match(self, query: str) -> Tuple[str, float]:
        """Find the most similar training query using TF-IDF and cosine similarity."""
        try:
            if not self.initialized or self.vectorizer is None or self.query_vectors is None:
                return "", 0.0
                
            query_vector = self.vectorizer.transform([query])
            similarities = cosine_similarity(query_vector, self.query_vectors)[0]
            
            best_match_idx = np.argmax(similarities)
            best_similarity = similarities[best_match_idx]
            best_command = self.commands[best_match_idx]
            
            return best_command, best_similarity
        except Exception as e:
            print(f"Warning: Similarity matching failed: {e}")
            return "", 0.0
    
    def _generate_sql_from_intent(self, intent: str, entities: Dict[str, List[str]], original_query: str) -> str:
        """Generate SQL commands based on intent and extracted entities."""
        
        if intent == 'database_select':
            if entities['tables']:
                table = entities['tables'][0]
                if entities['columns'] and entities['columns'][0] != '*':
                    columns = ', '.join(entities['columns'])
                    return f"DB SELECT {columns} FROM {table};"
                else:
                    return f"DB SELECT * FROM {table};"
            else:
                # Try to extract table from query
                words = original_query.lower().split()
                if 'from' in words:
                    try:
                        table_idx = words.index('from') + 1
                        if table_idx < len(words):
                            table = words[table_idx]
                            return f"DB SELECT * FROM {table};"
                    except:
                        pass
        
        elif intent == 'database_create':
            if entities['tables']:
                table = entities['tables'][0]
                # Clean table name (remove any non-alphanumeric chars)
                table = re.sub(r'[^\w]', '', table)
                
                if entities['columns']:
                    # Clean column names and create proper definitions
                    clean_cols = []
                    for col in entities['columns']:
                        col_clean = re.sub(r'[^\w]', '', col)
                        if col_clean and col_clean.lower() not in ['and', 'or', 'the', 'a']:
                            clean_cols.append(f"{col_clean} TEXT")
                    
                    if clean_cols:
                        cols_str = ', '.join(clean_cols)
                        return f"DB CREATE TABLE {table} ({cols_str});"
                    else:
                        return f"DB CREATE TABLE {table} (id INTEGER PRIMARY KEY, data TEXT);"
                else:
                    # Default schema when no columns specified
                    return f"DB CREATE TABLE {table} (id INTEGER PRIMARY KEY, name TEXT);"
        
        elif intent == 'database_insert':
            if entities['tables']:
                table = entities['tables'][0]
                if entities['values']:
                    values = ', '.join([f"'{val}'" for val in entities['values']])
                    return f"DB INSERT INTO {table} VALUES ({values});"
                else:
                    return f"DB INSERT INTO {table} VALUES (1, 'sample');"
        
        return ""
    
    def process_query(self, natural_query: str) -> Dict[str, any]:
        """
        Process a natural language query and convert it to formal DSL.
        
        Returns:
            Dict with 'command', 'confidence', 'method', and 'entities'
        """
        if not natural_query.strip():
            return {
                'command': '',
                'confidence': 0.0,
                'method': 'empty',
                'entities': {},
                'error': 'Empty query'
            }
        
        # Preprocess the query
        preprocessed = self._preprocess_text(natural_query)
        
        # Extract entities
        entities = self._extract_entities(preprocessed)
        
        # Method 1: Intent-based classification
        intent = self._classify_intent(preprocessed)
        if intent:
            if intent.startswith('system_'):
                command_map = {
                    'system_cpu': 'SHOW CPU',
                    'system_memory': 'SHOW MEMORY',
                    'system_tasks': 'SHOW TASKS'
                }
                command = command_map.get(intent, '')
                if command:
                    return {
                        'command': command,
                        'confidence': 0.9,
                        'method': 'intent_classification',
                        'entities': entities,
                        'intent': intent
                    }
            
            elif intent.startswith('database_'):
                if intent == 'database_tables':
                    command = "DB SELECT name FROM sqlite_master WHERE type='table';"
                else:
                    command = self._generate_sql_from_intent(intent, entities, natural_query)
                
                if command:
                    return {
                        'command': command,
                        'confidence': 0.85,
                        'method': 'intent_classification',
                        'entities': entities,
                        'intent': intent
                    }
        
        # Method 2: Similarity matching with training data
        best_command, similarity = self._similarity_match(preprocessed)
        if similarity > 0.3:  # Threshold for similarity
            return {
                'command': best_command,
                'confidence': similarity,
                'method': 'similarity_matching',
                'entities': entities
            }
        
        # Method 3: Fallback - try to detect SQL keywords
        sql_keywords = ['select', 'insert', 'update', 'delete', 'create', 'drop', 'alter']
        if any(keyword in preprocessed for keyword in sql_keywords):
            # Assume it's a SQL query
            return {
                'command': f"DB {natural_query}",
                'confidence': 0.6,
                'method': 'sql_detection',
                'entities': entities
            }
        
        # No match found
        return {
            'command': '',
            'confidence': 0.0,
            'method': 'no_match',
            'entities': entities,
            'error': 'Could not understand the query'
        }
    
    def get_suggestions(self, partial_query: str, limit: int = 5) -> List[str]:
        """Get query suggestions based on partial input."""
        if not partial_query.strip():
            return [
                "show cpu usage",
                "show memory status", 
                "list running processes",
                "show all tables",
                "SHOW CPU",
                "SHOW MEMORY",
                "SHOW TASKS",
                "DB SELECT * FROM sqlite_master;"
            ]
        
        partial_lower = partial_query.lower()
        suggestions = []
        
        # Find matching training queries (more liberal matching)
        for query, command in self.training_data:
            if partial_lower in query.lower() or query.lower().startswith(partial_lower):
                suggestions.append(query)
            # Also suggest the formal command
            if partial_lower in command.lower() or command.lower().startswith(partial_lower):
                suggestions.append(command)
        
        # Add context-aware suggestions
        if partial_lower.startswith('sh'):
            suggestions.extend([
                "show cpu usage",
                "show memory status",
                "show running processes",
                "show all tables",
                "SHOW CPU",
                "SHOW MEMORY", 
                "SHOW TASKS"
            ])
        
        if 'show' in partial_lower:
            suggestions.extend([
                "show cpu usage",
                "show memory status",
                "show running processes",
                "show all tables"
            ])
        
        if 'cpu' in partial_lower:
            suggestions.extend([
                "show cpu usage",
                "what is the cpu utilization",
                "check processor load",
                "SHOW CPU"
            ])
        
        if 'memory' in partial_lower or 'ram' in partial_lower:
            suggestions.extend([
                "show memory usage",
                "how much ram is used",
                "display memory status",
                "SHOW MEMORY"
            ])
        
        if 'db' in partial_lower or 'database' in partial_lower:
            suggestions.extend([
                "DB SELECT * FROM sqlite_master;",
                "DB CREATE TABLE test (id INTEGER);",
                "show all tables"
            ])
        
        # Remove duplicates and limit
        suggestions = list(dict.fromkeys(suggestions))
        return suggestions[:limit]

class QueryExpander:
    """Expand abbreviated or incomplete queries."""
    
    def __init__(self):
        self.expansions = {
            'cpu': 'show cpu usage',
            'mem': 'show memory usage',
            'memory': 'show memory usage',
            'processes': 'show running processes',
            'tasks': 'show running processes',
            'tables': 'show all tables',
            'procs': 'show running processes',
            'ram': 'show memory usage'
        }
    
    def expand_query(self, query: str) -> str:
        """Expand abbreviated queries to full natural language."""
        query_lower = query.lower().strip()
        
        if query_lower in self.expansions:
            return self.expansions[query_lower]
        
        # Check for partial matches
        for abbrev, expansion in self.expansions.items():
            if query_lower.startswith(abbrev):
                return expansion
        
        return query

# Example usage and testing
if __name__ == "__main__":
    processor = NLPQueryProcessor()
    expander = QueryExpander()
    
    test_queries = [
        "show me the cpu usage",
        "how much memory is being used",
        "list all running processes",
        "what tables are in the database",
        "select everything from users",
        "cpu",
        "memory status",
        "create a table called logs",
        "insert data into users table"
    ]
    
    print("=== NLP Query Processing Test ===\n")
    
    for query in test_queries:
        expanded = expander.expand_query(query)
        result = processor.process_query(expanded)
        
        print(f"Input: '{query}'")
        if expanded != query:
            print(f"Expanded: '{expanded}'")
        print(f"Command: '{result['command']}'")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Method: {result['method']}")
        if 'intent' in result:
            print(f"Intent: {result['intent']}")
        print("-" * 50)