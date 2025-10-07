# nlp_processor_enhanced.py
"""
Enhanced NLP Processor with improved accuracy and more command support
Includes support for all SPArC features: System, Performance, Arithmetic, File Ops
"""

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
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

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)


class NLPQueryProcessor:
    """Enhanced NLP processor for converting natural language to formal DSL commands."""
    
    def __init__(self):
        try:
            self.stemmer = PorterStemmer()
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english')) - {'to', 'from', 'into', 'all', 'no', 'not'}
        except Exception as e:
            print(f"Warning: NLTK data not fully available: {e}")
            self.stemmer = None
            self.lemmatizer = None
            self.stop_words = set()
        
        # Comprehensive training data for all SPArC features
        self.training_data = [
            # ===== SYSTEM QUERIES - CPU =====
            ("show me the cpu usage", "SHOW CPU"),
            ("what is the processor usage", "SHOW CPU"),
            ("how much cpu is being used", "SHOW CPU"),
            ("display cpu utilization", "SHOW CPU"),
            ("check processor load", "SHOW CPU"),
            ("cpu percentage", "SHOW CPU"),
            ("processor performance", "SHOW CPU"),
            ("system cpu status", "SHOW CPU"),
            ("cpu info", "SHOW CPU"),
            ("get cpu", "SHOW CPU"),
            ("processor info", "SHOW CPU"),
            ("show cpu cores", "SHOW CPU"),
            
            # ===== SYSTEM QUERIES - MEMORY =====
            ("show memory usage", "SHOW MEMORY"),
            ("how much ram is used", "SHOW MEMORY"),
            ("display memory status", "SHOW MEMORY"),
            ("check ram usage", "SHOW MEMORY"),
            ("memory consumption", "SHOW MEMORY"),
            ("available memory", "SHOW MEMORY"),
            ("system memory", "SHOW MEMORY"),
            ("ram status", "SHOW MEMORY"),
            ("memory info", "SHOW MEMORY"),
            ("get ram", "SHOW MEMORY"),
            ("show ram usage", "SHOW MEMORY"),
            
            # ===== SYSTEM QUERIES - GPU =====
            ("show gpu usage", "SHOW GPU"),
            ("graphics card info", "SHOW GPU"),
            ("display gpu status", "SHOW GPU"),
            ("check graphics card", "SHOW GPU"),
            ("gpu memory", "SHOW GPU"),
            ("video card info", "SHOW GPU"),
            ("show graphics card", "SHOW GPU"),
            
            # ===== SYSTEM QUERIES - TEMPERATURE =====
            ("show system temperatures", "SHOW TEMP"),
            ("check cpu temperature", "SHOW TEMP"),
            ("how hot is my system", "SHOW TEMP"),
            ("display temperatures", "SHOW TEMP"),
            ("temp info", "SHOW TEMP"),
            ("system temperature", "SHOW TEMP"),
            
            # ===== SYSTEM QUERIES - DISK =====
            ("show disk usage", "SHOW DISK"),
            ("disk space", "SHOW DISK"),
            ("storage info", "SHOW DISK"),
            ("how much disk space", "SHOW DISK"),
            ("drive information", "SHOW DISK"),
            ("storage status", "SHOW DISK"),
            
            # ===== SYSTEM QUERIES - TASKS/PROCESSES =====
            ("show running processes", "SHOW TASKS"),
            ("list all tasks", "SHOW TASKS"),
            ("what processes are running", "SHOW TASKS"),
            ("display active applications", "SHOW TASKS"),
            ("running programs", "SHOW TASKS"),
            ("active processes", "SHOW TASKS"),
            ("system tasks", "SHOW TASKS"),
            ("process list", "SHOW TASKS"),
            ("show tasks", "SHOW TASKS"),
            ("running apps", "SHOW TASKS"),
            
            # ===== SYSTEM QUERIES - NETWORK =====
            ("show network info", "SHOW NETWORK"),
            ("network status", "SHOW NETWORK"),
            ("display network interfaces", "SHOW NETWORK"),
            ("show network statistics", "SHOW NETWORK"),
            
            # ===== PERFORMANCE CONTROL - MODE =====
            ("set performance mode", "SET MODE PERFORMANCE"),
            ("switch to performance", "SET MODE PERFORMANCE"),
            ("enable high performance", "SET MODE PERFORMANCE"),
            ("performance mode on", "SET MODE PERFORMANCE"),
            ("set balanced mode", "SET MODE BALANCED"),
            ("switch to balanced", "SET MODE BALANCED"),
            ("balanced power plan", "SET MODE BALANCED"),
            ("set power saver mode", "SET MODE POWER_SAVER"),
            ("switch to power saving", "SET MODE POWER_SAVER"),
            ("enable power saver", "SET MODE POWER_SAVER"),
            ("battery saver mode", "SET MODE POWER_SAVER"),
            
            # ===== PERFORMANCE CONTROL - FAN =====
            ("set fan speed to 70", "SET FAN 70"),
            ("fan speed 50 percent", "SET FAN 50"),
            ("set fan to 100", "SET FAN 100"),
            ("adjust fan speed 60", "SET FAN 60"),
            
            # ===== ARITHMETIC OPERATIONS =====
            ("calculate 2 plus 3", "CALC 2+3"),
            ("what is 5 times 10", "CALC 5*10"),
            ("compute 100 divided by 4", "CALC 100/4"),
            ("solve 2 plus 3 times 5", "CALC 2+3*5"),
            ("calculate (5 + 10) divided by 3", "CALC (5+10)/3"),
            ("what is 10 minus 7", "CALC 10-7"),
            ("evaluate 8 plus 2", "CALC 8+2"),
            
            # ===== FILE OPERATIONS - COPY =====
            ("copy file1.txt to file2.txt", "COPY file1.txt file2.txt"),
            ("copy report.docx to backup folder", "COPY report.docx backup/"),
            ("duplicate data.csv to archive", "COPY data.csv archive/"),
            
            # ===== FILE OPERATIONS - MOVE =====
            ("move oldfile.txt to newlocation", "MOVE oldfile.txt newlocation/"),
            ("move data.csv to backup", "MOVE data.csv backup/"),
            ("relocate report.pdf to documents", "MOVE report.pdf documents/"),
            
            # ===== FILE OPERATIONS - DELETE =====
            ("delete oldfile.txt", "DELETE oldfile.txt"),
            ("remove temp.log", "DELETE temp.log"),
            ("erase backup.bak", "DELETE backup.bak"),
            
            # ===== FILE OPERATIONS - OPEN =====
            ("open notepad", "OPEN notepad"),
            ("launch calculator", "OPEN calc"),
            ("open file report.docx", "OPEN report.docx"),
            ("start program chrome", "OPEN chrome"),
            
            # ===== DATABASE - TABLE OPERATIONS =====
            ("show all tables", "DB SELECT name FROM sqlite_master WHERE type='table';"),
            ("list tables", "DB SELECT name FROM sqlite_master WHERE type='table';"),
            ("what tables exist", "DB SELECT name FROM sqlite_master WHERE type='table';"),
            ("display database tables", "DB SELECT name FROM sqlite_master WHERE type='table';"),
            
            # ===== DATABASE - SELECT =====
            ("select all from users", "DB SELECT * FROM users;"),
            ("get all records from logs", "DB SELECT * FROM logs;"),
            ("show everything in users table", "DB SELECT * FROM users;"),
            ("query users table", "DB SELECT * FROM users;"),
            
            # ===== DATABASE - CREATE =====
            ("create a table called users", "DB CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT);"),
            ("make a new table logs", "DB CREATE TABLE logs (id INTEGER PRIMARY KEY, message TEXT);"),
            ("create table products", "DB CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT);"),
            
            # ===== DATABASE - INSERT =====
            ("insert data into users", "DB INSERT INTO users VALUES (1, 'John Doe');"),
            ("add record to logs", "DB INSERT INTO logs VALUES (1, 'Log message');"),
            ("insert into products", "DB INSERT INTO products VALUES (1, 'Product A');"),
        ]
        
        # Enhanced intent patterns with regex
        self.intent_patterns = {
            # System intents
            'system_cpu': [
                r'\b(cpu|processor|processing)\b.*\b(usage|utilization|load|performance|status|info)\b',
                r'\b(show|display|check|get)\b.*\b(cpu|processor)\b',
                r'\b(how much|what is|what\'s)\b.*\b(cpu|processor)\b'
            ],
            'system_memory': [
                r'\b(memory|ram|mem)\b.*\b(usage|status|consumption|available|info)\b',
                r'\b(show|display|check|get)\b.*\b(memory|ram)\b',
                r'\b(how much|what is|what\'s)\b.*\b(memory|ram)\b'
            ],
            'system_gpu': [
                r'\b(gpu|graphics|video card)\b.*\b(usage|status|info|memory)\b',
                r'\b(show|display|check|get)\b.*\b(gpu|graphics|video)\b'
            ],
            'system_temp': [
                r'\b(temp|temperature|thermal)\b.*\b(info|status|sensor)\b',
                r'\b(how hot|show|display)\b.*\b(temp|temperature)\b'
            ],
            'system_disk': [
                r'\b(disk|drive|storage)\b.*\b(usage|space|info|status)\b',
                r'\b(how much|show|display)\b.*\b(disk|storage|space)\b'
            ],
            'system_tasks': [
                r'\b(process|task|program|application|app)\b.*\b(running|active|list)\b',
                r'\b(show|display|list|get)\b.*\b(process|task|program)\b',
                r'\b(what|which)\b.*\b(process|task|program)\b.*\b(running|active)\b'
            ],
            'system_network': [
                r'\b(network|net|interface)\b.*\b(info|status|statistics)\b',
                r'\b(show|display)\b.*\b(network|interface)\b'
            ],
            
            # Performance intents
            'perf_mode': [
                r'\b(set|switch|change|enable)\b.*\b(mode|plan)\b.*\b(performance|balanced|power|saver)\b',
                r'\b(performance|balanced|power saver)\b.*\b(mode|plan)\b'
            ],
            'perf_fan': [
                r'\b(set|adjust|change)\b.*\bfan\b.*\b(speed|percent)\b',
                r'\bfan\b.*\b(speed|percent)\b.*\b\d+\b'
            ],
            
            # Arithmetic intents
            'arithmetic': [
                r'\b(calculate|compute|eval|solve|what is)\b.*[\d\+\-\*\/\(\)]+',
                r'\b\d+\b.*\b(plus|minus|times|divided by|multiply|add|subtract)\b.*\b\d+\b'
            ],
            
            # File operation intents
            'file_copy': [
                r'\b(copy|duplicate)\b.*\b(file|folder)\b.*\b(to|into)\b',
                r'\bcopy\b.*\bto\b'
            ],
            'file_move': [
                r'\b(move|relocate|transfer)\b.*\b(file|folder)\b.*\b(to|into)\b',
                r'\bmove\b.*\bto\b'
            ],
            'file_delete': [
                r'\b(delete|remove|erase)\b.*\b(file|folder)\b',
                r'\b(delete|remove|erase)\b'
            ],
            'file_open': [
                r'\b(open|launch|start|run)\b.*\b(file|program|app)\b',
                r'\b(open|launch|start)\b'
            ],
            
            # Database intents
            'database_tables': [
                r'\b(show|list|display|get)\b.*\btable\b',
                r'\b(what|which)\b.*\btable\b.*\b(exist|available)\b',
                r'\btable\b.*\b(list|show|display)\b'
            ],
            'database_select': [
                r'\b(select|get|show|display|query)\b.*\bfrom\b',
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
                r'\badd\b.*\b(to|into)\b.*\btable\b',
                r'\bput\b.*\bin\b.*\btable\b'
            ]
        }
        
        # Initialize TF-IDF vectorizer
        try:
            self.vectorizer = TfidfVectorizer(
                lowercase=True,
                stop_words='english',
                ngram_range=(1, 3),
                max_features=1000,
                sublinear_tf=True
            )
            
            self._train_model()
            self.initialized = True
        except Exception as e:
            print(f"Warning: TF-IDF initialization failed: {e}")
            self.vectorizer = None
            self.initialized = False
    
    def _train_model(self):
        """Train the NLP model with training data."""
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
        text = text.lower()
        text = re.sub(r'[^\w\s\*\(\)\[\]<>=!+\-\/%]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract entities like table names, numbers, paths, etc."""
        entities = {
            'tables': [],
            'columns': [],
            'numbers': [],
            'paths': [],
            'operators': [],
            'modes': []
        }
        
        # Extract table names
        table_patterns = [
            r'\bfrom\s+(\w+)',
            r'\binto\s+(\w+)',
            r'\btable\s+(\w+)',
            r'\bcalled\s+(\w+)',
            r'\bnamed\s+(\w+)',
        ]
        for pattern in table_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities['tables'].extend(matches)
        
        # Extract numbers (for calculations, fan speed, etc.)
        entities['numbers'] = re.findall(r'\b\d+(?:\.\d+)?\b', text)
        
        # Extract file paths
        path_patterns = [
            r'\b[\w]+\.[a-zA-Z]{2,4}\b',  # Files with extensions
            r'\b[A-Za-z]:\\[^\s]+',  # Windows paths
            r'\b/[^\s]+',  # Unix paths
            r'\.{1,2}/[^\s]+'  # Relative paths
        ]
        for pattern in path_patterns:
            matches = re.findall(pattern, text)
            entities['paths'].extend(matches)
        
        # Extract power modes
        mode_keywords = ['performance', 'balanced', 'power saver', 'powersaver', 'battery']
        for mode in mode_keywords:
            if mode in text.lower():
                entities['modes'].append(mode.replace(' ', '_'))
        
        # Extract arithmetic operators
        operators = ['+', '-', '*', '/', '(', ')']
        for op in operators:
            if op in text:
                entities['operators'].append(op)
        
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
        """Find the most similar training query using TF-IDF."""
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
    
    def _generate_command_from_intent(self, intent: str, entities: Dict[str, List[str]], original_query: str) -> str:
        """Generate formal DSL command from intent and entities."""
        
        # System commands
        if intent == 'system_cpu':
            return 'SHOW CPU'
        elif intent == 'system_memory':
            return 'SHOW MEMORY'
        elif intent == 'system_gpu':
            return 'SHOW GPU'
        elif intent == 'system_temp':
            return 'SHOW TEMP'
        elif intent == 'system_disk':
            return 'SHOW DISK'
        elif intent == 'system_tasks':
            return 'SHOW TASKS'
        elif intent == 'system_network':
            return 'SHOW NETWORK'
        
        # Performance commands
        elif intent == 'perf_mode':
            if entities['modes']:
                mode = entities['modes'][0].upper()
                return f'SET MODE {mode}'
            else:
                # Try to detect mode from query
                if 'performance' in original_query.lower():
                    return 'SET MODE PERFORMANCE'
                elif 'balanced' in original_query.lower():
                    return 'SET MODE BALANCED'
                elif 'power' in original_query.lower() or 'battery' in original_query.lower():
                    return 'SET MODE POWER_SAVER'
        
        elif intent == 'perf_fan':
            if entities['numbers']:
                speed = entities['numbers'][0]
                return f'SET FAN {speed}'
        
        # Arithmetic
        elif intent == 'arithmetic':
            # Extract the mathematical expression
            expr = self._extract_math_expression(original_query)
            if expr:
                return f'CALC {expr}'
        
        # File operations
        elif intent == 'file_copy':
            if len(entities['paths']) >= 2:
                return f"COPY {entities['paths'][0]} {entities['paths'][1]}"
        
        elif intent == 'file_move':
            if len(entities['paths']) >= 2:
                return f"MOVE {entities['paths'][0]} {entities['paths'][1]}"
        
        elif intent == 'file_delete':
            if entities['paths']:
                return f"DELETE {entities['paths'][0]}"
        
        elif intent == 'file_open':
            if entities['paths']:
                return f"OPEN {entities['paths'][0]}"
            else:
                # Try to extract program name
                words = original_query.lower().split()
                for i, word in enumerate(words):
                    if word in ['open', 'launch', 'start', 'run'] and i + 1 < len(words):
                        return f"OPEN {words[i+1]}"
        
        # Database commands
        elif intent == 'database_tables':
            return "DB SELECT name FROM sqlite_master WHERE type='table';"
        
        elif intent == 'database_select':
            if entities['tables']:
                table = entities['tables'][0]
                return f"DB SELECT * FROM {table};"
        
        elif intent == 'database_create':
            if entities['tables']:
                table = re.sub(r'[^\w]', '', entities['tables'][0])
                return f"DB CREATE TABLE {table} (id INTEGER PRIMARY KEY, name TEXT);"
        
        elif intent == 'database_insert':
            if entities['tables']:
                table = entities['tables'][0]
                return f"DB INSERT INTO {table} VALUES (1, 'Sample data');"
        
        return ""
    
    def _extract_math_expression(self, query: str) -> str:
        """Extract mathematical expression from natural language."""
        # Replace word operators with symbols
        replacements = {
            r'\bplus\b': '+',
            r'\bminus\b': '-',
            r'\btimes\b': '*',
            r'\bmultiplied by\b': '*',
            r'\bdivided by\b': '/',
            r'\bover\b': '/',
            r'\badd\b': '+',
            r'\bsubtract\b': '-',
            r'\bmultiply\b': '*',
            r'\bdivide\b': '/'
        }
        
        expr = query.lower()
        for pattern, replacement in replacements.items():
            expr = re.sub(pattern, replacement, expr)
        
        # Extract expression with numbers and operators
        match = re.search(r'[\d\+\-\*\/\(\)\s]+', expr)
        if match:
            return match.group().strip()
        
        return ""
    
    def process_query(self, natural_query: str) -> Dict[str, any]:
        """
        Process natural language query and convert to formal DSL.
        
        Returns:
            Dict with command, confidence, method, entities, etc.
        """
        if not natural_query.strip():
            return {
                'command': '',
                'confidence': 0.0,
                'method': 'empty',
                'entities': {},
                'error': 'Empty query'
            }
        
        preprocessed = self._preprocess_text(natural_query)
        entities = self._extract_entities(preprocessed)
        
        # Method 1: Intent-based classification
        intent = self._classify_intent(preprocessed)
        if intent:
            command = self._generate_command_from_intent(intent, entities, natural_query)
            if command:
                return {
                    'command': command,
                    'confidence': 0.9,
                    'method': 'intent_classification',
                    'entities': entities,
                    'intent': intent
                }
        
        # Method 2: Similarity matching
        best_command, similarity = self._similarity_match(preprocessed)
        if similarity > 0.3:
            return {
                'command': best_command,
                'confidence': similarity,
                'method': 'similarity_matching',
                'entities': entities
            }
        
        # Method 3: Keyword detection fallback
        keywords_map = {
            'cpu': 'SHOW CPU',
            'memory': 'SHOW MEMORY',
            'ram': 'SHOW MEMORY',
            'gpu': 'SHOW GPU',
            'temp': 'SHOW TEMP',
            'disk': 'SHOW DISK',
            'tasks': 'SHOW TASKS',
            'processes': 'SHOW TASKS',
            'network': 'SHOW NETWORK'
        }
        
        for keyword, command in keywords_map.items():
            if keyword in preprocessed:
                return {
                    'command': command,
                    'confidence': 0.7,
                    'method': 'keyword_detection',
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
    
    def get_suggestions(self, partial_query: str, limit: int = 8) -> List[str]:
        """Get query suggestions for autocomplete."""
        if not partial_query.strip():
            return [
                "show cpu usage",
                "show memory status",
                "list running processes",
                "calculate 2 + 3 * 5",
                "set performance mode",
                "show all tables",
                "copy file.txt to backup/",
                "SHOW CPU"
            ]
        
        partial_lower = partial_query.lower()
        suggestions = []
        
        # Find matching training queries
        for query, command in self.training_data:
            if partial_lower in query.lower():
                suggestions.append(query)
            if partial_lower in command.lower():
                suggestions.append(command)
        
        # Context-aware suggestions
        if any(word in partial_lower for word in ['show', 'display', 'get']):
            suggestions.extend([
                "show cpu usage",
                "show memory status",
                "show disk space",
                "show running processes",
                "show gpu info",
                "show all tables"
            ])
        
        if any(word in partial_lower for word in ['calc', 'compute', 'calculate']):
            suggestions.extend([
                "calculate 2 + 3 * 5",
                "calculate (10 + 20) / 2",
                "calculate 100 - 45"
            ])
        
        if 'set' in partial_lower:
            suggestions.extend([
                "set performance mode",
                "set balanced mode",
                "set fan 70"
            ])
        
        if any(word in partial_lower for word in ['copy', 'move', 'delete', 'open']):
            suggestions.extend([
                "copy file.txt to backup/",
                "move oldfile.txt to archive/",
                "delete tempfile.txt",
                "open notepad"
            ])
        
        # Remove duplicates and limit
        suggestions = list(dict.fromkeys(suggestions))
        return suggestions[:limit]


class QueryExpander:
    """Expand abbreviated or shorthand queries."""
    
    def __init__(self):
        self.expansions = {
            # System shortcuts
            'cpu': 'show cpu usage',
            'mem': 'show memory usage',
            'memory': 'show memory usage',
            'ram': 'show memory usage',
            'gpu': 'show gpu info',
            'temp': 'show system temperatures',
            'disk': 'show disk usage',
            'storage': 'show disk usage',
            'processes': 'show running processes',
            'tasks': 'show running processes',
            'procs': 'show running processes',
            'network': 'show network info',
            'net': 'show network info',
            
            # Performance shortcuts
            'perf': 'set performance mode',
            'performance': 'set performance mode',
            'balanced': 'set balanced mode',
            'saver': 'set power saver mode',
            
            # Database shortcuts
            'tables': 'show all tables',
            'db': 'show all tables',
            
            # File operation shortcuts
            'calc': 'calculate',
            'compute': 'calculate',
        }
    
    def expand_query(self, query: str) -> str:
        """Expand abbreviated queries to full form."""
        query_lower = query.lower().strip()
        
        # Direct match
        if query_lower in self.expansions:
            return self.expansions[query_lower]
        
        # Partial match at start
        for abbrev, expansion in self.expansions.items():
            if query_lower.startswith(abbrev + ' ') or query_lower == abbrev:
                # Replace only the abbreviated part
                return expansion + query[len(abbrev):]
        
        return query


# Example usage and testing
if __name__ == "__main__":
    print("=== Enhanced NLP Query Processor Test ===\n")
    
    processor = NLPQueryProcessor()
    expander = QueryExpander()
    
    test_queries = [
        # System queries
        "show me the cpu usage",
        "how much memory is used",
        "list all running processes",
        "gpu info",
        "system temperatures",
        
        # Performance
        "set performance mode",
        "switch to power saver",
        "set fan speed to 75",
        
        # Arithmetic
        "calculate 2 plus 3 times 5",
        "what is 100 divided by 4",
        
        # File operations
        "copy report.docx to backup folder",
        "delete oldfile.txt",
        "open notepad",
        
        # Database
        "show all tables",
        "select everything from users",
        "create a table called products",
        
        # Abbreviations
        "cpu",
        "memory",
        "tasks",
    ]
    
    for query in test_queries:
        expanded = expander.expand_query(query)
        result = processor.process_query(expanded)
        
        print(f"Input: '{query}'")
        if expanded != query:
            print(f"Expanded: '{expanded}'")
        print(f"Command: '{result['command']}'")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Method: {result['method']}")
        if 'intent' in result:
            print(f"Intent: {result['intent']}")
        print("-" * 60)
