# dsl_core.py
import psutil
import sqlite3
import re
import os
from enum import Enum
from typing import List, Optional, Dict, Any
from nlp_processor import NLPQueryProcessor, QueryExpander

# ================================
# 1. TOKENS (Regular Languages + DFA)
# ================================
class TokenType(Enum):
    KEYWORD = "KEYWORD"
    IDENTIFIER = "IDENTIFIER"
    STRING = "STRING"
    NUMBER = "NUMBER"
    OPERATOR = "OPERATOR"
    LPAREN = "LPAREN"
    RPAREN = "RPAREN"
    COMMA = "COMMA"
    SEMICOLON = "SEMICOLON"
    EOF = "EOF"

class Token:
    def __init__(self, type_: TokenType, value: str, line: int = 1, col: int = 1):
        self.type = type_
        self.value = value
        self.line = line
        self.col = col

    def __repr__(self):
        return f"Token({self.type}, '{self.value}')"

# ================================
# 2. LEXER (DFA-inspired)
# ================================
class Lexer:
    KEYWORDS = {"SHOW", "TASKS", "CPU", "MEMORY", "DB", "SELECT", "FROM", "WHERE", "INSERT", "INTO", "VALUES", "CREATE", "TABLE", "DROP"}
    OPERATORS = {"=", "!=", "<", ">", "<=", ">=", "+", "-", "*", "/", "(", ")", ",", ";"}

    def __init__(self, text: str):
        self.text = re.sub(r'--.*', '', text)  # Remove comments
        self.pos = 0
        self.line = 1
        self.col = 1

    def tokenize(self) -> List[Token]:
        tokens = []
        while self.pos < len(self.text):
            char = self.text[self.pos]

            if char.isspace():
                if char == '\n':
                    self.line += 1
                    self.col = 1
                else:
                    self.col += 1
                self.pos += 1
                continue

            if char == '"':
                start = self.pos + 1
                self.pos += 1
                chars = []
                while self.pos < len(self.text) and self.text[self.pos] != '"':
                    if self.text[self.pos] == '\\':
                        self.pos += 1
                        if self.pos >= len(self.text):
                            break
                    chars.append(self.text[self.pos])
                    self.pos += 1
                if self.pos >= len(self.text):
                    raise SyntaxError(f"Unterminated string at line {self.line}")
                self.pos += 1  # skip closing quote
                value = ''.join(chars)
                tokens.append(Token(TokenType.STRING, value, self.line, self.col))
                self.col += len(value) + 2
                continue

            if char.isdigit():
                start = self.pos
                while self.pos < len(self.text) and (self.text[self.pos].isdigit() or self.text[self.pos] == '.'):
                    self.pos += 1
                value = self.text[start:self.pos]
                tokens.append(Token(TokenType.NUMBER, value, self.line, self.col))
                self.col += len(value)
                continue

            if char.isalpha() or char == '_':
                start = self.pos
                while self.pos < len(self.text) and (self.text[self.pos].isalnum() or self.text[self.pos] == '_'):
                    self.pos += 1
                value = self.text[start:self.pos]
                tok_type = TokenType.KEYWORD if value.upper() in self.KEYWORDS else TokenType.IDENTIFIER
                tokens.append(Token(tok_type, value, self.line, self.col))
                self.col += len(value)
                continue

            if char in "()":
                tok_type = TokenType.LPAREN if char == '(' else TokenType.RPAREN
                tokens.append(Token(tok_type, char, self.line, self.col))
                self.col += 1
                self.pos += 1
                continue

            if char == ',':
                tokens.append(Token(TokenType.COMMA, ',', self.line, self.col))
                self.col += 1
                self.pos += 1
                continue

            if char == ';':
                tokens.append(Token(TokenType.SEMICOLON, ';', self.line, self.col))
                self.col += 1
                self.pos += 1
                continue

            # Operators
            op = char
            if self.pos + 1 < len(self.text):
                next_char = self.text[self.pos + 1]
                if op + next_char in {"<=", ">=", "!=", "=="}:
                    op += next_char
                    self.pos += 1
            self.pos += 1
            tokens.append(Token(TokenType.OPERATOR, op, self.line, self.col))
            self.col += len(op)

        tokens.append(Token(TokenType.EOF, "", self.line, self.col))
        return tokens

# ================================
# 3. AST NODES (CFG)
# ================================
class ASTNode:
    pass

class ShowNode(ASTNode):
    def __init__(self, target: str):
        self.target = target.upper()

    def __repr__(self):
        return f"ShowNode(target='{self.target}')"

class DBQueryNode(ASTNode):
    def __init__(self, raw_sql: str):
        self.raw_sql = raw_sql

    def __repr__(self):
        return f"DBQueryNode(sql='{self.raw_sql}')"

# ================================
# 4. PARSER (Recursive Descent - CFG)
# ================================
class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    def current(self) -> Token:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return Token(TokenType.EOF, "", 0, 0)

    def eat(self, expected_type: TokenType, expected_value: Optional[str] = None):
        token = self.current()
        if token.type != expected_type:
            raise SyntaxError(f"Expected {expected_type}, got {token.type} at line {token.line}, col {token.col}")
        if expected_value and token.value.upper() != expected_value.upper():
            raise SyntaxError(f"Expected '{expected_value}', got '{token.value}' at line {token.line}, col {token.col}")
        self.pos += 1
        return token

    def parse(self) -> ASTNode:
        token = self.current()
        if token.type == TokenType.KEYWORD:
            if token.value.upper() == "SHOW":
                return self._parse_show()
            elif token.value.upper() == "DB":
                return self._parse_db()
        if token.type == TokenType.EOF:
            raise SyntaxError("Empty query")
        raise SyntaxError(f"Unexpected token: {token.value} at line {token.line}, col {token.col}")

    def _parse_show(self) -> ShowNode:
        self.eat(TokenType.KEYWORD, "SHOW")
        target_token = self.current()
        if target_token.type != TokenType.KEYWORD:
            raise SyntaxError(f"Expected target (TASKS, CPU, MEMORY), got {target_token.value}")
        self.eat(TokenType.KEYWORD)
        return ShowNode(target_token.value)

    def _parse_db(self) -> DBQueryNode:
        self.eat(TokenType.KEYWORD, "DB")
        sql_parts = []
        while self.current().type != TokenType.EOF:
            sql_parts.append(self.current().value)
            self.pos += 1
        raw_sql = " ".join(sql_parts)
        return DBQueryNode(raw_sql)

# ================================
# 5. SEMANTIC ANALYZER
# ================================
class SemanticAnalyzer:
    VALID_SHOW_TARGETS = {"TASKS", "CPU", "MEMORY"}

    def analyze(self, node: ASTNode):
        if isinstance(node, ShowNode):
            if node.target not in self.VALID_SHOW_TARGETS:
                raise ValueError(f"Invalid SHOW target: {node.target}. Valid: {self.VALID_SHOW_TARGETS}")

# ================================
# 6. EXECUTION ENGINE
# ================================
DB_FILE = "sparc.db"

class ExecutionEngine:
    def __init__(self, enable_nlp: bool = True):
        self.db_path = DB_FILE
        self._init_db()
        self.semantic_analyzer = SemanticAnalyzer()
        
        # NLP components
        self.enable_nlp = enable_nlp
        self.nlp_processor = None
        self.query_expander = None
        
        if enable_nlp:
            try:
                print("Initializing NLP components...")
                self.nlp_processor = NLPQueryProcessor()
                self.query_expander = QueryExpander()
                print("✅ NLP initialization successful")
            except ImportError as e:
                print(f"⚠️ NLP dependencies missing: {e}")
                print("Install with: pip install nltk scikit-learn")
                self.enable_nlp = False
            except Exception as e:
                print(f"⚠️ NLP initialization failed: {e}")
                self.enable_nlp = False

    def _init_db(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)

    def execute(self, query: str, use_nlp: bool = None) -> Dict[str, Any]:
        """
        Execute a query with optional NLP processing.
        
        Args:
            query: The input query (natural language or formal DSL)
            use_nlp: Override the default NLP setting for this query
            
        Returns:
            Dict containing result, metadata, and processing info
        """
        if not query.strip():
            return {
                'result': "",
                'success': True,
                'method': 'empty',
                'original_query': query,
                'formal_query': "",
                'confidence': 0.0
            }

        # Determine if we should use NLP
        should_use_nlp = use_nlp if use_nlp is not None else self.enable_nlp
        formal_query = query
        nlp_info = {}
        
        try:
            # Try NLP processing first if enabled
            if should_use_nlp and self.nlp_processor:
                # Check if it's already a formal query
                if not (query.upper().startswith(("SHOW ", "DB "))):
                    # Expand abbreviated queries
                    expanded_query = self.query_expander.expand_query(query)
                    
                    # Process with NLP
                    nlp_result = self.nlp_processor.process_query(expanded_query)
                    nlp_info = {
                        'expanded_query': expanded_query,
                        'nlp_confidence': nlp_result['confidence'],
                        'nlp_method': nlp_result['method'],
                        'entities': nlp_result.get('entities', {}),
                        'intent': nlp_result.get('intent', 'unknown')
                    }
                    
                    if nlp_result['command'] and nlp_result['confidence'] > 0.3:
                        formal_query = nlp_result['command']
                        nlp_info['nlp_used'] = True
                    else:
                        # NLP failed, try formal parsing
                        nlp_info['nlp_used'] = False
                        nlp_info['nlp_error'] = nlp_result.get('error', 'Low confidence')
                else:
                    nlp_info['nlp_used'] = False
                    nlp_info['reason'] = 'Already formal query'
            
            # FLA Pipeline - process the formal query
            lexer = Lexer(formal_query)
            tokens = lexer.tokenize()
            parser = Parser(tokens)
            ast = parser.parse()
            self.semantic_analyzer.analyze(ast)

            # Execute based on AST type
            if isinstance(ast, ShowNode):
                result = self._exec_show(ast.target)
                return {
                    'result': result,
                    'success': True,
                    'method': 'formal_dsl',
                    'original_query': query,
                    'formal_query': formal_query,
                    'ast_type': 'ShowNode',
                    'target': ast.target,
                    **nlp_info
                }
            elif isinstance(ast, DBQueryNode):
                result = self._exec_sql(ast.raw_sql)
                return {
                    'result': result,
                    'success': True,
                    'method': 'formal_dsl',
                    'original_query': query,
                    'formal_query': formal_query,
                    'ast_type': 'DBQueryNode',
                    'sql': ast.raw_sql,
                    **nlp_info
                }
            else:
                return {
                    'result': "❌ Unknown AST node",
                    'success': False,
                    'method': 'formal_dsl',
                    'original_query': query,
                    'formal_query': formal_query,
                    'error': 'Unknown AST node type',
                    **nlp_info
                }
                
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            
            # If formal parsing failed and we haven't tried NLP yet, try it now
            if should_use_nlp and self.nlp_processor and not nlp_info.get('nlp_used', False):
                try:
                    expanded_query = self.query_expander.expand_query(query)
                    nlp_result = self.nlp_processor.process_query(expanded_query)
                    
                    if nlp_result['command'] and nlp_result['confidence'] > 0.2:  # Lower threshold for fallback
                        # Try executing the NLP result
                        return self.execute(nlp_result['command'], use_nlp=False)
                except:
                    pass
            
            return {
                'result': error_msg,
                'success': False,
                'method': 'error',
                'original_query': query,
                'formal_query': formal_query,
                'error': str(e),
                **nlp_info
            }
    
    def execute_simple(self, query: str) -> str:
        """Simple execution method that returns just the result string (for backward compatibility)."""
        result_dict = self.execute(query)
        return result_dict['result']
    
    def get_query_suggestions(self, partial_query: str, limit: int = 5) -> List[str]:
        """Get query suggestions for autocomplete."""
        if self.nlp_processor:
            return self.nlp_processor.get_suggestions(partial_query, limit)
        return []

    def _exec_show(self, target: str) -> str:
        if target == "TASKS":
            lines = ["=== 🖥️ Running Tasks ==="]
            for p in psutil.process_iter(['pid', 'name']):
                lines.append(f"PID={p.info['pid']:5d} | NAME={p.info['name']}")
            return "\n".join(lines)
        elif target == "CPU":
            usage = psutil.cpu_percent(interval=0.5)
            return f"=== ⚡ CPU Usage ===\n{usage:.2f}%"
        elif target == "MEMORY":
            mem = psutil.virtual_memory()
            return (
                f"=== 💾 Memory Usage ===\n"
                f"Total: {mem.total // (1024**3)} GB | "
                f"Used: {mem.used // (1024**3)} GB | "
                f"{mem.percent:.1f}%"
            )

    def _exec_sql(self, sql: str) -> str:
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            cur.execute(sql)
            if sql.strip().upper().startswith("SELECT") or sql.strip().upper().startswith("PRAGMA"):
                rows = cur.fetchall()
                conn.close()
                if not rows:
                    return "✅ Query executed (no results)"
                return "\n".join(str(row) for row in rows)
            else:
                conn.commit()
                conn.close()
                return "✅ Query executed successfully"
        except sqlite3.Error as e:
            return f"❌ SQL Error: {e}"
        except Exception as e:
            return f"❌ Execution Error: {e}"

    def close(self):
        pass