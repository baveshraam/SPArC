# dsl_core_enhanced.py
"""
Enhanced DSL Core with Full SPArC Feature Set
Includes: System Queries, Performance Control, Arithmetic, File Operations
"""

import psutil
import sqlite3
import re
import os
import shutil
import subprocess
import platform
import ast as python_ast
from enum import Enum
from typing import List, Optional, Dict, Any, Tuple
from nlp_processor_enhanced import NLPQueryProcessor, QueryExpander

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

try:
    import wmi
    WMI_AVAILABLE = True
except ImportError:
    WMI_AVAILABLE = False


# ================================
# 1. TOKENS (Regular Languages + DFA)
# ================================
class TokenType(Enum):
    # Keywords
    KEYWORD = "KEYWORD"
    IDENTIFIER = "IDENTIFIER"
    STRING = "STRING"
    NUMBER = "NUMBER"
    FLOAT = "FLOAT"
    # Operators
    OPERATOR = "OPERATOR"
    LPAREN = "LPAREN"
    RPAREN = "RPAREN"
    COMMA = "COMMA"
    SEMICOLON = "SEMICOLON"
    EQUALS = "EQUALS"
    # Special
    PATH = "PATH"
    EOF = "EOF"


class Token:
    def __init__(self, type_: TokenType, value: str, line: int = 1, col: int = 1):
        self.type = type_
        self.value = value
        self.line = line
        self.col = col

    def __repr__(self):
        return f"Token({self.type}, '{self.value}', L{self.line}:C{self.col})"


# ================================
# 2. LEXER (DFA-inspired)
# ================================
class Lexer:
    """Enhanced Lexer supporting all SPArC commands"""
    
    KEYWORDS = {
        # System queries
        "SHOW", "TASKS", "CPU", "MEMORY", "GPU", "TEMP", "DISK", "NETWORK",
        "PROCESS", "LIST", "INFO", "USAGE", "STATUS",
        # Performance control
        "SET", "MODE", "FAN", "PERFORMANCE", "BALANCED", "POWER_SAVER",
        # Arithmetic
        "CALC", "CALCULATE", "EVAL",
        # File operations
        "COPY", "MOVE", "DELETE", "OPEN", "TO", "FROM",
        # Database
        "DB", "SELECT", "WHERE", "INSERT", "INTO", "VALUES",
        "CREATE", "TABLE", "DROP", "UPDATE", "ALTER"
    }
    
    OPERATORS = {"=", "!=", "<", ">", "<=", ">=", "+", "-", "*", "/", "(", ")", ",", ";"}

    def __init__(self, text: str):
        # Remove comments (-- style and # style)
        self.text = re.sub(r'(--|#).*', '', text)
        self.pos = 0
        self.line = 1
        self.col = 1

    def tokenize(self) -> List[Token]:
        """Tokenize input into list of tokens (Lexical Analysis)"""
        tokens = []
        
        while self.pos < len(self.text):
            char = self.text[self.pos]

            # Skip whitespace
            if char.isspace():
                if char == '\n':
                    self.line += 1
                    self.col = 1
                else:
                    self.col += 1
                self.pos += 1
                continue

            # String literals (quoted)
            if char in ('"', "'"):
                token = self._tokenize_string(char)
                tokens.append(token)
                continue

            # Numbers (integer or float)
            if char.isdigit() or (char == '.' and self.pos + 1 < len(self.text) and self.text[self.pos + 1].isdigit()):
                token = self._tokenize_number()
                tokens.append(token)
                continue

            # Identifiers and keywords
            if char.isalpha() or char == '_':
                token = self._tokenize_identifier()
                tokens.append(token)
                continue

            # File paths (Windows/Unix style)
            if char in ('.', '/', '\\') or (char.isalpha() and self.pos + 1 < len(self.text) and self.text[self.pos + 1] == ':'):
                # Check if this looks like a path
                if self._looks_like_path():
                    token = self._tokenize_path()
                    tokens.append(token)
                    continue

            # Parentheses
            if char in "()":
                tok_type = TokenType.LPAREN if char == '(' else TokenType.RPAREN
                tokens.append(Token(tok_type, char, self.line, self.col))
                self.col += 1
                self.pos += 1
                continue

            # Comma
            if char == ',':
                tokens.append(Token(TokenType.COMMA, ',', self.line, self.col))
                self.col += 1
                self.pos += 1
                continue

            # Semicolon
            if char == ';':
                tokens.append(Token(TokenType.SEMICOLON, ';', self.line, self.col))
                self.col += 1
                self.pos += 1
                continue

            # Operators (including multi-char like <=, >=, !=, ==)
            if char in "=!<>+-*/":
                token = self._tokenize_operator()
                tokens.append(token)
                continue

            # Unknown character
            raise SyntaxError(f"Unknown character '{char}' at line {self.line}, col {self.col}")

        tokens.append(Token(TokenType.EOF, "", self.line, self.col))
        return tokens

    def _tokenize_string(self, quote_char: str) -> Token:
        """Tokenize a quoted string"""
        start_col = self.col
        self.pos += 1  # Skip opening quote
        self.col += 1
        chars = []
        
        while self.pos < len(self.text) and self.text[self.pos] != quote_char:
            if self.text[self.pos] == '\\' and self.pos + 1 < len(self.text):
                # Handle escape sequences
                self.pos += 1
                escape_char = self.text[self.pos]
                if escape_char == 'n':
                    chars.append('\n')
                elif escape_char == 't':
                    chars.append('\t')
                elif escape_char == '\\':
                    chars.append('\\')
                elif escape_char == quote_char:
                    chars.append(quote_char)
                else:
                    chars.append(escape_char)
            else:
                chars.append(self.text[self.pos])
            self.pos += 1
            self.col += 1
        
        if self.pos >= len(self.text):
            raise SyntaxError(f"Unterminated string at line {self.line}, col {start_col}")
        
        self.pos += 1  # Skip closing quote
        self.col += 1
        value = ''.join(chars)
        return Token(TokenType.STRING, value, self.line, start_col)

    def _tokenize_number(self) -> Token:
        """Tokenize a number (integer or float)"""
        start = self.pos
        start_col = self.col
        has_dot = False
        
        while self.pos < len(self.text):
            char = self.text[self.pos]
            if char.isdigit():
                self.pos += 1
                self.col += 1
            elif char == '.' and not has_dot:
                has_dot = True
                self.pos += 1
                self.col += 1
            else:
                break
        
        value = self.text[start:self.pos]
        tok_type = TokenType.FLOAT if has_dot else TokenType.NUMBER
        return Token(tok_type, value, self.line, start_col)

    def _tokenize_identifier(self) -> Token:
        """Tokenize an identifier or keyword"""
        start = self.pos
        start_col = self.col
        
        while self.pos < len(self.text) and (self.text[self.pos].isalnum() or self.text[self.pos] == '_'):
            self.pos += 1
            self.col += 1
        
        value = self.text[start:self.pos]
        tok_type = TokenType.KEYWORD if value.upper() in self.KEYWORDS else TokenType.IDENTIFIER
        return Token(tok_type, value, self.line, start_col)

    def _looks_like_path(self) -> bool:
        """Check if the current position looks like a file path"""
        # Save current position
        saved_pos = self.pos
        
        # Look ahead for path patterns
        remaining = self.text[self.pos:]
        
        # Check for common path patterns
        path_patterns = [
            r'^[A-Za-z]:\\',  # Windows absolute: C:\
            r'^\.{1,2}[/\\]',  # Relative: ./ or ../
            r'^/',  # Unix absolute: /
            r'^\w+\.\w+',  # File with extension
        ]
        
        for pattern in path_patterns:
            if re.match(pattern, remaining):
                return True
        
        return False

    def _tokenize_path(self) -> Token:
        """Tokenize a file path"""
        start = self.pos
        start_col = self.col
        
        # Read until whitespace or special character
        while self.pos < len(self.text):
            char = self.text[self.pos]
            if char.isspace() or char in ',;)':
                break
            self.pos += 1
            self.col += 1
        
        value = self.text[start:self.pos]
        return Token(TokenType.PATH, value, self.line, start_col)

    def _tokenize_operator(self) -> Token:
        """Tokenize an operator (single or multi-character)"""
        start_col = self.col
        char = self.text[self.pos]
        op = char
        
        # Check for multi-character operators
        if self.pos + 1 < len(self.text):
            next_char = self.text[self.pos + 1]
            two_char = char + next_char
            if two_char in {"<=", ">=", "!=", "=="}:
                op = two_char
                self.pos += 2
                self.col += 2
                return Token(TokenType.OPERATOR, op, self.line, start_col)
        
        self.pos += 1
        self.col += 1
        
        if char == '=':
            return Token(TokenType.EQUALS, op, self.line, start_col)
        else:
            return Token(TokenType.OPERATOR, op, self.line, start_col)


# ================================
# 3. AST NODES (CFG)
# ================================
class ASTNode:
    """Base class for Abstract Syntax Tree nodes"""
    pass


class ShowNode(ASTNode):
    """AST node for SHOW commands (system queries)"""
    def __init__(self, target: str):
        self.target = target.upper()

    def __repr__(self):
        return f"ShowNode(target='{self.target}')"


class DBQueryNode(ASTNode):
    """AST node for database operations"""
    def __init__(self, raw_sql: str):
        self.raw_sql = raw_sql

    def __repr__(self):
        return f"DBQueryNode(sql='{self.raw_sql}')"


class CalcNode(ASTNode):
    """AST node for arithmetic calculations"""
    def __init__(self, expression: str):
        self.expression = expression

    def __repr__(self):
        return f"CalcNode(expr='{self.expression}')"


class SetModeNode(ASTNode):
    """AST node for performance mode settings"""
    def __init__(self, mode: str):
        self.mode = mode.lower()

    def __repr__(self):
        return f"SetModeNode(mode='{self.mode}')"


class SetFanNode(ASTNode):
    """AST node for fan speed control"""
    def __init__(self, speed: int):
        self.speed = speed

    def __repr__(self):
        return f"SetFanNode(speed={self.speed})"


class FileOpNode(ASTNode):
    """AST node for file operations"""
    def __init__(self, operation: str, source: str, destination: str = None):
        self.operation = operation.upper()
        self.source = source
        self.destination = destination

    def __repr__(self):
        if self.destination:
            return f"FileOpNode(op='{self.operation}', src='{self.source}', dest='{self.destination}')"
        return f"FileOpNode(op='{self.operation}', src='{self.source}')"


# ================================
# 4. PARSER (Recursive Descent - CFG)
# ================================
class Parser:
    """Enhanced Parser supporting full SPArC grammar"""
    
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    def current(self) -> Token:
        """Get current token"""
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return Token(TokenType.EOF, "", 0, 0)

    def peek(self, offset: int = 1) -> Token:
        """Look ahead at token"""
        if self.pos + offset < len(self.tokens):
            return self.tokens[self.pos + offset]
        return Token(TokenType.EOF, "", 0, 0)

    def eat(self, expected_type: TokenType, expected_value: Optional[str] = None) -> Token:
        """Consume a token if it matches expectations"""
        token = self.current()
        
        if token.type != expected_type:
            raise SyntaxError(
                f"Expected {expected_type}, got {token.type} ('{token.value}') "
                f"at line {token.line}, col {token.col}"
            )
        
        if expected_value and token.value.upper() != expected_value.upper():
            raise SyntaxError(
                f"Expected '{expected_value}', got '{token.value}' "
                f"at line {token.line}, col {token.col}"
            )
        
        self.pos += 1
        return token

    def parse(self) -> ASTNode:
        """Parse tokens into AST (Syntax Analysis)"""
        token = self.current()
        
        if token.type == TokenType.EOF:
            raise SyntaxError("Empty query")
        
        if token.type != TokenType.KEYWORD:
            raise SyntaxError(
                f"Expected command keyword, got {token.type} ('{token.value}') "
                f"at line {token.line}, col {token.col}"
            )
        
        keyword = token.value.upper()
        
        # Route to appropriate parser based on keyword
        if keyword == "SHOW":
            return self._parse_show()
        elif keyword == "DB":
            return self._parse_db()
        elif keyword in ("CALC", "CALCULATE", "EVAL"):
            return self._parse_calc()
        elif keyword == "SET":
            return self._parse_set()
        elif keyword in ("COPY", "MOVE", "DELETE", "OPEN"):
            return self._parse_file_operation()
        else:
            raise SyntaxError(f"Unknown command: {keyword} at line {token.line}, col {token.col}")

    def _parse_show(self) -> ShowNode:
        """Parse SHOW command"""
        self.eat(TokenType.KEYWORD, "SHOW")
        target_token = self.current()
        
        if target_token.type != TokenType.KEYWORD:
            raise SyntaxError(
                f"Expected target (CPU, MEMORY, GPU, TEMP, DISK, TASKS), "
                f"got {target_token.value} at line {target_token.line}, col {target_token.col}"
            )
        
        self.eat(TokenType.KEYWORD)
        return ShowNode(target_token.value)

    def _parse_db(self) -> DBQueryNode:
        """Parse database command"""
        self.eat(TokenType.KEYWORD, "DB")
        
        # Collect all remaining tokens as SQL
        sql_parts = []
        while self.current().type != TokenType.EOF:
            sql_parts.append(self.current().value)
            self.pos += 1
        
        raw_sql = " ".join(sql_parts)
        return DBQueryNode(raw_sql)

    def _parse_calc(self) -> CalcNode:
        """Parse calculation command"""
        self.eat(TokenType.KEYWORD)  # CALC/CALCULATE/EVAL
        
        # Collect expression tokens
        expr_parts = []
        paren_depth = 0
        
        while self.current().type != TokenType.EOF:
            token = self.current()
            
            if token.type == TokenType.LPAREN:
                paren_depth += 1
            elif token.type == TokenType.RPAREN:
                paren_depth -= 1
            
            expr_parts.append(token.value)
            self.pos += 1
            
            # Stop at semicolon if not inside parentheses
            if token.type == TokenType.SEMICOLON and paren_depth == 0:
                break
        
        expression = " ".join(expr_parts).strip().rstrip(';')
        return CalcNode(expression)

    def _parse_set(self) -> ASTNode:
        """Parse SET command (mode or fan)"""
        self.eat(TokenType.KEYWORD, "SET")
        
        next_token = self.current()
        
        if next_token.value.upper() == "MODE":
            return self._parse_set_mode()
        elif next_token.value.upper() == "FAN":
            return self._parse_set_fan()
        else:
            raise SyntaxError(
                f"Expected MODE or FAN after SET, got {next_token.value} "
                f"at line {next_token.line}, col {next_token.col}"
            )

    def _parse_set_mode(self) -> SetModeNode:
        """Parse SET MODE command"""
        self.eat(TokenType.KEYWORD, "MODE")
        mode_token = self.current()
        
        if mode_token.type != TokenType.KEYWORD and mode_token.type != TokenType.IDENTIFIER:
            raise SyntaxError(
                f"Expected mode (PERFORMANCE, BALANCED, POWER_SAVER), "
                f"got {mode_token.value} at line {mode_token.line}, col {mode_token.col}"
            )
        
        self.pos += 1
        return SetModeNode(mode_token.value)

    def _parse_set_fan(self) -> SetFanNode:
        """Parse SET FAN command"""
        self.eat(TokenType.KEYWORD, "FAN")
        speed_token = self.current()
        
        if speed_token.type != TokenType.NUMBER:
            raise SyntaxError(
                f"Expected number for fan speed, got {speed_token.value} "
                f"at line {speed_token.line}, col {speed_token.col}"
            )
        
        self.pos += 1
        speed = int(speed_token.value)
        return SetFanNode(speed)

    def _parse_file_operation(self) -> FileOpNode:
        """Parse file operation command"""
        op_token = self.current()
        operation = op_token.value.upper()
        self.eat(TokenType.KEYWORD)
        
        # Get source path
        source_token = self.current()
        if source_token.type not in (TokenType.STRING, TokenType.PATH, TokenType.IDENTIFIER):
            raise SyntaxError(
                f"Expected file path, got {source_token.value} "
                f"at line {source_token.line}, col {source_token.col}"
            )
        
        source = source_token.value
        self.pos += 1
        
        # For COPY and MOVE, get destination
        if operation in ("COPY", "MOVE"):
            # Optional TO keyword
            if self.current().value.upper() == "TO":
                self.eat(TokenType.KEYWORD, "TO")
            
            dest_token = self.current()
            if dest_token.type not in (TokenType.STRING, TokenType.PATH, TokenType.IDENTIFIER):
                raise SyntaxError(
                    f"Expected destination path, got {dest_token.value} "
                    f"at line {dest_token.line}, col {dest_token.col}"
                )
            
            destination = dest_token.value
            self.pos += 1
            return FileOpNode(operation, source, destination)
        
        # DELETE and OPEN don't need destination
        return FileOpNode(operation, source)


# ================================
# 5. SEMANTIC ANALYZER
# ================================
class SemanticAnalyzer:
    """Semantic validation of AST nodes"""
    
    VALID_SHOW_TARGETS = {"CPU", "MEMORY", "GPU", "TEMP", "DISK", "TASKS", "NETWORK", "PROCESS"}
    VALID_MODES = {"performance", "balanced", "power_saver", "powersaver"}

    def analyze(self, node: ASTNode):
        """Perform semantic analysis (Semantic Analysis stage)"""
        if isinstance(node, ShowNode):
            if node.target not in self.VALID_SHOW_TARGETS:
                raise ValueError(
                    f"Invalid SHOW target: {node.target}. "
                    f"Valid targets: {', '.join(sorted(self.VALID_SHOW_TARGETS))}"
                )
        
        elif isinstance(node, SetModeNode):
            if node.mode not in self.VALID_MODES:
                raise ValueError(
                    f"Invalid mode: {node.mode}. "
                    f"Valid modes: performance, balanced, power_saver"
                )
        
        elif isinstance(node, SetFanNode):
            if not 0 <= node.speed <= 100:
                raise ValueError(f"Fan speed must be between 0-100, got {node.speed}")
        
        elif isinstance(node, CalcNode):
            # Validate expression doesn't contain dangerous operations
            dangerous = ['import', 'exec', 'eval', 'compile', '__']
            for danger in dangerous:
                if danger in node.expression.lower():
                    raise ValueError(f"Expression contains forbidden operation: {danger}")


# ================================
# 6. EXECUTION ENGINE
# ================================
DB_FILE = "sparc.db"


class ExecutionEngine:
    """Enhanced execution engine with full SPArC capabilities"""
    
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
            except Exception as e:
                print(f"⚠️ NLP initialization failed: {e}")
                self.enable_nlp = False
        
        # Check system capabilities
        self.gpu_available = GPU_AVAILABLE
        self.wmi_available = WMI_AVAILABLE and platform.system() == "Windows"

    def _init_db(self):
        """Initialize database"""
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)) if os.path.dirname(self.db_path) else ".", exist_ok=True)

    def execute(self, query: str, use_nlp: bool = None) -> Dict[str, Any]:
        """
        Execute a query through the full FLA pipeline
        
        Args:
            query: Input query (natural language or formal DSL)
            use_nlp: Override default NLP setting
        
        Returns:
            Dict with execution results and metadata
        """
        if not query.strip():
            return {
                'result': "",
                'success': True,
                'method': 'empty',
                'original_query': query,
                'formal_query': "",
                'stages': {'lexical': 'skipped', 'syntax': 'skipped', 'semantic': 'skipped', 'execution': 'skipped'}
            }

        # Tracking for each FLA stage
        stages = {
            'lexical': 'not_started',
            'syntax': 'not_started',
            'semantic': 'not_started',
            'execution': 'not_started'
        }
        
        should_use_nlp = use_nlp if use_nlp is not None else self.enable_nlp
        formal_query = query
        nlp_info = {}
        
        try:
            # === NLP Processing (if enabled) ===
            if should_use_nlp and self.nlp_processor:
                if not query.upper().startswith(("SHOW ", "DB ", "CALC ", "SET ", "COPY ", "MOVE ", "DELETE ", "OPEN ")):
                    expanded_query = self.query_expander.expand_query(query)
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
                        nlp_info['nlp_used'] = False
                else:
                    nlp_info['nlp_used'] = False
            
            # === Stage 1: Lexical Analysis (Tokenization) ===
            stages['lexical'] = 'started'
            lexer = Lexer(formal_query)
            tokens = lexer.tokenize()
            stages['lexical'] = 'completed'
            
            # === Stage 2: Syntax Analysis (Parsing) ===
            stages['syntax'] = 'started'
            parser = Parser(tokens)
            ast = parser.parse()
            stages['syntax'] = 'completed'
            
            # === Stage 3: Semantic Analysis ===
            stages['semantic'] = 'started'
            self.semantic_analyzer.analyze(ast)
            stages['semantic'] = 'completed'
            
            # === Stage 4: Execution ===
            stages['execution'] = 'started'
            result = self._execute_ast(ast)
            stages['execution'] = 'completed'
            
            return {
                'result': result,
                'success': True,
                'method': 'formal_dsl',
                'original_query': query,
                'formal_query': formal_query,
                'ast_type': type(ast).__name__,
                'ast_node': ast,
                'tokens_count': len(tokens) - 1,  # Exclude EOF
                'stages': stages,
                **nlp_info
            }
            
        except SyntaxError as e:
            stages[self._get_failed_stage(stages)] = 'failed'
            error_msg = f"❌ Syntax Error: {str(e)}"
            
            # Try NLP fallback if not already tried
            if should_use_nlp and self.nlp_processor and not nlp_info.get('nlp_used', False):
                try:
                    expanded = self.query_expander.expand_query(query)
                    nlp_result = self.nlp_processor.process_query(expanded)
                    if nlp_result['command'] and nlp_result['confidence'] > 0.2:
                        return self.execute(nlp_result['command'], use_nlp=False)
                except:
                    pass
            
            return {
                'result': error_msg,
                'success': False,
                'method': 'error',
                'error_type': 'SyntaxError',
                'error_stage': self._get_failed_stage(stages),
                'original_query': query,
                'formal_query': formal_query,
                'error': str(e),
                'stages': stages,
                **nlp_info
            }
            
        except ValueError as e:
            stages['semantic'] = 'failed'
            error_msg = f"❌ Semantic Error: {str(e)}"
            return {
                'result': error_msg,
                'success': False,
                'method': 'error',
                'error_type': 'SemanticError',
                'error_stage': 'semantic',
                'original_query': query,
                'formal_query': formal_query,
                'error': str(e),
                'stages': stages,
                **nlp_info
            }
            
        except Exception as e:
            stages[self._get_failed_stage(stages)] = 'failed'
            error_msg = f"❌ Execution Error: {str(e)}"
            return {
                'result': error_msg,
                'success': False,
                'method': 'error',
                'error_type': type(e).__name__,
                'error_stage': self._get_failed_stage(stages),
                'original_query': query,
                'formal_query': formal_query,
                'error': str(e),
                'stages': stages,
                **nlp_info
            }

    def _get_failed_stage(self, stages: Dict[str, str]) -> str:
        """Determine which stage failed"""
        for stage, status in stages.items():
            if status == 'started':
                return stage
        return 'execution'

    def _execute_ast(self, node: ASTNode) -> str:
        """Execute an AST node"""
        if isinstance(node, ShowNode):
            return self._exec_show(node.target)
        elif isinstance(node, DBQueryNode):
            return self._exec_sql(node.raw_sql)
        elif isinstance(node, CalcNode):
            return self._exec_calc(node.expression)
        elif isinstance(node, SetModeNode):
            return self._exec_set_mode(node.mode)
        elif isinstance(node, SetFanNode):
            return self._exec_set_fan(node.speed)
        elif isinstance(node, FileOpNode):
            return self._exec_file_operation(node)
        else:
            raise RuntimeError(f"Unknown AST node type: {type(node)}")

    # === System Query Executors ===
    
    def _exec_show(self, target: str) -> str:
        """Execute SHOW commands"""
        if target == "CPU":
            return self._show_cpu()
        elif target == "MEMORY":
            return self._show_memory()
        elif target == "GPU":
            return self._show_gpu()
        elif target == "TEMP":
            return self._show_temperature()
        elif target == "DISK":
            return self._show_disk()
        elif target in ("TASKS", "PROCESS"):
            return self._show_tasks()
        elif target == "NETWORK":
            return self._show_network()
        else:
            return f"❌ Unknown target: {target}"

    def _show_cpu(self) -> str:
        """Show CPU information"""
        cpu_percent = psutil.cpu_percent(interval=0.5, percpu=False)
        cpu_count_logical = psutil.cpu_count(logical=True)
        cpu_count_physical = psutil.cpu_count(logical=False)
        cpu_freq = psutil.cpu_freq()
        
        lines = [
            "=== ⚡ CPU Information ===",
            f"Usage: {cpu_percent:.1f}%",
            f"Physical Cores: {cpu_count_physical}",
            f"Logical Cores: {cpu_count_logical}",
        ]
        
        if cpu_freq:
            lines.append(f"Frequency: {cpu_freq.current:.2f} MHz (Max: {cpu_freq.max:.2f} MHz)")
        
        # Per-core usage
        per_core = psutil.cpu_percent(interval=0.1, percpu=True)
        lines.append("\nPer-Core Usage:")
        for i, percent in enumerate(per_core):
            lines.append(f"  Core {i}: {percent:.1f}%")
        
        return "\n".join(lines)

    def _show_memory(self) -> str:
        """Show memory information"""
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        lines = [
            "=== 💾 Memory Information ===",
            f"Total RAM: {mem.total / (1024**3):.2f} GB",
            f"Available: {mem.available / (1024**3):.2f} GB",
            f"Used: {mem.used / (1024**3):.2f} GB ({mem.percent:.1f}%)",
            f"Free: {mem.free / (1024**3):.2f} GB",
            "",
            "=== Swap Memory ===",
            f"Total Swap: {swap.total / (1024**3):.2f} GB",
            f"Used Swap: {swap.used / (1024**3):.2f} GB ({swap.percent:.1f}%)",
            f"Free Swap: {swap.free / (1024**3):.2f} GB",
        ]
        
        return "\n".join(lines)

    def _show_gpu(self) -> str:
        """Show GPU information"""
        if not self.gpu_available:
            return "❌ GPU monitoring not available (install gputil: pip install gputil)"
        
        try:
            gpus = GPUtil.getGPUs()
            if not gpus:
                return "ℹ️ No NVIDIA GPUs detected"
            
            lines = ["=== 🎮 GPU Information ==="]
            for i, gpu in enumerate(gpus):
                lines.extend([
                    f"\nGPU {i}: {gpu.name}",
                    f"  Usage: {gpu.load * 100:.1f}%",
                    f"  Memory: {gpu.memoryUsed:.0f} MB / {gpu.memoryTotal:.0f} MB ({gpu.memoryUtil * 100:.1f}%)",
                    f"  Temperature: {gpu.temperature:.1f}°C",
                ])
            
            return "\n".join(lines)
        except Exception as e:
            return f"❌ Error reading GPU info: {e}"

    def _show_temperature(self) -> str:
        """Show system temperatures"""
        try:
            temps = psutil.sensors_temperatures()
            if not temps:
                return "ℹ️ Temperature sensors not available on this system"
            
            lines = ["=== 🌡️ System Temperatures ==="]
            for name, entries in temps.items():
                lines.append(f"\n{name}:")
                for entry in entries:
                    lines.append(f"  {entry.label or 'Sensor'}: {entry.current:.1f}°C (High: {entry.high}°C, Critical: {entry.critical}°C)")
            
            return "\n".join(lines)
        except AttributeError:
            return "❌ Temperature monitoring not supported on this platform"
        except Exception as e:
            return f"❌ Error reading temperatures: {e}"

    def _show_disk(self) -> str:
        """Show disk information"""
        lines = ["=== 💿 Disk Information ==="]
        
        partitions = psutil.disk_partitions()
        for partition in partitions:
            lines.append(f"\nDevice: {partition.device}")
            lines.append(f"  Mountpoint: {partition.mountpoint}")
            lines.append(f"  File system: {partition.fstype}")
            
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                lines.append(f"  Total: {usage.total / (1024**3):.2f} GB")
                lines.append(f"  Used: {usage.used / (1024**3):.2f} GB ({usage.percent:.1f}%)")
                lines.append(f"  Free: {usage.free / (1024**3):.2f} GB")
            except PermissionError:
                lines.append("  (Permission denied)")
        
        return "\n".join(lines)

    def _show_tasks(self) -> str:
        """Show running processes"""
        lines = ["=== 🖥️ Running Processes (Top 20 by Memory) ==="]
        
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'memory_percent', 'cpu_percent']):
            try:
                processes.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        # Sort by memory usage
        processes.sort(key=lambda p: p['memory_percent'] or 0, reverse=True)
        
        lines.append(f"{'PID':<8} {'Memory %':<10} {'CPU %':<10} {'Name'}")
        lines.append("-" * 60)
        
        for proc in processes[:20]:
            lines.append(
                f"{proc['pid']:<8} "
                f"{proc['memory_percent'] or 0:<10.2f} "
                f"{proc['cpu_percent'] or 0:<10.2f} "
                f"{proc['name']}"
            )
        
        return "\n".join(lines)

    def _show_network(self) -> str:
        """Show network information"""
        lines = ["=== 🌐 Network Information ==="]
        
        # Network interfaces
        if_addrs = psutil.net_if_addrs()
        lines.append("\nNetwork Interfaces:")
        for interface_name, addresses in if_addrs.items():
            lines.append(f"\n{interface_name}:")
            for addr in addresses:
                if addr.family == 2:  # AF_INET (IPv4)
                    lines.append(f"  IPv4: {addr.address}")
                elif addr.family == 23:  # AF_INET6 (IPv6)
                    lines.append(f"  IPv6: {addr.address}")
                elif addr.family == -1:  # AF_LINK (MAC)
                    lines.append(f"  MAC: {addr.address}")
        
        # Network statistics
        net_io = psutil.net_io_counters()
        lines.append("\n=== Network Statistics ===")
        lines.append(f"Bytes Sent: {net_io.bytes_sent / (1024**2):.2f} MB")
        lines.append(f"Bytes Received: {net_io.bytes_recv / (1024**2):.2f} MB")
        lines.append(f"Packets Sent: {net_io.packets_sent}")
        lines.append(f"Packets Received: {net_io.packets_recv}")
        
        return "\n".join(lines)

    # === Performance Control Executors ===
    
    def _exec_set_mode(self, mode: str) -> str:
        """Execute SET MODE command"""
        if not self.wmi_available and platform.system() == "Windows":
            return self._set_mode_windows(mode)
        elif platform.system() == "Linux":
            return self._set_mode_linux(mode)
        elif platform.system() == "Darwin":
            return "ℹ️ Power mode control not implemented for macOS"
        else:
            return f"❌ Power mode control not supported on {platform.system()}"

    def _set_mode_windows(self, mode: str) -> str:
        """Set power mode on Windows"""
        # Map mode to Windows power plan GUIDs
        power_plans = {
            "performance": "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c",
            "balanced": "381b4222-f694-41f0-9685-ff5bb260df2e",
            "power_saver": "a1841308-3541-4fab-bc81-f71556f20b4a",
            "powersaver": "a1841308-3541-4fab-bc81-f71556f20b4a"
        }
        
        guid = power_plans.get(mode)
        if not guid:
            return f"❌ Unknown power mode: {mode}"
        
        try:
            # Use powercfg to set active power plan
            result = subprocess.run(
                ["powercfg", "/setactive", guid],
                capture_output=True,
                text=True,
                check=True
            )
            return f"✅ Power mode set to: {mode.replace('_', ' ').title()}"
        except subprocess.CalledProcessError as e:
            return f"❌ Failed to set power mode: {e.stderr}"
        except FileNotFoundError:
            return "❌ powercfg command not found (Windows required)"

    def _set_mode_linux(self, mode: str) -> str:
        """Set power mode on Linux"""
        # This is a placeholder - actual implementation would depend on the system
        governors = {
            "performance": "performance",
            "balanced": "ondemand",
            "power_saver": "powersave",
            "powersaver": "powersave"
        }
        
        governor = governors.get(mode)
        if not governor:
            return f"❌ Unknown power mode: {mode}"
        
        try:
            # Try to set CPU governor (requires root)
            result = subprocess.run(
                ["cpupower", "frequency-set", "-g", governor],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                return f"✅ CPU governor set to: {governor}"
            else:
                return f"ℹ️ CPU governor setting requires root privileges\nTry: sudo cpupower frequency-set -g {governor}"
        except FileNotFoundError:
            return "ℹ️ cpupower not found. Install linux-tools or cpufrequtils package"

    def _exec_set_fan(self, speed: int) -> str:
        """Execute SET FAN command"""
        # Fan control is highly hardware-specific
        # This is a simulation/placeholder
        return f"ℹ️ Fan control is hardware-specific and not yet implemented\nRequested fan speed: {speed}%\n\nNote: Use vendor-specific tools like:\n- MSI Afterburner\n- EVGA Precision\n- hwinfo\n- lm-sensors (Linux)"

    # === Arithmetic Executor ===
    
    def _exec_calc(self, expression: str) -> str:
        """Execute arithmetic calculation"""
        try:
            # Remove any leading/trailing whitespace
            expression = expression.strip()
            
            # Create a safe evaluation environment
            # Only allow basic math operations
            safe_dict = {
                '__builtins__': {},
                'abs': abs,
                'min': min,
                'max': max,
                'pow': pow,
                'round': round,
            }
            
            # Parse and evaluate the expression safely
            parsed = python_ast.parse(expression, mode='eval')
            
            # Validate that only safe operations are used
            for node in python_ast.walk(parsed):
                if isinstance(node, (python_ast.Import, python_ast.ImportFrom, python_ast.Call)):
                    if isinstance(node, python_ast.Call) and not isinstance(node.func, python_ast.Name):
                        raise ValueError("Invalid function call in expression")
            
            result = eval(compile(parsed, '<string>', 'eval'), safe_dict)
            
            return f"=== 🔢 Calculation Result ===\nExpression: {expression}\nResult: {result}"
        
        except SyntaxError as e:
            return f"❌ Invalid expression syntax: {e}"
        except ZeroDivisionError:
            return "❌ Division by zero error"
        except Exception as e:
            return f"❌ Calculation error: {e}"

    # === File Operation Executors ===
    
    def _exec_file_operation(self, node: FileOpNode) -> str:
        """Execute file operations"""
        operation = node.operation
        source = node.source
        destination = node.destination
        
        try:
            if operation == "COPY":
                return self._file_copy(source, destination)
            elif operation == "MOVE":
                return self._file_move(source, destination)
            elif operation == "DELETE":
                return self._file_delete(source)
            elif operation == "OPEN":
                return self._file_open(source)
            else:
                return f"❌ Unknown file operation: {operation}"
        except Exception as e:
            return f"❌ File operation error: {e}"

    def _file_copy(self, source: str, destination: str) -> str:
        """Copy file or directory"""
        if not os.path.exists(source):
            raise FileNotFoundError(f"Source not found: {source}")
        
        if os.path.isfile(source):
            shutil.copy2(source, destination)
            return f"✅ File copied successfully\nFrom: {source}\nTo: {destination}"
        elif os.path.isdir(source):
            shutil.copytree(source, destination)
            return f"✅ Directory copied successfully\nFrom: {source}\nTo: {destination}"
        else:
            raise ValueError(f"Source is neither file nor directory: {source}")

    def _file_move(self, source: str, destination: str) -> str:
        """Move file or directory"""
        if not os.path.exists(source):
            raise FileNotFoundError(f"Source not found: {source}")
        
        shutil.move(source, destination)
        return f"✅ Moved successfully\nFrom: {source}\nTo: {destination}"

    def _file_delete(self, path: str) -> str:
        """Delete file or directory"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path not found: {path}")
        
        if os.path.isfile(path):
            os.remove(path)
            return f"✅ File deleted: {path}"
        elif os.path.isdir(path):
            shutil.rmtree(path)
            return f"✅ Directory deleted: {path}"
        else:
            raise ValueError(f"Path is neither file nor directory: {path}")

    def _file_open(self, path: str) -> str:
        """Open file with default application"""
        if not os.path.exists(path):
            # Check if it's a program name (like 'notepad')
            try:
                if platform.system() == "Windows":
                    subprocess.Popen([path], shell=True)
                    return f"✅ Opened: {path}"
                elif platform.system() == "Darwin":
                    subprocess.Popen(["open", path])
                    return f"✅ Opened: {path}"
                elif platform.system() == "Linux":
                    subprocess.Popen(["xdg-open", path])
                    return f"✅ Opened: {path}"
            except Exception as e:
                raise FileNotFoundError(f"Cannot open: {path} ({e})")
        
        # Open existing file
        try:
            if platform.system() == "Windows":
                os.startfile(path)
            elif platform.system() == "Darwin":
                subprocess.Popen(["open", path])
            elif platform.system() == "Linux":
                subprocess.Popen(["xdg-open", path])
            return f"✅ Opened: {path}"
        except Exception as e:
            return f"❌ Failed to open: {e}"

    # === Database Executor ===
    
    def _exec_sql(self, sql: str) -> str:
        """Execute SQL command"""
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            cur.execute(sql)
            
            if sql.strip().upper().startswith(("SELECT", "PRAGMA")):
                rows = cur.fetchall()
                conn.close()
                
                if not rows:
                    return "✅ Query executed (no results)"
                
                # Format results nicely
                lines = ["=== 🗄️ Query Results ==="]
                for row in rows:
                    lines.append(str(row))
                
                return "\n".join(lines)
            else:
                conn.commit()
                rows_affected = cur.rowcount
                conn.close()
                return f"✅ Query executed successfully ({rows_affected} rows affected)"
        
        except sqlite3.Error as e:
            return f"❌ SQL Error: {e}"
        except Exception as e:
            return f"❌ Execution Error: {e}"

    def close(self):
        """Cleanup resources"""
        pass
