# Project Abstract

## SPARC: Custom Command Line with Domain-Specific Language

This project presents the design and implementation of an advanced domain-specific query language (DSL) that seamlessly integrates Natural Language Processing (NLP) with Formal Language Automata (FLA) theory to create a unified interface for both real-time system monitoring and persistent database operations. The system enables users to interact using either natural language queries ("show me the cpu usage") or formal DSL syntax ("SHOW CPU"), automatically processing both through a sophisticated multi-layered architecture.

### Natural Language Processing Layer

The NLP component employs multiple complementary techniques for robust natural language understanding:

- **Intent Classification**: Pattern-based regex matching identifies user intent categories (system monitoring vs. database operations)
- **Entity Extraction**: Sophisticated parsing extracts relevant entities including table names, column specifications, numeric values, and query parameters
- **TF-IDF Similarity Matching**: Machine learning-based similarity scoring using Term Frequency-Inverse Document Frequency vectorization with cosine similarity measures to match user queries against a trained corpus of natural language examples
- **Query Expansion**: Intelligent expansion of abbreviated queries ("cpu" → "show cpu usage") through contextual mapping
- **Confidence Scoring**: Multi-method confidence evaluation ensuring high-quality natural language to formal DSL translation

### Formal Language Automata Architecture

The formal language processing follows rigorous compiler front-end principles:

- **DFA-based Lexical Analysis**: Deterministic Finite Automaton-inspired tokenizer recognizing keywords, identifiers, operators, literals, and punctuation according to regular language patterns
- **Context-Free Grammar (CFG)**: Formally specified grammar defining valid query structures with unambiguous production rules
- **Recursive Descent Parser**: Top-down parser constructing Abstract Syntax Trees (AST) representing hierarchical query structure
- **Semantic Analysis**: Type checking and referential integrity validation ensuring logical consistency before execution
- **Comprehensive Error Handling**: Detailed syntax and semantic error reporting with line/column precision

### Dual Execution Engine

The execution layer supports heterogeneous query types:

- **System Monitoring**: Real-time system resource queries via psutil library providing CPU utilization, memory statistics, and process enumeration across Windows, macOS, and Linux platforms
- **Database Operations**: Full SQLite integration supporting CREATE, INSERT, SELECT, UPDATE, DELETE operations with ACID compliance and persistent storage in `sparc.db`
- **Query Routing**: Intelligent routing based on AST node types (ShowNode for system queries, DBQueryNode for database operations)

### Enhanced User Interface

A modern PyQt5-based interface provides:

- **Multi-tab Architecture**: Separate visualization for system metrics, database results, and debug information
- **Intelligent Auto-completion**: Context-aware query suggestions based on partial input and historical patterns
- **Query History Management**: Persistent query history with quick access and replay functionality
- **Real-time Processing Indicators**: Visual feedback during query processing with confidence scores and method identification
- **Responsive Design**: High-DPI support with modern dark theme inspired by the Catppuccin color palette

### Technical Innovation

This implementation demonstrates several novel integrations:

1. **Hybrid Processing Pipeline**: Seamless fallback between NLP and formal parsing, allowing users to mix natural language and formal syntax within the same session
2. **Multi-method Confidence Evaluation**: Combination of pattern matching, similarity scoring, and semantic validation for robust query interpretation
3. **Unified Result Presentation**: Consistent formatting and visualization regardless of input method (natural language vs. formal DSL)
4. **Extensible Architecture**: Modular design supporting easy addition of new NLP training data, DSL grammar rules, and execution targets

### Academic Contributions

The project provides comprehensive demonstration of:

- **Formal Language Theory**: Regular expressions, context-free grammars, and parsing algorithms
- **Natural Language Processing**: Intent classification, entity extraction, and similarity matching
- **Compiler Construction**: Complete front-end implementation from lexical analysis through code generation
- **Software Architecture**: Clean separation of concerns with modular, extensible design
- **Human-Computer Interaction**: Intuitive interface design bridging technical precision with user accessibility

This system serves as both a practical tool for system administration and database management, and a comprehensive educational platform demonstrating the integration of theoretical computer science concepts with modern software engineering practices. The dual-mode operation (NLP + FLA) showcases how classical formal methods can be enhanced with machine learning techniques to create more intuitive and accessible computing interfaces.

**Keywords**: Domain-Specific Languages, Natural Language Processing, Formal Language Automata, Compiler Design, System Monitoring, Database Query Processing, Human-Computer Interaction