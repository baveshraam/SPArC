# SPARC - Custom Command Line with DSL

**SPARC** is a custom command line interface featuring a specialized Domain-Specific Language (DSL) designed for efficient system monitoring and database operations. It combines formal language processing with natural language understanding to provide flexible query capabilities.

## 🚀 Quick Start

```bash
# Install dependencies
pip install PyQt5 psutil nltk scikit-learn

# Run the application
python main.py
```

## 📁 Core Files

- `main.py` - Application launcher
- `ui.py` - User interface 
- `dsl_core.py` - Core DSL engine with FLA processing
- `nlp_processor.py` - Natural language processing
- `requirements.txt` - Dependencies
- `test_clean.py` - Basic functionality test

## 🎯 Commands

### System Commands
- `SHOW CPU` - Display CPU usage
- `SHOW MEMORY` - Display memory stats  
- `SHOW TASKS` - List running processes

### Database Commands
- `DB SELECT * FROM sqlite_master;` - Show tables
- `DB CREATE TABLE test (id INTEGER);` - Create table
- `DB INSERT INTO test VALUES (1);` - Insert data
- `DB SELECT * FROM test;` - Query data

### Natural Language (when NLP enabled)
- "show cpu usage"
- "how much memory is used"
- "list running processes"
- "show all tables"

## ✨ Features

- **Custom DSL**: Specialized Domain-Specific Language for system and database operations
- **Dual Interface**: Support both formal DSL syntax and natural language queries
- **Clean Output**: Each command result replaces previous output
- **Tabbed Interface**: System/Database/Debug tabs with smart routing
- **Autocomplete**: Intelligent suggestions as you type
- **Error Handling**: Clear error messages and graceful degradation
- **Cross-platform**: Works on Windows, macOS, and Linux

## 🔧 UI Behavior

- **System Tab**: Shows SHOW commands (CPU, MEMORY, TASKS)
- **Database Tab**: Shows DB commands and SQL results
- **Debug Tab**: Shows processing details and metadata
- **Clear Button**: Clears all tabs and resets to clean state