# ui.py
import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QLineEdit, QPushButton, QTabWidget, QLabel,
    QCheckBox, QProgressBar, QFrame, QSplitter, QListWidget, QGroupBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

class MainWindow(QMainWindow):
    def __init__(self, engine):
        super().__init__()
        self.engine = engine
        self.query_history = []
        self.history_index = -1
        
        self.setWindowTitle("SPARC — Custom Command Line with DSL")
        self.resize(1400, 900)
        self.setStyleSheet("""
            QMainWindow { 
                background-color: #1e1e2e; 
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QLabel { 
                color: #cdd6f4; 
                font-size: 14px; 
            }
            QLineEdit { 
                background: #313244; 
                color: #cdd6f4; 
                border: 2px solid #585b70; 
                border-radius: 6px;
                padding: 10px; 
                font-family: 'Consolas', monospace; 
                font-size: 14px;
                selection-background-color: #74c7ec;
            }
            QLineEdit:focus {
                border-color: #a6e3a1;
            }
            QPushButton {
                background: #a6e3a1;
                color: #1e1e2e;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-weight: bold;
                font-size: 14px;
                min-height: 20px;
            }
            QPushButton:hover { 
                background: #94e2d5; 
            }
            QPushButton:pressed {
                background: #89dceb;
            }
            QPushButton:disabled {
                background: #6c7086;
                color: #9399b2;
            }
            QTextEdit {
                background: #1e1e2e;
                color: #cdd6f4;
                font-family: 'Consolas', monospace;
                font-size: 13px;
                border: 1px solid #585b70;
                border-radius: 6px;
                padding: 8px;
                line-height: 1.4;
            }
            QTabWidget::pane { 
                border: 1px solid #585b70; 
                border-radius: 6px;
                background: #1e1e2e;
            }
            QTabBar::tab { 
                background: #313244; 
                color: #cdd6f4; 
                padding: 12px 20px; 
                margin-right: 2px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                font-size: 13px;
            }
            QTabBar::tab:selected { 
                background: #a6e3a1; 
                color: #1e1e2e;
                font-weight: bold;
            }
            QTabBar::tab:hover:!selected {
                background: #45475a;
            }
            QCheckBox {
                color: #cdd6f4;
                font-size: 13px;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border-radius: 3px;
                border: 2px solid #585b70;
                background: #313244;
            }
            QCheckBox::indicator:checked {
                background: #a6e3a1;
                border-color: #a6e3a1;
            }
            QGroupBox {
                color: #cdd6f4;
                border: 1px solid #585b70;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #f5c2e7;
            }
            QListWidget {
                background: #313244;
                color: #cdd6f4;
                border: 1px solid #585b70;
                border-radius: 6px;
                font-family: 'Consolas', monospace;
                font-size: 12px;
            }
            QListWidget::item {
                padding: 5px;
                border-bottom: 1px solid #45475a;
            }
            QListWidget::item:selected {
                background: #a6e3a1;
                color: #1e1e2e;
            }
            QProgressBar {
                border: 1px solid #585b70;
                border-radius: 6px;
                background: #313244;
                text-align: center;
                color: #cdd6f4;
            }
            QProgressBar::chunk {
                background: #a6e3a1;
                border-radius: 5px;
            }
            QFrame {
                background: #1e1e2e;
                border: 1px solid #585b70;
                border-radius: 6px;
            }
        """)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # Header
        header_layout = QVBoxLayout()
        title = QLabel("⚡ SPARC — Custom Command Line with DSL")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 24px; font-weight: bold; margin: 15px; color: #f5c2e7;")
        
        subtitle = QLabel("Custom Domain-Specific Language + Natural Language Processing")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("font-size: 14px; color: #9399b2; margin-bottom: 10px;")
        
        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)
        main_layout.addLayout(header_layout)

        # Controls section
        controls_frame = QFrame()
        controls_layout = QVBoxLayout(controls_frame)
        
        # NLP settings
        settings_group = QGroupBox("⚙️ Processing Settings")
        settings_layout = QHBoxLayout(settings_group)
        
        self.nlp_checkbox = QCheckBox("Enable NLP Processing")
        self.nlp_checkbox.setChecked(self.engine.enable_nlp)
        self.nlp_checkbox.stateChanged.connect(self._toggle_nlp)
        
        settings_layout.addWidget(self.nlp_checkbox)
        settings_layout.addStretch()
        
        controls_layout.addWidget(settings_group)

        # Input section
        input_group = QGroupBox("💬 Query Input")
        input_layout = QVBoxLayout(input_group)
        
        # Input line without autocomplete
        self.input_line = QLineEdit()
        self.input_line.setPlaceholderText("Try: 'show cpu usage' or 'list all tables' or formal 'SHOW CPU'")
        self.input_line.returnPressed.connect(self.run_query)
        
        # Button row
        button_layout = QHBoxLayout()
        run_btn = QPushButton("🚀 Execute")
        run_btn.clicked.connect(self.run_query)
        
        clear_btn = QPushButton("🗑️ Clear")
        clear_btn.clicked.connect(self._clear_outputs)
        
        examples_btn = QPushButton("💡 Examples")
        examples_btn.clicked.connect(self._show_examples)
        
        button_layout.addWidget(run_btn)
        button_layout.addWidget(clear_btn)
        button_layout.addWidget(examples_btn)
        button_layout.addStretch()
        
        input_layout.addWidget(self.input_line)
        input_layout.addLayout(button_layout)
        controls_layout.addWidget(input_group)
        
        main_layout.addWidget(controls_frame)

        # Create splitter for main content
        splitter = QSplitter(Qt.Horizontal)
        
        # Left side - outputs
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        self.tabs = QTabWidget()
        self.system_output = QTextEdit()
        self.db_output = QTextEdit()
        self.debug_output = QTextEdit()
        
        self.system_output.setReadOnly(True)
        self.db_output.setReadOnly(True)
        self.debug_output.setReadOnly(True)
        
        self.tabs.addTab(self.system_output, "🖥️ System")
        self.tabs.addTab(self.db_output, "🗄️ Database") 
        self.tabs.addTab(self.debug_output, "🔍 Debug")
        left_layout.addWidget(self.tabs)
        
        # Right side - history and examples
        right_widget = QWidget()
        right_widget.setMaximumWidth(300)
        right_layout = QVBoxLayout(right_widget)
        
        # Query history
        history_group = QGroupBox("📝 Query History")
        history_layout = QVBoxLayout(history_group)
        self.history_list = QListWidget()
        self.history_list.itemDoubleClicked.connect(self._load_from_history)
        history_layout.addWidget(self.history_list)
        right_layout.addWidget(history_group)
        
        # Quick examples
        examples_group = QGroupBox("🎯 Quick Examples")
        examples_layout = QVBoxLayout(examples_group)
        
        example_queries = [
            "show cpu usage",
            "show memory status", 
            "list running processes",
            "show all tables",
            "DB CREATE TABLE test (id INTEGER);",
            "DB SELECT * FROM sqlite_master;"
        ]
        
        for query in example_queries:
            btn = QPushButton(query)
            btn.setStyleSheet("text-align: left; padding: 5px; font-size: 11px;")
            btn.clicked.connect(lambda checked, q=query: self._set_query(q))
            examples_layout.addWidget(btn)
        
        right_layout.addWidget(examples_group)
        right_layout.addStretch()
        
        # Add to splitter
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 3)  # Left side gets more space
        splitter.setStretchFactor(1, 1)
        
        main_layout.addWidget(splitter)

        # Status bar
        status_layout = QHBoxLayout()
        self.status = QLabel("🟢 Ready — Try natural language or formal DSL syntax")
        self.status.setStyleSheet("color: #a6e3a1; padding: 8px; font-weight: bold;")
        
        self.processing_indicator = QProgressBar()
        self.processing_indicator.setVisible(False)
        self.processing_indicator.setMaximumHeight(6)
        
        status_layout.addWidget(self.status)
        status_layout.addStretch()
        status_layout.addWidget(self.processing_indicator)
        
        main_layout.addLayout(status_layout)

        self._show_intro()

    def _show_intro(self):
        self.system_output.setPlainText("🖥️ System output will appear here...\nTry commands like: SHOW CPU, SHOW MEMORY, SHOW TASKS\n")
        self.db_output.setPlainText("🗄️ Database output will appear here...\nTry commands like: DB SELECT * FROM sqlite_master;\n")
        self.debug_output.setPlainText("🔍 Debug information will appear here...\n")

    def _toggle_nlp(self, state):
        """Toggle NLP processing on/off."""
        self.engine.enable_nlp = state == Qt.Checked
        status_text = "🟢 NLP Enabled" if self.engine.enable_nlp else "🟡 NLP Disabled (Formal DSL only)"
        self.status.setText(status_text)
        
        # Update placeholder text
        if self.engine.enable_nlp:
            self.input_line.setPlaceholderText("Try: 'show cpu usage' or 'list all tables' or formal 'SHOW CPU'")
        else:
            self.input_line.setPlaceholderText("Enter formal query: SHOW CPU | DB SELECT * FROM table")



    def _clear_outputs(self):
        """Clear all output windows."""
        self.system_output.clear()
        self.db_output.clear()
        self.debug_output.clear()
        self._show_intro()
        self.status.setText("🗑️ Outputs cleared")

    def _show_examples(self):
        """Show example queries in the debug tab."""
        examples = (
            "🎯 Natural Language Examples:\n\n"
            "System Monitoring:\n"
            " • 'show me the cpu usage'\n"
            " • 'what is the processor utilization'\n"
            " • 'how much memory is being used'\n"
            " • 'display ram status'\n"
            " • 'list all running processes'\n"
            " • 'what programs are active'\n\n"
            "Database Queries:\n"
            " • 'show all tables'\n"
            " • 'what tables exist in the database'\n"
            " • 'select everything from users'\n"
            " • 'get all records from logs'\n"
            " • 'create a table called customers'\n"
            " • 'insert data into products'\n\n"
            "Formal DSL Commands:\n"
            " • SHOW CPU\n"
            " • SHOW MEMORY\n"
            " • SHOW TASKS\n"
            " • DB SELECT name FROM sqlite_master WHERE type='table';\n"
            " • DB CREATE TABLE users (id INTEGER, name TEXT);\n"
            " • DB INSERT INTO users VALUES (1, 'Alice');\n"
            " • DB SELECT * FROM users;\n\n"
            "🔍 The system automatically detects whether you're using\n"
            "natural language or formal syntax and processes accordingly!\n"
        )
        self.debug_output.setPlainText(examples)
        self.tabs.setCurrentIndex(2)  # Switch to debug tab

    def _set_query(self, query):
        """Set the input field to a specific query."""
        self.input_line.setText(query)
        self.input_line.setFocus()

    def _load_from_history(self, item):
        """Load a query from history."""
        query = item.text()
        self.input_line.setText(query)
        self.input_line.setFocus()

    def _add_to_history(self, query):
        """Add a query to the history list."""
        if query and query not in self.query_history:
            self.query_history.append(query)
            self.history_list.addItem(query)
            
            # Limit history to 20 items
            if len(self.query_history) > 20:
                self.query_history.pop(0)
                self.history_list.takeItem(0)

    def run_query(self):
        query = self.input_line.text().strip()
        if not query:
            return

        # Add to history
        self._add_to_history(query)
        
        # Show progress
        self.processing_indicator.setVisible(True)
        self.processing_indicator.setRange(0, 0)  # Indeterminate progress
        self.status.setText(f"🔄 Processing: {query[:50]}{'...' if len(query) > 50 else ''}")
        QApplication.processEvents()

        try:
            # Execute query with enhanced engine
            result_dict = self.engine.execute(query)
            result = result_dict['result']
            
            # Clear input
            self.input_line.clear()
            
            # Add debug information
            debug_info = self._format_debug_info(result_dict)
            self.debug_output.setPlainText(debug_info)
            
            # Route to correct tab based on content and type
            if result_dict.get('ast_type') == 'ShowNode' or any(keyword in result for keyword in ["Running Tasks", "CPU Usage", "Memory Usage"]):
                self.system_output.setPlainText(f"📊 Query: {query}\n\n{result}")
                self.tabs.setCurrentIndex(0)
            elif result_dict.get('ast_type') == 'DBQueryNode' or "DB" in query.upper():
                self.db_output.setPlainText(f"🗄️ Query: {query}\n\n{result}")
                self.tabs.setCurrentIndex(1)
            else:
                # Default to system tab for other outputs
                self.system_output.setPlainText(f"❓ Query: {query}\n\n{result}")
                self.tabs.setCurrentIndex(0)

            # Update status based on success
            if result_dict['success']:
                confidence_info = ""
                if 'nlp_confidence' in result_dict:
                    confidence_info = f" (NLP: {result_dict['nlp_confidence']:.1%})"
                self.status.setText(f"✅ Done - {result_dict['method']}{confidence_info}")
            else:
                self.status.setText(f"❌ Error - {result_dict.get('error', 'Unknown error')}")
                
        except Exception as e:
            self.status.setText(f"❌ Error: {str(e)}")
            self.debug_output.setPlainText(f"🚨 Exception: {str(e)}\nQuery: {query}\n\nFull Error Details:\n{str(e)}")
            self.system_output.setPlainText(f"❌ Error executing query: {query}\n\n{str(e)}")
            self.tabs.setCurrentIndex(0)  # Show system tab for errors
            
        finally:
            # Hide progress indicator
            self.processing_indicator.setVisible(False)

    def _format_debug_info(self, result_dict):
        """Format debug information for display."""
        timestamp = __import__('datetime').datetime.now().strftime("%H:%M:%S")
        
        debug_lines = [
            f"⏰ {timestamp} - Query Debug Info",
            f"📝 Original Query: {result_dict['original_query']}",
            f"🔧 Formal Query: {result_dict['formal_query']}",
            f"🎯 Processing Method: {result_dict['method']}",
            f"✅ Success: {result_dict['success']}",
            ""
        ]
        
        if 'nlp_used' in result_dict:
            debug_lines.extend([
                "🧠 NLP Processing:",
                f"   Used: {result_dict['nlp_used']}",
                f"   Method: {result_dict.get('nlp_method', 'N/A')}",
                f"   Confidence: {result_dict.get('nlp_confidence', 0):.1%}",
            ])
            
            if 'intent' in result_dict:
                debug_lines.append(f"   Intent: {result_dict['intent']}")
            
            if 'entities' in result_dict and result_dict['entities']:
                entities_str = ', '.join([f"{k}: {v}" for k, v in result_dict['entities'].items() if v])
                if entities_str:
                    debug_lines.append(f"   Entities: {entities_str}")
            debug_lines.append("")
        
        if 'ast_type' in result_dict:
            debug_lines.extend([
                "🌳 AST Information:",
                f"   Type: {result_dict['ast_type']}"
            ])
            
            if result_dict['ast_type'] == 'ShowNode' and 'target' in result_dict:
                debug_lines.append(f"   Target: {result_dict['target']}")
            elif result_dict['ast_type'] == 'DBQueryNode' and 'sql' in result_dict:
                debug_lines.append(f"   SQL: {result_dict['sql']}")
            debug_lines.append("")
        
        if 'error' in result_dict:
            debug_lines.extend([
                "❌ Error Information:",
                f"   {result_dict['error']}"
            ])
        
        return "\n".join(debug_lines)

    def closeEvent(self, event):
        self.engine.close()
        event.accept()