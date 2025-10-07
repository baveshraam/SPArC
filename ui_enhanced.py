# ui_enhanced.py
"""
Enhanced Modern UI for SPArC DSL System
Features: Modern design, smooth animations, better UX, visual feedback
"""

import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QLineEdit, QPushButton, QTabWidget, QLabel,
    QCheckBox, QProgressBar, QFrame, QSplitter, QListWidget, 
    QGroupBox, QScrollArea, QStatusBar, QToolButton, QGraphicsDropShadowEffect
)
from PyQt5.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, pyqtProperty, QPoint
from PyQt5.QtGui import QFont, QPalette, QColor, QLinearGradient, QIcon, QPainter, QBrush

class ModernLineEdit(QLineEdit):
    """Custom line edit with modern styling and animations"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._focused = False
        self.setup_effects()
    
    def setup_effects(self):
        # Add shadow effect
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)
        shadow.setColor(QColor(0, 0, 0, 80))
        shadow.setOffset(0, 2)
        self.setGraphicsEffect(shadow)
    
    def focusInEvent(self, event):
        self._focused = True
        self.update_style()
        super().focusInEvent(event)
    
    def focusOutEvent(self, event):
        self._focused = False
        self.update_style()
        super().focusOutEvent(event)
    
    def update_style(self):
        if self._focused:
            self.setStyleSheet(self.styleSheet() + """
                border: 2px solid #89b4fa;
                background: #1e1e2e;
            """)
        else:
            self.setStyleSheet(self.styleSheet().replace("border: 2px solid #89b4fa;", "border: 2px solid #45475a;"))


class AnimatedButton(QPushButton):
    """Button with hover animations"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setup_effects()
    
    def setup_effects(self):
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(166, 227, 161, 100))
        shadow.setOffset(0, 4)
        self.setGraphicsEffect(shadow)


class MainWindow(QMainWindow):
    def __init__(self, engine):
        super().__init__()
        self.engine = engine
        self.query_history = []
        self.history_index = -1
        
        self.setWindowTitle("⚡ SPArC DSL - Modern Edition")
        self.resize(1600, 1000)
        
        # Apply modern dark theme with gradients
        self.apply_modern_theme()
        
        # Setup UI
        self.setup_ui()
        
        # Start animations
        self.start_intro_animation()
    
    def apply_modern_theme(self):
        """Apply clean modern dark theme with proper contrast"""
        self.setStyleSheet("""
            QMainWindow { 
                background: #1a1b26;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            
            /* Labels */
            QLabel { 
                color: #c0caf5;
                font-size: 13px;
                border: none;
                background: transparent;
            }
            
            /* Modern Input Field */
            QLineEdit { 
                background: #24283b;
                color: #c0caf5;
                border: 2px solid #414868;
                border-radius: 8px;
                padding: 12px 16px;
                font-family: 'Consolas', monospace;
                font-size: 14px;
                selection-background-color: #7aa2f7;
                selection-color: #1a1b26;
            }
            QLineEdit:focus {
                border: 2px solid #7aa2f7;
                background: #1f2335;
            }
            QLineEdit:hover {
                border-color: #565f89;
            }
            
            /* Modern Buttons */
            QPushButton {
                background: #7aa2f7;
                color: #1a1b26;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-weight: 600;
                font-size: 13px;
                min-height: 16px;
            }
            QPushButton:hover { 
                background: #89b4fa;
            }
            QPushButton:pressed {
                background: #6a92e7;
            }
            QPushButton:disabled {
                background: #414868;
                color: #565f89;
            }
            
            /* Modern Text Areas */
            QTextEdit {
                background: #1f2335;
                color: #c0caf5;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 13px;
                border: 1px solid #414868;
                border-radius: 8px;
                padding: 10px;
                selection-background-color: #7aa2f7;
                selection-color: #1a1b26;
            }
            
            /* Modern Tabs */
            QTabWidget::pane { 
                border: 1px solid #414868;
                border-radius: 8px;
                background: #1a1b26;
                top: -1px;
            }
            QTabBar::tab { 
                background: #24283b;
                color: #787c99;
                padding: 10px 20px;
                margin-right: 2px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                font-size: 12px;
                font-weight: 500;
                min-width: 80px;
            }
            QTabBar::tab:selected { 
                background: #7aa2f7;
                color: #1a1b26;
                font-weight: 700;
            }
            QTabBar::tab:hover:!selected {
                background: #414868;
                color: #c0caf5;
            }
            
            /* Modern Checkbox */
            QCheckBox {
                color: #c0caf5;
                font-size: 13px;
                spacing: 8px;
                padding: 4px;
                background: transparent;
            }
            QCheckBox::indicator {
                width: 20px;
                height: 20px;
                border-radius: 4px;
                border: 2px solid #414868;
                background: #24283b;
            }
            QCheckBox::indicator:checked {
                background: #7aa2f7;
                border-color: #7aa2f7;
            }
            QCheckBox::indicator:hover {
                border-color: #7aa2f7;
            }
            
            /* Modern Group Box */
            QGroupBox {
                color: #bb9af7;
                border: 1px solid #414868;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 12px;
                font-weight: 600;
                font-size: 13px;
                background: transparent;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
                color: #bb9af7;
                background: #1a1b26;
            }
            
            /* Modern List Widget */
            QListWidget {
                background: #1f2335;
                color: #c0caf5;
                border: 1px solid #414868;
                border-radius: 6px;
                font-family: 'Consolas', monospace;
                font-size: 12px;
                padding: 4px;
            }
            QListWidget::item {
                padding: 8px 10px;
                border-radius: 4px;
                margin: 1px;
                color: #c0caf5;
            }
            QListWidget::item:hover {
                background: #414868;
            }
            QListWidget::item:selected {
                background: #7aa2f7;
                color: #1a1b26;
                font-weight: 600;
            }
            
            /* Modern Progress Bar */
            QProgressBar {
                border: none;
                border-radius: 6px;
                background: #24283b;
                text-align: center;
                color: #c0caf5;
                font-weight: 600;
                height: 6px;
            }
            QProgressBar::chunk {
                background: #7aa2f7;
                border-radius: 5px;
            }
            
            /* Modern Frames */
            QFrame {
                background: transparent;
                border: none;
            }
            
            /* Status Bar */
            QStatusBar {
                background: #16161e;
                color: #c0caf5;
                border-top: 1px solid #414868;
                font-weight: 500;
            }
            
            /* Scrollbar */
            QScrollBar:vertical {
                background: #1f2335;
                width: 10px;
                border-radius: 5px;
                margin: 0;
            }
            QScrollBar::handle:vertical {
                background: #414868;
                border-radius: 5px;
                min-height: 30px;
            }
            QScrollBar::handle:vertical:hover {
                background: #565f89;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar:horizontal {
                background: #1f2335;
                height: 10px;
                border-radius: 5px;
            }
            QScrollBar::handle:horizontal {
                background: #414868;
                border-radius: 5px;
            }
            QScrollBar::handle:horizontal:hover {
                background: #565f89;
            }
        """)
    
    def setup_ui(self):
        """Setup the modern UI layout"""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(25, 25, 25, 25)

        # === HEADER SECTION ===
        header_frame = QFrame()
        header_frame.setStyleSheet("""
            QFrame {
                background: #24283b;
                border: 1px solid #414868;
                border-radius: 10px;
                padding: 15px;
            }
        """)
        header_layout = QVBoxLayout(header_frame)
        header_layout.setSpacing(5)
        
        # Title
        title = QLabel("⚡ SPArC DSL System")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""
            font-size: 28px;
            font-weight: 700;
            color: #7aa2f7;
            margin: 5px;
            background: transparent;
        """)
        
        subtitle = QLabel("System | Performance | Arithmetic | Commands | Database")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("""
            font-size: 13px;
            color: #9ece6a;
            margin: 2px;
            font-weight: 500;
            background: transparent;
        """)
        
        tagline = QLabel("Domain-Specific Language with Natural Language Processing")
        tagline.setAlignment(Qt.AlignCenter)
        tagline.setStyleSheet("""
            font-size: 11px;
            color: #787c99;
            font-style: italic;
            background: transparent;
        """)
        
        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)
        header_layout.addWidget(tagline)
        main_layout.addWidget(header_frame)

        # === CONTROLS SECTION ===
        controls_frame = QFrame()
        controls_frame.setStyleSheet("QFrame { background: transparent; border: none; }")
        controls_layout = QVBoxLayout(controls_frame)
        controls_layout.setSpacing(12)
        
        # NLP Settings
        settings_group = QGroupBox("⚙️ Processing Settings")
        settings_layout = QHBoxLayout(settings_group)
        
        self.nlp_checkbox = QCheckBox("🧠 Enable Natural Language Processing")
        self.nlp_checkbox.setChecked(self.engine.enable_nlp)
        self.nlp_checkbox.stateChanged.connect(self._toggle_nlp)
        self.nlp_checkbox.setStyleSheet("font-size: 13px; font-weight: 600; background: transparent;")
        
        settings_layout.addWidget(self.nlp_checkbox)
        settings_layout.addStretch()
        
        controls_layout.addWidget(settings_group)

        # Input Section
        input_group = QGroupBox("💬 Query Input")
        input_layout = QVBoxLayout(input_group)
        input_layout.setSpacing(10)
        
        self.input_line = QLineEdit()
        self.input_line.setPlaceholderText("Try: 'show cpu usage' or 'SHOW CPU' or 'calculate 2 + 3 * 5'")
        self.input_line.returnPressed.connect(self.run_query)
        self.input_line.setMinimumHeight(40)
        
        # Button row
        button_layout = QHBoxLayout()
        button_layout.setSpacing(8)
        
        self.run_btn = QPushButton("🚀 Execute")
        self.run_btn.clicked.connect(self.run_query)
        self.run_btn.setMinimumHeight(36)
        
        clear_btn = QPushButton("🗑️ Clear")
        clear_btn.clicked.connect(self._clear_outputs)
        clear_btn.setMinimumHeight(36)
        clear_btn.setStyleSheet("""
            QPushButton {
                background: #f7768e;
                color: #1a1b26;
            }
            QPushButton:hover {
                background: #ff7a93;
            }
            QPushButton:pressed {
                background: #e7667e;
            }
        """)
        
        examples_btn = QPushButton("💡 Examples")
        examples_btn.clicked.connect(self._show_examples)
        examples_btn.setMinimumHeight(36)
        examples_btn.setStyleSheet("""
            QPushButton {
                background: #e0af68;
                color: #1a1b26;
            }
            QPushButton:hover {
                background: #f0bf78;
            }
            QPushButton:pressed {
                background: #d09f58;
            }
        """)
        
        button_layout.addWidget(self.run_btn, 2)
        button_layout.addWidget(clear_btn, 1)
        button_layout.addWidget(examples_btn, 1)
        
        input_layout.addWidget(self.input_line)
        input_layout.addLayout(button_layout)
        controls_layout.addWidget(input_group)
        
        main_layout.addWidget(controls_frame)

        # === MAIN CONTENT AREA ===
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(2)
        
        # Left side - Output tabs
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        
        self.system_output = QTextEdit()
        self.db_output = QTextEdit()
        self.debug_output = QTextEdit()
        
        self.system_output.setReadOnly(True)
        self.db_output.setReadOnly(True)
        self.debug_output.setReadOnly(True)
        
        self.tabs.addTab(self.system_output, "🖥️  System")
        self.tabs.addTab(self.db_output, "🗄️  Database")
        self.tabs.addTab(self.debug_output, "🔍  Debug")
        
        left_layout.addWidget(self.tabs)
        
        # Right side - History and Examples
        right_widget = QWidget()
        right_widget.setMaximumWidth(320)
        right_widget.setStyleSheet("background: transparent;")
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(12)
        
        # History section
        history_group = QGroupBox("📝 Query History")
        history_layout = QVBoxLayout(history_group)
        self.history_list = QListWidget()
        self.history_list.itemDoubleClicked.connect(self._load_from_history)
        history_layout.addWidget(self.history_list)
        right_layout.addWidget(history_group)
        
        # Quick examples section
        examples_group = QGroupBox("🎯 Quick Examples")
        examples_layout = QVBoxLayout(examples_group)
        examples_layout.setSpacing(6)
        
        example_queries = [
            ("💻 CPU Usage", "show cpu usage"),
            ("💾 Memory Status", "show memory status"),
            ("📊 Running Tasks", "list running processes"),
            ("🔢 Calculate", "CALC (10+20)/2"),
            ("⚡ Performance", "SET MODE PERFORMANCE"),
            ("🗄️ Show Tables", "DB SELECT name FROM sqlite_master WHERE type='table';"),
        ]
        
        for label, query in example_queries:
            btn = QPushButton(label)
            btn.setStyleSheet("""
                QPushButton {
                    text-align: left;
                    padding: 8px 12px;
                    font-size: 11px;
                    background: #24283b;
                    border: 1px solid #414868;
                    color: #c0caf5;
                }
                QPushButton:hover {
                    background: #414868;
                    border-color: #7aa2f7;
                    color: #c0caf5;
                }
                QPushButton:pressed {
                    background: #7aa2f7;
                    color: #1a1b26;
                }
            """)
            btn.clicked.connect(lambda checked, q=query: self._set_query(q))
            examples_layout.addWidget(btn)
        
        right_layout.addWidget(examples_group)
        right_layout.addStretch()
        
        # Add to splitter
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)
        
        main_layout.addWidget(splitter, 1)

        # === STATUS BAR ===
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        self.status_label = QLabel("🟢 Ready — Try natural language or formal DSL syntax")
        self.status_label.setStyleSheet("""
            color: #9ece6a;
            padding: 6px 10px;
            font-weight: 600;
            font-size: 12px;
            background: transparent;
        """)
        
        self.processing_indicator = QProgressBar()
        self.processing_indicator.setVisible(False)
        self.processing_indicator.setMaximumHeight(6)
        self.processing_indicator.setMaximumWidth(200)
        self.processing_indicator.setTextVisible(False)
        
        self.status_bar.addWidget(self.status_label, 1)
        self.status_bar.addPermanentWidget(self.processing_indicator)

        # Show intro message
        self._show_intro()
    
    def start_intro_animation(self):
        """Subtle intro animation"""
        self.setWindowOpacity(0)
        animation = QPropertyAnimation(self, b"windowOpacity")
        animation.setDuration(500)
        animation.setStartValue(0)
        animation.setEndValue(1)
        animation.setEasingCurve(QEasingCurve.OutCubic)
        animation.start()
        self._intro_anim = animation  # Keep reference

    def _show_intro(self):
        """Show modern intro message"""
        intro_system = """
╔══════════════════════════════════════════════════════════════════╗
║                    🖥️  SYSTEM MONITOR                            ║
╚══════════════════════════════════════════════════════════════════╝

Welcome to SPArC System Monitoring!

📊 Available Commands:
  • SHOW CPU      - Display CPU usage and information
  • SHOW MEMORY   - Display RAM and swap memory
  • SHOW TASKS    - List top running processes
  • SHOW DISK     - Display disk usage
  • SHOW GPU      - Display GPU information (requires gputil)
  • SHOW NETWORK  - Display network interfaces

💡 Natural Language Examples:
  • "show me the cpu usage"
  • "how much memory is used"
  • "list running processes"

Try executing a command above! 🚀
"""
        
        intro_db = """
╔══════════════════════════════════════════════════════════════════╗
║                    🗄️  DATABASE MANAGER                          ║
╚══════════════════════════════════════════════════════════════════╝

Welcome to SPArC Database Operations!

📊 Available Commands:
  • DB SELECT * FROM table;              - Query data
  • DB CREATE TABLE name (...);          - Create table
  • DB INSERT INTO table VALUES (...);   - Insert data
  • DB UPDATE table SET ... WHERE ...;   - Update data
  • DB DELETE FROM table WHERE ...;      - Delete data

💡 Natural Language Examples:
  • "show all tables"
  • "select everything from users"
  • "create a table called products"

Database file: sparc.db 🗄️
"""
        
        intro_debug = """
╔══════════════════════════════════════════════════════════════════╗
║                    🔍  DEBUG CONSOLE                             ║
╚══════════════════════════════════════════════════════════════════╝

Welcome to SPArC Debug Console!

This tab shows detailed execution information:
  ⚙️  FLA Pipeline Stages (Lexical → Syntax → Semantic → Execution)
  🧠 NLP Processing Details (confidence, intent, entities)
  ❌ Error Messages and Stack Traces
  📊 Execution Metadata

Execute any command to see the detailed pipeline! 🔬
"""
        
        self.system_output.setPlainText(intro_system)
        self.db_output.setPlainText(intro_db)
        self.debug_output.setPlainText(intro_debug)

    def _toggle_nlp(self, state):
        """Toggle NLP with visual feedback"""
        self.engine.enable_nlp = state == Qt.Checked
        
        if self.engine.enable_nlp:
            self.status_label.setText("🧠 NLP Enabled - Natural language processing active")
            self.status_label.setStyleSheet("color: #7aa2f7; padding: 6px 10px; font-weight: 600; background: transparent;")
            self.input_line.setPlaceholderText("Try: 'show cpu usage' or 'calculate 2 + 3'")
        else:
            self.status_label.setText("⚙️ NLP Disabled - Formal DSL syntax only")
            self.status_label.setStyleSheet("color: #e0af68; padding: 6px 10px; font-weight: 600; background: transparent;")
            self.input_line.setPlaceholderText("Enter formal query: SHOW CPU | DB SELECT * FROM table")

    def _clear_outputs(self):
        """Clear outputs with animation"""
        self.system_output.clear()
        self.db_output.clear()
        self.debug_output.clear()
        self._show_intro()
        self.status_label.setText("🗑️ Outputs cleared")
        
        # Flash animation
        QTimer.singleShot(2000, lambda: self.status_label.setText("🟢 Ready"))

    def _show_examples(self):
        """Show comprehensive examples"""
        examples = """
╔══════════════════════════════════════════════════════════════════╗
║                    💡 COMMAND EXAMPLES                           ║
╚══════════════════════════════════════════════════════════════════╝

🖥️  SYSTEM MONITORING:
  Natural Language:
    • show me the cpu usage
    • what is the processor utilization
    • how much memory is being used
    • display ram status
    • list all running processes
    • what programs are active
    
  Formal DSL:
    • SHOW CPU
    • SHOW MEMORY
    • SHOW TASKS
    • SHOW DISK
    • SHOW NETWORK

⚡ PERFORMANCE CONTROL:
  Natural Language:
    • set performance mode
    • switch to balanced mode
    • enable power saver
    
  Formal DSL:
    • SET MODE PERFORMANCE
    • SET MODE BALANCED
    • SET MODE POWER_SAVER
    • SET FAN 70

🔢 ARITHMETIC:
  Natural Language:
    • calculate 2 plus 3
    • what is 5 times 10
    • compute 100 divided by 4
    
  Formal DSL:
    • CALC 2+3
    • CALC (10+20)/2
    • CALC 2+3*5

📁 FILE OPERATIONS:
  Natural Language:
    • copy file.txt to backup
    • move report.docx to archive
    • delete temp.txt
    
  Formal DSL:
    • COPY file.txt backup/
    • MOVE old.txt archive/
    • DELETE temp.log
    • OPEN notepad

🗄️  DATABASE:
  Natural Language:
    • show all tables
    • select everything from users
    • create a table called products
    
  Formal DSL:
    • DB SELECT * FROM sqlite_master WHERE type='table';
    • DB CREATE TABLE test (id INTEGER, name TEXT);
    • DB INSERT INTO test VALUES (1, 'Sample');
    • DB SELECT * FROM test;

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💡 TIP: The system automatically detects whether you're using
natural language or formal syntax and processes accordingly!

🎨 TIP: Check the Debug tab after each command to see the
complete FLA pipeline execution details!
"""
        self.debug_output.setPlainText(examples)
        self.tabs.setCurrentIndex(2)

    def _set_query(self, query):
        """Set query in input field"""
        self.input_line.setText(query)
        self.input_line.setFocus()

    def _load_from_history(self, item):
        """Load query from history"""
        query = item.text()
        self.input_line.setText(query)
        self.input_line.setFocus()

    def _add_to_history(self, query):
        """Add query to history"""
        if query and query not in self.query_history:
            self.query_history.append(query)
            self.history_list.addItem(query)
            
            if len(self.query_history) > 20:
                self.query_history.pop(0)
                self.history_list.takeItem(0)

    def run_query(self):
        """Execute query with visual feedback"""
        query = self.input_line.text().strip()
        if not query:
            return

        # Add to history
        self._add_to_history(query)
        
        # Show processing animation
        self.processing_indicator.setVisible(True)
        self.processing_indicator.setRange(0, 0)  # Indeterminate
        self.status_label.setText(f"🔄 Processing: {query[:50]}{'...' if len(query) > 50 else ''}")
        self.run_btn.setEnabled(False)
        QApplication.processEvents()

        try:
            # Execute query
            result_dict = self.engine.execute(query)
            result = result_dict['result']
            
            # Clear input
            self.input_line.clear()
            
            # Format and display output
            self._display_result(query, result_dict)
            
            # Update status
            if result_dict['success']:
                confidence_info = ""
                if 'nlp_confidence' in result_dict:
                    confidence_info = f" (🧠 {result_dict['nlp_confidence']:.0%})"
                self.status_label.setText(f"✅ Success - {result_dict['method']}{confidence_info}")
                self.status_label.setStyleSheet("color: #9ece6a; padding: 6px 10px; font-weight: 600; background: transparent;")
            else:
                self.status_label.setText(f"❌ Error - {result_dict.get('error', 'Unknown error')}")
                self.status_label.setStyleSheet("color: #f7768e; padding: 6px 10px; font-weight: 600; background: transparent;")
                
        except Exception as e:
            self.status_label.setText(f"❌ Exception: {str(e)}")
            self.status_label.setStyleSheet("color: #f7768e; padding: 6px 10px; font-weight: 600; background: transparent;")
            self.debug_output.setPlainText(f"🚨 EXCEPTION\n\n{str(e)}\n\nQuery: {query}")
            self.system_output.setPlainText(f"❌ Error executing query: {query}\n\n{str(e)}")
            self.tabs.setCurrentIndex(0)
            
        finally:
            self.processing_indicator.setVisible(False)
            self.run_btn.setEnabled(True)
            
            # Reset status after 3 seconds
            QTimer.singleShot(3000, self._reset_status)

    def _reset_status(self):
        """Reset status bar to default"""
        if self.engine.enable_nlp:
            self.status_label.setText("🧠 NLP Enabled - Ready for queries")
            self.status_label.setStyleSheet("color: #7aa2f7; padding: 6px 10px; font-weight: 600; background: transparent;")
        else:
            self.status_label.setText("🟢 Ready - Formal DSL mode")
            self.status_label.setStyleSheet("color: #9ece6a; padding: 6px 10px; font-weight: 600; background: transparent;")

    def _display_result(self, query, result_dict):
        """Display result in appropriate tab with formatting"""
        result = result_dict['result']
        
        # Format debug info
        debug_info = self._format_debug_info(result_dict)
        self.debug_output.setPlainText(debug_info)
        
        # Route to appropriate tab
        if result_dict.get('ast_type') == 'ShowNode' or any(keyword in result for keyword in ["Running Tasks", "CPU", "Memory", "Disk", "Network", "GPU", "Temperature"]):
            formatted = f"""
╔══════════════════════════════════════════════════════════════════╗
║  SYSTEM QUERY RESULT                                             ║
╚══════════════════════════════════════════════════════════════════╝

📝 Query: {query}

{result}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Executed successfully
"""
            self.system_output.setPlainText(formatted)
            self.tabs.setCurrentIndex(0)
            
        elif result_dict.get('ast_type') == 'DBQueryNode' or "DB" in query.upper()[:5]:
            formatted = f"""
╔══════════════════════════════════════════════════════════════════╗
║  DATABASE QUERY RESULT                                           ║
╚══════════════════════════════════════════════════════════════════╝

📝 Query: {query}

{result}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Executed successfully
"""
            self.db_output.setPlainText(formatted)
            self.tabs.setCurrentIndex(1)
            
        elif result_dict.get('ast_type') in ['CalcNode', 'SetModeNode', 'SetFanNode', 'FileOpNode']:
            formatted = f"""
╔══════════════════════════════════════════════════════════════════╗
║  COMMAND RESULT                                                  ║
╚══════════════════════════════════════════════════════════════════╝

📝 Query: {query}

{result}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Executed successfully
"""
            self.system_output.setPlainText(formatted)
            self.tabs.setCurrentIndex(0)
            
        else:
            formatted = f"""
╔══════════════════════════════════════════════════════════════════╗
║  RESULT                                                          ║
╚══════════════════════════════════════════════════════════════════╝

📝 Query: {query}

{result}
"""
            self.system_output.setPlainText(formatted)
            self.tabs.setCurrentIndex(0)

    def _format_debug_info(self, result_dict):
        """Format debug information with modern styling"""
        timestamp = __import__('datetime').datetime.now().strftime("%H:%M:%S")
        
        lines = [
            "╔══════════════════════════════════════════════════════════════════╗",
            "║                    🔍 DEBUG INFORMATION                          ║",
            "╚══════════════════════════════════════════════════════════════════╝",
            "",
            f"⏰ Timestamp: {timestamp}",
            f"📝 Original Query: {result_dict['original_query']}",
            f"🔧 Formal Query: {result_dict['formal_query']}",
            f"🎯 Method: {result_dict['method']}",
            f"{'✅' if result_dict['success'] else '❌'} Status: {'Success' if result_dict['success'] else 'Failed'}",
            ""
        ]
        
        if 'nlp_used' in result_dict:
            lines.extend([
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                "🧠 NLP PROCESSING:",
                f"   Used: {'Yes' if result_dict['nlp_used'] else 'No'}",
                f"   Method: {result_dict.get('nlp_method', 'N/A')}",
                f"   Confidence: {result_dict.get('nlp_confidence', 0):.1%}",
            ])
            
            if 'intent' in result_dict:
                lines.append(f"   Intent: {result_dict['intent']}")
            
            if 'entities' in result_dict and result_dict['entities']:
                entities_str = ', '.join([f"{k}: {v}" for k, v in result_dict['entities'].items() if v])
                if entities_str:
                    lines.append(f"   Entities: {entities_str}")
            lines.append("")
        
        if 'stages' in result_dict:
            lines.extend([
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                "🔬 FLA PIPELINE STAGES:",
            ])
            for stage, status in result_dict['stages'].items():
                icon = "✅" if status == "completed" else "❌" if status == "failed" else "⏭️"
                lines.append(f"   {icon} {stage.title()}: {status}")
            lines.append("")
        
        if 'ast_type' in result_dict:
            lines.extend([
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                "🌳 AST INFORMATION:",
                f"   Type: {result_dict['ast_type']}"
            ])
            
            if 'target' in result_dict:
                lines.append(f"   Target: {result_dict['target']}")
            elif 'sql' in result_dict:
                lines.append(f"   SQL: {result_dict['sql'][:80]}...")
            lines.append("")
        
        if 'error' in result_dict:
            lines.extend([
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                "❌ ERROR INFORMATION:",
                f"   {result_dict['error']}"
            ])
        
        lines.append("\n" + "═" * 70)
        
        return "\n".join(lines)

    def closeEvent(self, event):
        """Handle window close"""
        self.engine.close()
        event.accept()


# For testing
if __name__ == "__main__":
    print("Use main_enhanced.py to run the application")
