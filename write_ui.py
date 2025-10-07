# Temporary script to write the clean UI
code = r"""# ui_enhanced.py
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
"""

with open('ui_enhanced.py', 'w', encoding='utf-8') as f:
    f.write(code)

print("UI file created successfully!")
