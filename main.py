# main.py
import sys
import os
from PyQt5.QtWidgets import QApplication, QMessageBox, QSplashScreen
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QPainter, QColor, QFont
from ui import MainWindow
from dsl_core import ExecutionEngine

def create_splash_screen():
    """Create a simple splash screen."""
    # Create a simple colored pixmap
    pixmap = QPixmap(400, 200)
    pixmap.fill(QColor('#1e1e2e'))
    
    painter = QPainter(pixmap)
    painter.setPen(QColor('#f5c2e7'))
    painter.setFont(QFont('Arial', 16, QFont.Bold))
    painter.drawText(pixmap.rect(), Qt.AlignCenter, 
                     "⚡ SPARC\nCustom DSL System\nInitializing...")
    painter.end()
    
    splash = QSplashScreen(pixmap)
    splash.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
    return splash

def main():
    # Set up high DPI support
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    # Show splash screen
    splash = create_splash_screen()
    splash.show()
    app.processEvents()
    
    try:
        # Initialize the engine (this may take time for NLP setup)
        splash.showMessage("Loading NLP components...", Qt.AlignBottom | Qt.AlignCenter, QColor('#a6e3a1'))
        app.processEvents()
        
        # Try to initialize with NLP first
        engine = None
        nlp_failed = False
        
        try:
            engine = ExecutionEngine(enable_nlp=True)
        except Exception as nlp_error:
            print(f"NLP initialization failed: {nlp_error}")
            splash.showMessage("NLP failed, using formal DSL only...", Qt.AlignBottom | Qt.AlignCenter, QColor('#f9e2af'))
            app.processEvents()
            nlp_failed = True
            engine = ExecutionEngine(enable_nlp=False)
        
        splash.showMessage("Creating interface...", Qt.AlignBottom | Qt.AlignCenter, QColor('#a6e3a1'))
        app.processEvents()
        
        # Create main window
        window = MainWindow(engine)
        
        # Show window and hide splash
        window.show()
        splash.finish(window)
        
        # Show welcome message if NLP failed to initialize
        if nlp_failed or not engine.enable_nlp:
            QMessageBox.information(window, "NLP Notice", 
                                   "NLP components could not be initialized.\n"
                                   "The system will work with formal DSL syntax only.\n\n"
                                   "Supported commands:\n"
                                   "• SHOW CPU\n"
                                   "• SHOW MEMORY\n" 
                                   "• SHOW TASKS\n"
                                   "• DB <any SQLite command>\n\n"
                                   "To enable NLP, ensure NLTK and scikit-learn are properly installed.")
        
        sys.exit(app.exec_())
        
    except Exception as e:
        splash.hide()
        
        # Show error dialog
        error_box = QMessageBox()
        error_box.setIcon(QMessageBox.Critical)
        error_box.setWindowTitle("Initialization Error")
        error_box.setText(f"Failed to initialize SPARC:\n\n{str(e)}")
        error_box.setDetailedText(f"Error details:\n{str(e)}")
        error_box.exec_()
        
        sys.exit(1)

if __name__ == "__main__":
    main()