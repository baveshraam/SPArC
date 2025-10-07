# main_enhanced.py
"""
Enhanced main entry point for SPArC DSL System
Supports all features: System Queries, Performance Control, Arithmetic, File Operations
"""

import sys
import os
from PyQt5.QtWidgets import QApplication, QMessageBox, QSplashScreen
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QPainter, QColor, QFont
from ui_enhanced import MainWindow  # Use modern enhanced UI

# Import enhanced engine
try:
    from dsl_core_enhanced import ExecutionEngine
    ENHANCED_ENGINE = True
except ImportError:
    print("Enhanced engine not found, falling back to basic engine")
    from dsl_core_enhanced import ExecutionEngine
    ENHANCED_ENGINE = False


def create_splash_screen():
    """Create an enhanced splash screen."""
    pixmap = QPixmap(500, 250)
    pixmap.fill(QColor('#1e1e2e'))
    
    painter = QPainter(pixmap)
    
    # Title
    painter.setPen(QColor('#f5c2e7'))
    painter.setFont(QFont('Arial', 20, QFont.Bold))
    painter.drawText(pixmap.rect().adjusted(0, 40, 0, -150), Qt.AlignCenter, "⚡ SPArC DSL")
    
    # Subtitle
    painter.setPen(QColor('#a6e3a1'))
    painter.setFont(QFont('Arial', 12))
    painter.drawText(pixmap.rect().adjusted(0, 80, 0, -100), Qt.AlignCenter, 
                     "System | Performance | Arithmetic | Commands")
    
    # Description
    painter.setPen(QColor('#9399b2'))
    painter.setFont(QFont('Arial', 10))
    painter.drawText(pixmap.rect().adjusted(0, 120, 0, -50), Qt.AlignCenter,
                     "Custom Domain-Specific Language")
    painter.drawText(pixmap.rect().adjusted(0, 140, 0, -30), Qt.AlignCenter,
                     "with Natural Language Processing")
    
    painter.end()
    
    splash = QSplashScreen(pixmap)
    splash.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
    return splash


def main():
    """Main entry point for SPArC application."""
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
        # Initialize engine
        splash.showMessage("⚙️ Loading DSL Engine...", 
                          Qt.AlignBottom | Qt.AlignCenter, QColor('#a6e3a1'))
        app.processEvents()
        
        engine = None
        nlp_failed = False
        
        # Try to initialize with NLP
        try:
            splash.showMessage("🧠 Initializing NLP Components...", 
                              Qt.AlignBottom | Qt.AlignCenter, QColor('#a6e3a1'))
            app.processEvents()
            
            engine = ExecutionEngine(enable_nlp=True)
            
            if not engine.enable_nlp:
                nlp_failed = True
                
        except Exception as nlp_error:
            print(f"NLP initialization failed: {nlp_error}")
            splash.showMessage("⚠️ NLP unavailable, using formal DSL only...", 
                              Qt.AlignBottom | Qt.AlignCenter, QColor('#f9e2af'))
            app.processEvents()
            nlp_failed = True
            engine = ExecutionEngine(enable_nlp=False)
        
        # Create main window
        splash.showMessage("🎨 Creating User Interface...", 
                          Qt.AlignBottom | Qt.AlignCenter, QColor('#a6e3a1'))
        app.processEvents()
        
        window = MainWindow(engine)
        
        # Show window and hide splash
        window.show()
        splash.finish(window)
        
        # Show welcome message if needed
        if nlp_failed or not engine.enable_nlp:
            msg = QMessageBox(window)
            msg.setIcon(QMessageBox.Information)
            msg.setWindowTitle("SPArC - NLP Notice")
            msg.setText("NLP components could not be initialized.")
            msg.setInformativeText(
                "The system will work with formal DSL syntax only.\n\n"
                "Supported commands:\n"
                "• System: SHOW CPU, SHOW MEMORY, SHOW TASKS, SHOW GPU\n"
                "• Performance: SET MODE PERFORMANCE, SET FAN 70\n"
                "• Arithmetic: CALC 2+3*5, CALC (10+20)/2\n"
                "• File Ops: COPY src dest, MOVE src dest, DELETE file\n"
                "• Database: DB SELECT * FROM table;\n\n"
                "To enable NLP, run:\n"
                "pip install nltk scikit-learn"
            )
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
        else:
            # Show welcome with NLP enabled
            msg = QMessageBox(window)
            msg.setIcon(QMessageBox.Information)
            msg.setWindowTitle("🎉 SPArC - Ready!")
            msg.setText("Welcome to SPArC DSL System!")
            msg.setInformativeText(
                f"All features enabled:\n"
                f"✅ Natural Language Processing\n"
                f"✅ System Monitoring (CPU, Memory, GPU, Disk, Tasks)\n"
                f"✅ Performance Control (Power Modes, Fan Speed)\n"
                f"✅ Arithmetic Calculations\n"
                f"✅ File Operations (Copy, Move, Delete, Open)\n"
                f"✅ Database Operations (SQLite)\n\n"
                f"Try saying: 'show me the cpu usage' or 'calculate 2 + 3 * 5'"
            )
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
        
        # Run application
        sys.exit(app.exec_())
        
    except Exception as e:
        splash.hide()
        
        # Show error dialog
        error_box = QMessageBox()
        error_box.setIcon(QMessageBox.Critical)
        error_box.setWindowTitle("SPArC - Initialization Error")
        error_box.setText(f"Failed to initialize SPArC:\n\n{str(e)}")
        error_box.setDetailedText(
            f"Error details:\n{str(e)}\n\n"
            f"Traceback:\n{__import__('traceback').format_exc()}"
        )
        error_box.exec_()
        
        sys.exit(1)


if __name__ == "__main__":
    print("=" * 60)
    print("SPArC - Custom Domain-Specific Language System")
    print("System | Performance | Arithmetic | Commands")
    print("=" * 60)
    print()
    
    main()
