# -*- coding: utf-8 -*-
"""
Log functionality for the XGeoAI Pollution Analyst plugin.
"""

import logging
import gzip
from logging.handlers import RotatingFileHandler
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QApplication
from PyQt5.QtGui import QColor, QTextCharFormat, QTextCursor


class LogManager:
    """
    Manages logging functionality for the plugin.
    """

    @staticmethod
    def configure_logging(log_widget, log_file=None, log_level=logging.DEBUG, timestamp_fmt="%Y-%m-%d %H:%M:%S"):
        """
        Configures logging to output messages to the QTextEdit widget and optionally to a file.

        :param log_widget: QTextEdit widget to display log messages.
        :param log_file: Path to the log file (optional).
        :param log_level: Logging level (e.g., logging.DEBUG, logging.INFO).
        :param timestamp_fmt: Timestamp format string.
        """
        if log_widget is None:
            logging.error("Log widget is None during logging configuration.")
            raise ValueError("Log widget cannot be None.")

        # Clear existing handlers
        logging.getLogger().handlers.clear()

        # Add QTextEditHandler
        log_handler = QTextEditHandler(log_widget)
        log_handler.setFormatter(logging.Formatter(f"%(asctime)s - %(levelname)s - %(message)s", timestamp_fmt))
        logging.getLogger().addHandler(log_handler)

        # Add FileHandler if log_file is provided
        if log_file:
            file_handler = RotatingFileHandler(log_file, maxBytes=1024 * 1024, backupCount=5)
            file_handler.setFormatter(logging.Formatter(f"%(asctime)s - %(levelname)s - %(message)s", timestamp_fmt))
            logging.getLogger().addHandler(file_handler)

        # Set logging level
        logging.getLogger().setLevel(log_level)

    @staticmethod
    def save_log(log_widget):
        """
        Saves the log content to a file chosen by the user.

        :param log_widget: QTextEdit widget containing log messages.
        """
        if not LogManager._validate_widget(log_widget):
            return

        try:
            file_path, _ = QFileDialog.getSaveFileName(None, "Save Log", "", "Text Files (*.txt);;All Files (*)")
            if file_path:
                with open(file_path, "w", encoding="utf-8") as file:
                    file.write(log_widget.toPlainText())
                QMessageBox.information(None, "Success", "Log saved successfully!")
        except Exception as e:
            QMessageBox.critical(None, "Error", f"Failed to save log: {e}")
            logging.error(f"Error saving log: {e}")

    @staticmethod
    def save_compressed_log(log_widget):
        """
        Saves the log content to a compressed file chosen by the user.

        :param log_widget: QTextEdit widget containing log messages.
        """
        if not LogManager._validate_widget(log_widget):
            return

        try:
            file_path, _ = QFileDialog.getSaveFileName(None, "Save Compressed Log", "", "Gzip Files (*.gz);;All Files (*)")
            if file_path:
                with gzip.open(file_path, "wt", encoding="utf-8") as file:
                    file.write(log_widget.toPlainText())
                QMessageBox.information(None, "Success", "Log saved and compressed successfully!")
        except Exception as e:
            QMessageBox.critical(None, "Error", f"Failed to save compressed log: {e}")
            logging.error(f"Error saving compressed log: {e}")

    @staticmethod
    def export_log_csv(log_widget):
        """
        Exports the log content to a CSV file chosen by the user.

        :param log_widget: QTextEdit widget containing log messages.
        """
        if not LogManager._validate_widget(log_widget):
            return

        try:
            file_path, _ = QFileDialog.getSaveFileName(None, "Export Log as CSV", "", "CSV Files (*.csv);;All Files (*)")
            if file_path:
                with open(file_path, "w", encoding="utf-8") as file:
                    file.write("Timestamp,Level,Message\n")
                    for line in log_widget.toPlainText().splitlines():
                        parts = line.split(" - ")
                        if len(parts) == 3:
                            file.write(f"{parts[0]},{parts[1]},{parts[2]}\n")
                QMessageBox.information(None, "Success", "Log exported to CSV successfully!")
        except Exception as e:
            QMessageBox.critical(None, "Error", f"Failed to export log: {e}")
            logging.error(f"Error exporting log: {e}")

    @staticmethod
    def copy_log(log_widget):
        """
        Copies the log content to the clipboard.

        :param log_widget: QTextEdit widget containing log messages.
        """
        if not LogManager._validate_widget(log_widget):
            return

        try:
            log_text = log_widget.toPlainText()
            if log_text:
                clipboard = QApplication.clipboard()
                clipboard.setText(log_text)
                QMessageBox.information(None, "Success", "Log copied to clipboard!")
            else:
                QMessageBox.warning(None, "Warning", "Log is empty, nothing to copy.")
        except Exception as e:
            QMessageBox.critical(None, "Error", f"Failed to copy log: {e}")
            logging.error(f"Error copying log: {e}")

    @staticmethod
    def clear_log(log_widget):
        """
        Clears the log content.

        :param log_widget: QTextEdit widget containing log messages.
        """
        if not LogManager._validate_widget(log_widget):
            return

        try:
            log_widget.clear()
            QMessageBox.information(None, "Success", "Log cleared.")
        except Exception as e:
            QMessageBox.critical(None, "Error", f"Failed to clear log: {e}")
            logging.error(f"Error clearing log: {e}")

    @staticmethod
    def search_log(log_widget, text):
        """
        Searches for text in the log widget.

        :param log_widget: QTextEdit widget containing log messages.
        :param text: Text to search for.
        """
        if not LogManager._validate_widget(log_widget):
            return

        cursor = log_widget.textCursor()
        cursor.movePosition(QTextCursor.Start)
        while log_widget.find(text, cursor):
            log_widget.setTextCursor(cursor)

    @staticmethod
    def _validate_widget(widget):
        """
        Validates that the widget is not None and is visible.

        :param widget: QWidget to validate.
        :return: True if the widget is valid, False otherwise.
        """
        if widget is None:
            logging.warning("Log widget is None.")
            QMessageBox.warning(None, "Warning", "Log widget is not accessible (None).")
            return False
        if not widget.isVisible():
            logging.warning("Log widget is not visible.")
            QMessageBox.warning(None, "Warning", "Log widget is not accessible (not visible).")
            return False
        return True


class QTextEditHandler(logging.Handler):
    """
    A custom logging handler that writes log messages to a QTextEdit widget.
    """

    def __init__(self, widget):
        super().__init__()
        if widget is None:
            raise ValueError("Log widget cannot be None.")
        self.widget = widget
        self._setup_highlighting()

    def _setup_highlighting(self):
        """
        Sets up text highlighting for different log levels.
        """
        self.highlight_formats = {
            logging.DEBUG: QTextCharFormat(),
            logging.INFO: QTextCharFormat(),
            logging.WARNING: QTextCharFormat(),
            logging.ERROR: QTextCharFormat(),
            logging.CRITICAL: QTextCharFormat(),
        }
        self.highlight_formats[logging.WARNING].setForeground(QColor("orange"))
        self.highlight_formats[logging.ERROR].setForeground(QColor("red"))
        self.highlight_formats[logging.CRITICAL].setForeground(QColor("darkred"))

    def emit(self, record):
        """
        Emits a log record to the QTextEdit widget.
        """
        try:
            log_entry = self.format(record)
            cursor = QTextCursor(self.widget.document())
            cursor.movePosition(QTextCursor.End)
            cursor.insertText(log_entry + "\n", self.highlight_formats.get(record.levelno, QTextCharFormat()))
        except RuntimeError as e:
            logging.error(f"RuntimeError in QTextEditHandler.emit: {e}")
        except Exception as e:
            logging.error(f"Unexpected error in QTextEditHandler.emit: {e}")