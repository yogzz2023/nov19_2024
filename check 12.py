import sys
import numpy as np
import csv
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel, QComboBox, QTextEdit,
                             QHBoxLayout, QSplitter, QCheckBox, QLineEdit, QDialog, QGridLayout, QGroupBox, QRadioButton,
                             QFrame, QSizePolicy, QToolButton, QTabWidget, QTableWidget, QTableWidgetItem, QScrollArea)
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtCore import Qt

import pyqtgraph as pg


class OutputStream:
    """Custom stream class to redirect stdout."""

    def __init__(self, text_edit):
        self.text_edit = text_edit

    def write(self, text):
        self.text_edit.append(text)

    def flush(self):
        pass  # No need to implement flush for QTextEdit


class KalmanFilterGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.tracks = []
        self.input_file = None
        self.filter_mode = "CV"  # Default filter mode
        self.control_panel_collapsed = False
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Kalman Filter GUI')
        self.setGeometry(100, 100, 1200, 600)
        self.setStyleSheet("""
            QWidget {
                background-color: #222222;
                color: #ffffff;
                font-family: "Arial", sans-serif;
            }
            QPushButton, QToolButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px 16px;
                font-size: 16px;
                border-radius: 4px;
            }
            QPushButton:hover, QToolButton:hover {
                background-color: #3e8e41;
            }
            QLabel, QComboBox, QLineEdit, QTextEdit {
                background-color: #333333;
                color: white;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 5px;
            }
            QTabWidget::pane {
                background-color: #222222;
            }
        """)

        # Main layout
        main_layout = QHBoxLayout()

        # Left Control Panel
        left_layout = QVBoxLayout()
        self.collapse_button = QToolButton()
        self.collapse_button.setText("=")
        self.collapse_button.clicked.connect(self.toggle_control_panel)
        left_layout.addWidget(self.collapse_button)

        self.control_panel = QWidget()
        control_layout = QVBoxLayout()
        self.control_panel.setLayout(control_layout)
        left_layout.addWidget(self.control_panel)

        self.file_upload_button = QPushButton("Upload File")
        self.file_upload_button.clicked.connect(self.select_file)
        control_layout.addWidget(self.file_upload_button)

        self.config_button = QPushButton("System Configuration")
        control_layout.addWidget(self.config_button)

        self.track_mode_label = QLabel("Track Mode")
        control_layout.addWidget(self.track_mode_label)
        self.track_mode_combo = QComboBox()
        self.track_mode_combo.addItems(["3-state", "5-state", "7-state"])
        control_layout.addWidget(self.track_mode_combo)

        self.jpda_radio = QRadioButton("JPDA")
        self.jpda_radio.setChecked(True)
        control_layout.addWidget(self.jpda_radio)

        self.munkres_radio = QRadioButton("Munkres")
        control_layout.addWidget(self.munkres_radio)

        self.cv_filter_button = QPushButton("CV Filter")
        self.cv_filter_button.clicked.connect(lambda: self.select_filter("CV"))
        control_layout.addWidget(self.cv_filter_button)

        self.process_button = QPushButton("Process")
        self.process_button.clicked.connect(self.process_data)
        control_layout.addWidget(self.process_button)

        # Add left panel to main layout
        main_layout.addLayout(left_layout)

        # Right Output and Plot
        right_layout = QVBoxLayout()

        self.tab_widget = QTabWidget()
        self.output_tab = QWidget()
        self.plot_tab = QWidget()
        self.track_info_tab = QWidget()

        self.tab_widget.addTab(self.output_tab, "Output")
        self.tab_widget.addTab(self.plot_tab, "Plot")
        self.tab_widget.addTab(self.track_info_tab, "Track Info")

        # Output tab setup
        self.output_display = QTextEdit()
        self.output_display.setReadOnly(True)
        output_layout = QVBoxLayout()
        output_layout.addWidget(self.output_display)
        self.output_tab.setLayout(output_layout)

        # Plot tab setup
        self.plot_widget = pg.GraphicsLayoutWidget()
        plot_layout = QVBoxLayout()
        plot_layout.addWidget(self.plot_widget)
        self.plot_tab.setLayout(plot_layout)

        # Track Info tab setup
        self.csv_table = QTableWidget()
        track_info_layout = QVBoxLayout()
        track_info_layout.addWidget(self.csv_table)
        self.track_info_tab.setLayout(track_info_layout)

        # Add tab widget to the right layout
        right_layout.addWidget(self.tab_widget)
        main_layout.addLayout(right_layout)

        # Redirect stdout to the output display
        sys.stdout = OutputStream(self.output_display)

        self.setLayout(main_layout)

    def toggle_control_panel(self):
        self.control_panel_collapsed = not self.control_panel_collapsed
        self.control_panel.setVisible(not self.control_panel_collapsed)
        self.adjustSize()

    def select_file(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Input File", "", "CSV Files (*.csv);;All Files (*)", options=options)
        if file_name:
            self.input_file = file_name
            print(f"File selected: {self.input_file}")

    def process_data(self):
        if not self.input_file:
            print("Please upload a file first.")
            return

        track_mode = self.track_mode_combo.currentText()
        association_type = "JPDA" if self.jpda_radio.isChecked() else "Munkres"

        print(f"Processing data with:\nFile: {self.input_file}\nTrack Mode: {track_mode}\nAssociation: {association_type}\nFilter: {self.filter_mode}")
        # Here you would call your Kalman filter processing function
        # self.tracks = kalman_filter_main(self.input_file, track_mode, association_type)

    def select_filter(self, mode):
        self.filter_mode = mode
        print(f"Selected filter: {self.filter_mode}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = KalmanFilterGUI()
    gui.show()
    sys.exit(app.exec_())
