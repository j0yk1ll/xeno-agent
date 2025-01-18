import logging
import sys
import queue
import threading

from pathlib import Path

from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QMainWindow,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QStackedWidget,
    QComboBox,
    QLineEdit,
    QGroupBox,
    QLabel,
    QTextEdit,
    QSizePolicy,
    QMessageBox,
    QScrollArea,
    QFrame,
    QCheckBox,
)
from PySide6 import QtSvg, QtGui, QtCore

from src.utils.threads.conversation_agent_thread import conversation_agent_thread
from src.utils.threads.tts_thread import tts_thread
from src.utils.threads.stt_thread import stt_thread
from src.utils.tts import get_available_voices
from src.utils.settings_manager import SettingsManager

###############################################################################
# Configure logging
###############################################################################
LOGGING_LEVEL = logging.DEBUG

# We'll place logs in ~/.xeno/app.log or C:\Users\<username>\.xeno/app.log
xeno_dir = Path.home() / ".xeno"
xeno_dir.mkdir(parents=True, exist_ok=True)  # Ensure the folder exists

log_file_path = xeno_dir / "app.log"

logger = logging.getLogger()
logger.setLevel(LOGGING_LEVEL)

# Stream handler
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(LOGGING_LEVEL)

# File handler with absolute path in ~/.xeno or C:\Users\<username>\.xeno
try:
    file_handler = logging.FileHandler(log_file_path, mode="w")
    file_handler.setLevel(LOGGING_LEVEL)
except Exception as e:
    stream_handler.setLevel(LOGGING_LEVEL)
    logger.error(f"Failed to create file handler: {e}")
    file_handler = None

# Formatter
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
stream_handler.setFormatter(formatter)
if file_handler:
    file_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(stream_handler)
if file_handler:
    logger.addHandler(file_handler)

###############################################################################
# Custom AnimatedGradientLabel
###############################################################################
class AnimatedGradientLabel(QLabel):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.gradient_offset = 0.0  # Initial offset for the gradient

        # Timer to update the gradient animation
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_gradient)
        self.timer.start(10)  # Update every 10 ms for smooth animation

        # Font settings
        font = QtGui.QFont("Arial", 24, QtGui.QFont.Bold)
        self.setFont(font)

    def update_gradient(self):
        self.gradient_offset += 0.005
        if self.gradient_offset >= 1.0:
            self.gradient_offset -= 1.0
        self.update()  # Trigger a repaint

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        rect = self.rect()

        # Calculate the shift in pixels based on the offset
        shift = self.gradient_offset * rect.width()

        # Define the gradient shifted by the offset
        gradient = QtGui.QLinearGradient(
            rect.left() + shift, 0, rect.right() + shift, 0
        )
        gradient.setSpread(QtGui.QGradient.RepeatSpread)

        # Define color stops
        gradient.setColorAt(0.0, QtGui.QColor("white"))
        gradient.setColorAt(0.25, QtGui.QColor("white"))
        gradient.setColorAt(0.5, QtGui.QColor("lightgray"))
        gradient.setColorAt(0.75, QtGui.QColor("white"))
        gradient.setColorAt(1.0, QtGui.QColor("white"))

        painter.setPen(QtCore.Qt.NoPen)
        path = QtGui.QPainterPath()
        font = self.font()

        # Position to center the text
        metrics = QtGui.QFontMetrics(font)
        text_width = metrics.horizontalAdvance(self.text())
        text_height = metrics.height()
        x = (rect.width() - text_width) / 2
        y = (rect.height() + text_height) / 2 - metrics.descent()

        path.addText(x, y, font, self.text())
        painter.fillPath(path, QtGui.QBrush(gradient))

        painter.end()

###############################################################################
# Helper function to color an SVG
###############################################################################
def QColoredSVGIcon(svg_path, color):
    if isinstance(color, int):
        color = QtGui.QColor(QtGui.QColor.fromRgb(color))
    elif isinstance(color, str):
        color = QtGui.QColor(color)
    elif not isinstance(color, QtGui.QColor):
        raise ValueError("Invalid color format. Use QColor, int, or hex string.")

    image = QtGui.QImage(64, 64, QtGui.QImage.Format_ARGB32)
    image.fill(QtCore.Qt.transparent)

    svg_renderer = QtSvg.QSvgRenderer(svg_path)
    if not svg_renderer.isValid():
        raise FileNotFoundError(f"Invalid or corrupted SVG file: {svg_path}")

    painter = QtGui.QPainter(image)
    try:
        svg_renderer.render(painter)
        painter.setCompositionMode(QtGui.QPainter.CompositionMode_SourceIn)
        painter.fillRect(image.rect(), color)
    finally:
        painter.end()

    return QtGui.QIcon(QtGui.QPixmap.fromImage(image))

###############################################################################
# Main Window
###############################################################################
class MainWindow(QMainWindow):

    def __init__(self, settings_manager: SettingsManager, agent_inbound: queue.Queue, agent_outbound: queue.Queue, tts_inbound: queue.Queue, stt_outbound: queue.Queue,
                 stt_is_recording_event: threading.Event):
        super().__init__()
        self.setWindowTitle("Xeno Agent")
        self.setGeometry(100, 100, 400, 600)

        # Register queues
        self.agent_inbound = agent_inbound  # UI -> agent
        self.agent_outbound = agent_outbound  # agent -> UI
        self.tts_inbound = tts_inbound  # UI -> tts
        self.stt_outbound = stt_outbound  # stt -> UI

        # *** Added *** Initialize voice recording state
        self.is_voice_recording = False  # Tracks if voice recording is active

        # Register STT control event
        self.stt_is_recording_event = stt_is_recording_event

        # Store settings manager
        self.settings_manager = settings_manager

        # Read settings from the manager
        self.completion_model_id = self.settings_manager.get_settings_key("completion_model_id", "ollama/qwen2.5-coder")
        self.completion_api_base = self.settings_manager.get_settings_key("completion_api_base", "http://localhost:11434")
        self.completion_api_key = self.settings_manager.get_settings_key("completion_api_key", "")

        self.embedding_model_id = self.settings_manager.get_settings_key("embedding_model_id", "ollama/granite-embedding")
        self.embedding_api_base = self.settings_manager.get_settings_key("embedding_api_base", "http://localhost:11434")
        self.embedding_api_key = self.settings_manager.get_settings_key("embedding_api_key", "")

        # *** Added *** Browser Use Model
        self.browser_use_model_id = self.settings_manager.get_settings_key("browser_use_model_id", "ollama/qwen2.5-coder")
        self.browser_use_api_base = self.settings_manager.get_settings_key("browser_use_api_base", "http://localhost:11434")
        self.browser_use_api_key = self.settings_manager.get_settings_key("browser_use_api_key", "")

        self.tts_voice = self.settings_manager.get_settings_key("voice", "af_sky")
        self.tts_desired_sample_rate = int(self.settings_manager.get_settings_key("desired_sample_rate", 24000))

        self.searxng_enabled = self.settings_manager.get_settings_key("searxng_enabled", False)

        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        self.main_screen = self.create_main_screen()
        self.settings_screen = self.create_settings_screen()

        self.stacked_widget.addWidget(self.main_screen)
        self.stacked_widget.addWidget(self.settings_screen)

        # Initialize Chat Panel
        self.panel_height = 250
        self.chat_visible = False

        self.chat_panel = self.create_chat_panel()
        self.chat_panel.setGeometry(0, self.height(), self.width(), self.panel_height)
        self.chat_panel.hide()

        # Animation setup for the chat panel
        self.animation = QtCore.QPropertyAnimation(self.chat_panel, b"geometry")
        self.animation.setDuration(300)
        self.animation.setEasingCurve(QtCore.QEasingCurve.OutCubic)

        # Initialize separate timers for agent and STT queues
        self.agent_timer = QtCore.QTimer()
        self.agent_timer.timeout.connect(self.listen_to_agent_responses)
        self.agent_timer.start(200)  # check every 200ms

        self.stt_timer = QtCore.QTimer()
        self.stt_timer.timeout.connect(self.listen_to_stt_transcriptions)
        self.stt_timer.start(200)  # check every 200ms

    ###########################################################################
    # Screen creation
    ###########################################################################
    def create_main_screen(self):
        main_screen = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(16, 16, 16, 16)
        main_screen.setLayout(layout)

        # -------------------------
        # Settings Button
        # -------------------------
        settings_button = QPushButton()
        settings_button.setIcon(
            QColoredSVGIcon("assets/settings.svg", QtGui.QColor("lightgray"))
        )
        settings_button.setFixedSize(32, 32)
        settings_button.setIconSize(settings_button.size() * 0.75)
        settings_button.setCursor(
            QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        )
        settings_button.setStyleSheet(
            """
                QPushButton {
                    color: white;
                    border-radius: 8px;
                }
                QPushButton:hover {
                    background-color: #282828;
                }
            """
        )
        settings_button.clicked.connect(self.open_settings)

        top_layout = QHBoxLayout()
        top_layout.addStretch()
        top_layout.addWidget(settings_button)

        # -------------------------
        # Animated Gradient Text inside a Scroll Area
        # -------------------------
        self.response_label = AnimatedGradientLabel("How can I help you?")
        self.response_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

        self.response_scroll_area = QScrollArea()  # Store as instance variable
        self.response_scroll_area.setWidgetResizable(False)
        self.response_scroll_area.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.response_scroll_area.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.response_scroll_area.setFrameShape(QFrame.NoFrame)
        self.response_scroll_area.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Fixed
        )
        self.response_scroll_area.setFixedHeight(50)  # Adjust height as needed

        self.response_scroll_area.setWidget(self.response_label)

        # -------------------------
        # Layout to Make Scroll Area 80% Width
        # -------------------------
        scroll_container_layout = QHBoxLayout()
        scroll_container_layout.setContentsMargins(0, 0, 0, 0)
        scroll_container_layout.addStretch(1)  # 10% Spacer
        scroll_container_layout.addWidget(
            self.response_scroll_area, 8
        )  # 80% Scroll Area
        scroll_container_layout.addStretch(1)  # 10% Spacer

        scroll_container = QWidget()
        scroll_container.setLayout(scroll_container_layout)
        scroll_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        # -------------------------
        # Bottom Buttons
        # -------------------------
        chat_button = QPushButton()
        chat_button.setIcon(
            QColoredSVGIcon("assets/chat_bubble.svg", QtGui.QColor("lightgray"))
        )
        chat_button.setFixedSize(50, 50)
        chat_button.setIconSize(chat_button.size() * 0.5)
        chat_button.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        chat_button.setStyleSheet(
            """
                QPushButton {
                    color: white;
                    border-radius: 25px;
                    background-color: #282828;
                }
                QPushButton:hover {
                    background-color: #2F2F2F;
                }
            """
        )
        chat_button.clicked.connect(self.toggle_chat_panel)

        self.speech_button = QPushButton()
        self.speech_button.setIcon(
            QColoredSVGIcon("assets/microphone.svg", QtGui.QColor("lightgray"))
        )
        self.speech_button.setFixedSize(50, 50)
        self.speech_button.setIconSize(self.speech_button.size() * 0.5)
        self.speech_button.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.speech_button.setStyleSheet(
            """
                QPushButton {
                    color: white;
                    border-radius: 25px;
                    background-color: #282828;
                }
                QPushButton:hover {
                    background-color: #2F2F2F;
                }
            """
        )
        # *** Modified *** Connect speech_button to toggle_voice_recording
        self.speech_button.clicked.connect(self.toggle_voice_recording)

        bottom_layout = QHBoxLayout()
        bottom_layout.addStretch()
        bottom_layout.addWidget(chat_button)
        bottom_layout.addSpacing(20)
        bottom_layout.addWidget(self.speech_button)
        bottom_layout.addStretch()

        # -------------------------
        # Assemble the Main Layout
        # -------------------------
        layout.addLayout(top_layout)
        layout.addStretch()
        layout.addWidget(scroll_container, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addStretch()
        layout.addLayout(bottom_layout)

        return main_screen

    def create_model_settings_group(self, title: str):
        container = QWidget()
        container_layout = QVBoxLayout()

        label = QLabel(title)
        label.setStyleSheet("font-weight: bold; font-size: 14px;")
        container_layout.addWidget(label)
        container_layout.addSpacing(5)

        group = QGroupBox()
        layout = QVBoxLayout()

        label_width = 120  # for uniform label widths

        # Model Provider
        model_provider_layout = QHBoxLayout()
        model_provider_label = QLabel("Model Provider:")
        model_provider_label.setFixedWidth(label_width)
        model_provider_dropdown = QComboBox()
        model_provider_dropdown.addItems(["ollama", "openai", "gemini"])
        model_provider_dropdown.setStyleSheet(
            """
                QComboBox {
                    height: 30px;
                    padding: 5px;
                }
            """
        )
        model_provider_layout.addWidget(model_provider_label)
        model_provider_layout.addWidget(model_provider_dropdown)

        # Model Name
        model_name_layout = QHBoxLayout()
        model_name_label = QLabel("Model Name:")
        model_name_label.setFixedWidth(label_width)
        model_name_input = QLineEdit()
        model_name_input.setPlaceholderText("qwen2.5-coder")
        model_name_input.setStyleSheet(
            """
                QLineEdit {
                    height: 30px;
                    padding: 5px;
                }
            """
        )
        model_name_layout.addWidget(model_name_label)
        model_name_layout.addWidget(model_name_input)

        # API Base
        api_base_layout = QHBoxLayout()
        api_base_label = QLabel("API Base:")
        api_base_label.setFixedWidth(label_width)
        api_base_input = QLineEdit()
        api_base_input.setPlaceholderText("http://localhost:11434")
        api_base_input.setStyleSheet(
            """
                QLineEdit {
                    height: 30px;
                    padding: 5px;
                }
            """
        )
        api_base_layout.addWidget(api_base_label)
        api_base_layout.addWidget(api_base_input)

        # API Key
        api_key_layout = QHBoxLayout()
        api_key_label = QLabel("API Key:")
        api_key_label.setFixedWidth(label_width)
        api_key_input = QLineEdit()
        api_key_input.setPlaceholderText("sk-123")
        api_key_input.setEchoMode(QLineEdit.Password)
        api_key_input.setStyleSheet(
            """
                QLineEdit {
                    height: 30px;
                    padding: 5px;
                }
            """
        )
        api_key_layout.addWidget(api_key_label)
        api_key_layout.addWidget(api_key_input)

        layout.addLayout(model_provider_layout)
        layout.addLayout(model_name_layout)
        layout.addLayout(api_base_layout)
        layout.addLayout(api_key_layout)
        group.setLayout(layout)

        container_layout.addWidget(group)
        container.setLayout(container_layout)

        # Store widgets for later access
        self.settings_widgets = getattr(self, "settings_widgets", {})
        self.settings_widgets[title] = {
            "model_provider": model_provider_dropdown,
            "model_name": model_name_input,
            "api_base": api_base_input,
            "api_key": api_key_input,
        }

        return container

    def create_tts_settings_group(self, title: str):
        container = QWidget()
        container_layout = QVBoxLayout()

        label = QLabel(title)
        label.setStyleSheet("font-weight: bold; font-size: 14px;")
        container_layout.addWidget(label)
        container_layout.addSpacing(5)

        group = QGroupBox()
        layout = QVBoxLayout()

        label_width = 120  # for uniform label widths

        # Voice Selection
        voice_layout = QHBoxLayout()
        voice_label = QLabel("Voice:")
        voice_label.setFixedWidth(label_width)
        voice_dropdown = QComboBox()
        voice_dropdown.setStyleSheet(
            """
                QComboBox {
                    height: 30px;
                    padding: 5px;
                }
            """
        )
        try:
            available_voices = get_available_voices()
            voice_dropdown.addItems(available_voices)
        except FileNotFoundError as e:
            logging.error(e)
            voice_dropdown.addItem("Default Voice")
            voice_dropdown.setEnabled(False)

        voice_layout.addWidget(voice_label)
        voice_layout.addWidget(voice_dropdown)

        # Desired Sample Rate
        desired_sample_rate_layout = QHBoxLayout()
        desired_sample_rate_label = QLabel("Desired Sample Rate:")
        desired_sample_rate_label.setFixedWidth(label_width)
        desired_sample_rate_input = QLineEdit()
        desired_sample_rate_input.setPlaceholderText("24000")
        desired_sample_rate_input.setStyleSheet(
            """
                QLineEdit {
                    height: 30px;
                    padding: 5px;
                }
            """
        )

        desired_sample_rate_layout.addWidget(desired_sample_rate_label)
        desired_sample_rate_layout.addWidget(desired_sample_rate_input)

        layout.addLayout(voice_layout)
        layout.addLayout(desired_sample_rate_layout)
        group.setLayout(layout)

        container_layout.addWidget(group)
        container.setLayout(container_layout)

        # Store widgets for later access
        self.settings_widgets[title] = {
            "voice_dropdown": voice_dropdown,
            "desired_sample_rate_input": desired_sample_rate_input,  # Added this line
        }

        return container

    
    def create_searxng_settings_group(self, title: str):
        container = QWidget()
        container_layout = QVBoxLayout()

        label = QLabel(title)
        label.setStyleSheet("font-weight: bold; font-size: 14px;")
        container_layout.addWidget(label)
        container_layout.addSpacing(5)

        group = QGroupBox()
        layout = QVBoxLayout()

        label_width = 120  # for uniform label widths

        # Enable Searxng
        searxng_enabled_layout = QHBoxLayout()
        searxng_enabled_label = QLabel("Enable Searxng:")
        searxng_enabled_label.setFixedWidth(label_width)
        searxng_enabled_input = QCheckBox()
        searxng_enabled_input.setStyleSheet(
            """
                QCheckBox {
                    height: 30px;
                    padding: 5px;
                }
            """
        )

        searxng_enabled_layout.addWidget(searxng_enabled_label)
        searxng_enabled_layout.addWidget(searxng_enabled_input)

        layout.addLayout(searxng_enabled_layout)
        group.setLayout(layout)

        container_layout.addWidget(group)
        container.setLayout(container_layout)

        # Store widgets for later access
        self.settings_widgets[title] = {
            "searxng_enabled": searxng_enabled_input,
        }

        return container


    def create_settings_screen(self):
        settings_screen = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(16, 16, 16, 16)
        settings_screen.setLayout(main_layout)

        # -------------------------
        # Back Button
        # -------------------------
        back_button = QPushButton()
        back_button.setIcon(
            QColoredSVGIcon("assets/back_arrow.svg", QtGui.QColor("lightgray"))
        )
        back_button.setFixedSize(28, 28)
        back_button.setIconSize(back_button.size() * 0.75)
        back_button.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        back_button.setStyleSheet(
            """
                QPushButton {
                    color: white;
                    border-radius: 8px;
                }
                QPushButton:hover {
                    background-color: #282828;
                }
            """
        )
        back_button.clicked.connect(self.open_main)

        top_layout = QHBoxLayout()
        top_layout.addWidget(back_button)
        top_layout.addStretch()

        main_layout.addLayout(top_layout)
        main_layout.addSpacing(10)

        # -------------------------
        # Scroll Area for Settings
        # -------------------------
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("QScrollArea { border: none; }")  # Optional: Remove border

        # Container widget for scroll area
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout()
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(15)  # Space between groups

        # Create separate groups for Completion, Embedding, Browser Use, TTS, and STT
        completion_model_group = self.create_model_settings_group("Completion Model")
        embedding_model_group = self.create_model_settings_group("Embedding Model")
        browser_use_model_group = self.create_model_settings_group("Browser Use Model")
        tts_group = self.create_tts_settings_group("Text-to-Speech (TTS)")
        searxng_group = self.create_searxng_settings_group("Searxng")

        # Add groups to the scroll layout
        scroll_layout.addWidget(completion_model_group)
        scroll_layout.addWidget(embedding_model_group)
        scroll_layout.addWidget(browser_use_model_group)
        scroll_layout.addWidget(tts_group)
        scroll_layout.addWidget(searxng_group)
        scroll_layout.addStretch()  # Pushes content to the top

        scroll_widget.setLayout(scroll_layout)
        scroll_area.setWidget(scroll_widget)

        main_layout.addWidget(scroll_area)

        # -------------------------
        # Save Button
        # -------------------------
        save_button = QPushButton("Save")
        save_button.setFixedSize(100, 40)
        save_button.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        save_button.setStyleSheet(
            """
                QPushButton {
                    color: white;
                    border-radius: 25px;
                    background-color: #282828;
                }
                QPushButton:hover {
                    background-color: #2F2F2F;
                }
            """
        )
        save_button.clicked.connect(self.save_settings)

        bottom_layout = QHBoxLayout()
        bottom_layout.addStretch()
        bottom_layout.addWidget(save_button)

        main_layout.addLayout(bottom_layout)

        # -------------------------
        # Initialize Settings Values
        # -------------------------
        # Set initial values for Completion Model
        completion_model_group_widgets = self.settings_widgets["Completion Model"]
        completion_provider, completion_name = self.completion_model_id.split("/", 1)
        completion_model_group_widgets["model_provider"].setCurrentText(completion_provider)
        completion_model_group_widgets["model_name"].setText(completion_name)
        completion_model_group_widgets["api_base"].setText(self.completion_api_base or "")
        completion_model_group_widgets["api_key"].setText(self.completion_api_key or "")

        # Set initial values for Embedding Model
        embedding_model_group_widgets = self.settings_widgets["Embedding Model"]
        embedding_provider, embedding_name = self.embedding_model_id.split("/", 1)
        embedding_model_group_widgets["model_provider"].setCurrentText(embedding_provider)
        embedding_model_group_widgets["model_name"].setText(embedding_name)
        embedding_model_group_widgets["api_base"].setText(self.embedding_api_base or "")
        embedding_model_group_widgets["api_key"].setText(self.embedding_api_key or "")

        # Set initial values for Browser Use Model
        browser_use_model_group_widgets = self.settings_widgets["Browser Use Model"]
        browser_provider, browser_name = self.browser_use_model_id.split("/", 1)
        browser_use_model_group_widgets["model_provider"].setCurrentText(browser_provider)
        browser_use_model_group_widgets["model_name"].setText(browser_name)
        browser_use_model_group_widgets["api_base"].setText(self.browser_use_api_base or "")
        browser_use_model_group_widgets["api_key"].setText(self.browser_use_api_key or "")

        # Set initial values for TTS Voice and Desired Sample Rate
        tts_group_widgets = self.settings_widgets["Text-to-Speech (TTS)"]
        current_tts_voice = self.tts_voice
        voice_dropdown = tts_group_widgets["voice_dropdown"]
        index = voice_dropdown.findText(current_tts_voice)
        if index != -1:
            voice_dropdown.setCurrentIndex(index)
        else:
            voice_dropdown.addItem(current_tts_voice)
            voice_dropdown.setCurrentText(current_tts_voice)
        
        # Initialize Desired Sample Rate
        desired_sample_rate_input = tts_group_widgets.get("desired_sample_rate_input")
        if desired_sample_rate_input:
            desired_sample_rate_input.setText(str(self.tts_desired_sample_rate))
        
        # Set initial state for Searxng
        searxng_group_widgets = self.settings_widgets["Searxng"]
        searxng_enabled_input = searxng_group_widgets.get("searxng_enabled")
        if searxng_enabled_input:
            searxng_enabled_input.setChecked(self.searxng_enabled)

        return settings_screen

    ###########################################################################
    # Chat Panel
    ###########################################################################
    def create_chat_panel(self):
        chat_panel = QWidget(self.main_screen)
        chat_panel.setStyleSheet(
            """
            background-color: #333333;
            border-top-left-radius: 5px;
            border-top-right-radius: 5px;
        """
        )
        chat_panel.setFixedHeight(self.panel_height)
        chat_panel.setLayout(None)

        # Close button
        self.close_button = QPushButton(chat_panel)
        self.close_button.setIcon(
            QColoredSVGIcon("assets/close.svg", QtGui.QColor("lightgray"))
        )
        self.close_button.setFixedSize(24, 24)
        self.close_button.setIconSize(self.close_button.size() * 0.75)
        self.close_button.setCursor(
            QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        )
        self.close_button.setStyleSheet(
            """
            QPushButton {
                color: white;
                border: none;
            }
            QPushButton:hover {
                background-color: #444444;
                border-radius: 12px;
            }
        """
        )
        self.close_button.clicked.connect(self.toggle_chat_panel)
        self.close_button.move(chat_panel.width() - self.close_button.width() - 16, 16)

        # Chat TextEdit
        self.chat_text = QTextEdit(chat_panel)
        self.chat_text.setPlaceholderText("Enter your message...")
        self.chat_text.setStyleSheet(
            """
            QTextEdit {
                background-color: #444444;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 8px;
            }
        """
        )
        self.chat_text.setGeometry(
            16,
            16 + self.close_button.height() + 8,
            chat_panel.width() - 32,
            self.panel_height - 16 - self.close_button.height() - 8 - 16,
        )

        # Submit Button
        self.submit_button = QPushButton(chat_panel)
        self.submit_button.setIcon(
            QColoredSVGIcon("assets/submit.svg", QtGui.QColor("lightgray"))
        )
        self.submit_button.setFixedSize(32, 32)
        self.submit_button.setIconSize(self.submit_button.size() * 0.5)
        self.submit_button.setCursor(
            QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        )
        self.submit_button.setStyleSheet(
            """
            QPushButton {
                color: white;
                border-radius: 8px;
                background-color: #282828;
            }
            QPushButton:hover {
                background-color: #2F2F2F;
            }
        """
        )
        self.submit_button.clicked.connect(self.submit_text)
        self.update_submit_button_position()

        return chat_panel

    def update_submit_button_position(self):
        if hasattr(self, "chat_text") and hasattr(self, "submit_button"):
            text_geo = self.chat_text.geometry()
            submit_x = text_geo.x() + text_geo.width() - self.submit_button.width() - 10
            submit_y = (
                text_geo.y() + text_geo.height() - self.submit_button.height() - 10
            )
            self.submit_button.move(submit_x, submit_y)

    ###########################################################################
    # Chat Logic
    ###########################################################################
    def toggle_chat_panel(self):
        if self.chat_visible:
            # Animate to hide
            start_rect = self.chat_panel.geometry()
            end_rect = QtCore.QRect(0, self.height(), self.width(), self.panel_height)
            self.animation.setStartValue(start_rect)
            self.animation.setEndValue(end_rect)
            self.animation.start()
            self.chat_visible = False
        else:
            # Show panel before animating
            self.chat_panel.show()
            self.chat_panel.setGeometry(
                0, self.height(), self.width(), self.panel_height
            )

            # Reposition child elements
            self.close_button.move(
                self.chat_panel.width() - self.close_button.width() - 16, 16
            )
            self.chat_text.setGeometry(
                16,
                16 + self.close_button.height() + 8,
                self.chat_panel.width() - 32,
                self.panel_height - 16 - self.close_button.height() - 8 - 16,
            )
            self.update_submit_button_position()

            # Animate to show
            start_rect = QtCore.QRect(0, self.height(), self.width(), self.panel_height)
            end_rect = QtCore.QRect(
                0, self.height() - self.panel_height, self.width(), self.panel_height
            )
            self.animation.setStartValue(start_rect)
            self.animation.setEndValue(end_rect)
            self.animation.start()
            self.chat_visible = True

    def submit_text(self):
        text = self.chat_text.toPlainText().strip()
        if text:
            self.agent_inbound.put(text)  # send to agent
            self.chat_text.clear()

    def listen_to_agent_responses(self):
        """
        Check if there's any new response from the agent.
        If so, display it and optionally send to TTS.
        """
        while True:
            try:
                response = self.agent_outbound.get_nowait()
                if response is None:
                    break  # Sentinel value to ignore
                logging.debug(f"UI received {response}")
                # Show in UI
                self.response_label.setText(response)
                self.response_label.adjustSize()
                # Also send to TTS if desired
                self.tts_inbound.put(response)
            except queue.Empty:
                break  # queue is empty

    def listen_to_stt_transcriptions(self):
        """
        Check if there's any new transcription from STT.
        If so, send it to the agent.
        """
        while True:
            try:
                transcription = self.stt_outbound.get_nowait()
                if transcription is None:
                    break  # Sentinel value to ignore
                logging.debug(f"Received transcription: {transcription}")
                # Send transcription to agent
                self.agent_inbound.put(transcription)
            except queue.Empty:
                break  # queue is empty

    ###########################################################################
    # Navigation / resize
    ###########################################################################
    def open_main(self):
        self.stacked_widget.setCurrentWidget(self.main_screen)

    def open_settings(self):
        self.stacked_widget.setCurrentWidget(self.settings_screen)

    def resizeEvent(self, event):
        super().resizeEvent(event)

        # Adjust the chat panel if it's visible
        if self.chat_visible:
            new_y = self.height() - self.panel_height
            self.chat_panel.setGeometry(0, new_y, self.width(), self.panel_height)
            self.close_button.move(
                self.chat_panel.width() - self.close_button.width() - 16, 16
            )
            self.chat_text.setGeometry(
                16,
                16 + self.close_button.height() + 8,
                self.chat_panel.width() - 32,
                self.panel_height - 16 - self.close_button.height() - 8 - 16,
            )
            self.update_submit_button_position()
        else:
            self.chat_panel.setGeometry(
                0, self.height(), self.width(), self.panel_height
            )

        # Dynamically adjust the width of the scroll area
        available_width = self.width() - 32  # Considering layout margins
        scroll_area_width = int(available_width * 0.8)  # 80% of available width

        self.response_scroll_area.setFixedWidth(scroll_area_width)
        self.response_scroll_area.setFixedHeight(50)  # Keep consistent height
        self.response_label.adjustSize()
        self.response_label.setFixedHeight(self.response_scroll_area.height())

    ###########################################################################
    # Settings Logic
    ###########################################################################
    def save_settings(self):
        try:
            # Retrieve Completion Model settings
            completion_widgets = self.settings_widgets["Completion Model"]
            completion_provider = completion_widgets["model_provider"].currentText()
            completion_name = completion_widgets["model_name"].text().strip()
            completion_api_base = completion_widgets["api_base"].text().strip()
            completion_api_key = completion_widgets["api_key"].text().strip()

            # Retrieve Embedding Model settings
            embedding_widgets = self.settings_widgets["Embedding Model"]
            embedding_provider = embedding_widgets["model_provider"].currentText()
            embedding_name = embedding_widgets["model_name"].text().strip()
            embedding_api_base = embedding_widgets["api_base"].text().strip()
            embedding_api_key = embedding_widgets["api_key"].text().strip()

            # Retrieve Browser Use Model settings
            browser_widgets = self.settings_widgets["Browser Use Model"]
            browser_provider = browser_widgets["model_provider"].currentText()
            browser_name = browser_widgets["model_name"].text().strip()
            browser_api_base = browser_widgets["api_base"].text().strip()
            browser_api_key = browser_widgets["api_key"].text().strip()

            # Retrieve TTS settings
            tts_widget = self.settings_widgets.get("Text-to-Speech (TTS)", {})
            tts_voice = tts_widget.get("voice_dropdown", QComboBox()).currentText()
            
            # Validate and retrieve Desired Sample Rate
            desired_sample_rate_input = tts_widget.get("desired_sample_rate_input")
            if desired_sample_rate_input:
                desired_sample_rate_str = desired_sample_rate_input.text().strip()
                if not desired_sample_rate_str.isdigit():
                    raise ValueError("Desired Sample Rate must be an integer.")
                tts_desired_sample_rate = int(desired_sample_rate_str)
            else:
                tts_desired_sample_rate = self.tts_desired_sample_rate  # Fallback to current value

            # Retrieve Searxng settings
            searxng_widget = self.settings_widgets["Searxng"]
            searxng_enabled = searxng_widget["searxng_enabled"].isChecked()

            # Construct model IDs
            completion_model_id = f"{completion_provider}/{completion_name}"
            embedding_model_id = f"{embedding_provider}/{embedding_name}"
            browser_use_model_id = f"{browser_provider}/{browser_name}"

            # Update settings in the manager
            self.settings_manager.set_settings_key("completion_model_id", completion_model_id)
            self.settings_manager.set_settings_key("completion_api_base", completion_api_base)
            self.settings_manager.set_settings_key("completion_api_key", completion_api_key)

            self.settings_manager.set_settings_key("embedding_model_id", embedding_model_id)
            self.settings_manager.set_settings_key("embedding_api_base", embedding_api_base)
            self.settings_manager.set_settings_key("embedding_api_key", embedding_api_key)

            self.settings_manager.set_settings_key("browser_use_model_id", browser_use_model_id)
            self.settings_manager.set_settings_key("browser_use_api_base", browser_api_base)
            self.settings_manager.set_settings_key("browser_use_api_key", browser_api_key)

            self.settings_manager.set_settings_key("voice", tts_voice)
            self.settings_manager.set_settings_key("desired_sample_rate", tts_desired_sample_rate)

            self.settings_manager.set_settings_key("searxng_enabled", searxng_enabled)

            # Save settings to the YAML file
            self.settings_manager.save_settings()

            # Update internal variables
            self.completion_model_id = completion_model_id
            self.completion_api_base = completion_api_base
            self.completion_api_key = completion_api_key

            self.embedding_model_id = embedding_model_id
            self.embedding_api_base = embedding_api_base
            self.embedding_api_key = embedding_api_key

            self.browser_use_model_id = browser_use_model_id
            self.browser_use_api_base = browser_api_base
            self.browser_use_api_key = browser_api_key

            self.tts_voice = tts_voice
            self.tts_desired_sample_rate = tts_desired_sample_rate

            self.searxng_enabled = searxng_enabled

            QMessageBox.information(
                self, "Settings Saved", "Settings have been saved successfully."
            )
        except ValueError as ve:
            QMessageBox.warning(self, "Invalid Input", str(ve))
        except Exception as e:
            logging.error(f"Error saving settings: {e}")
            QMessageBox.critical(self, "Error", "An unexpected error occurred while saving settings.")


    ###########################################################################
    # *** Added *** Voice Recording Logic
    ###########################################################################
    def toggle_voice_recording(self):
        if not self.is_voice_recording:
            # Start voice recording
            self.is_voice_recording = True
            self.stt_is_recording_event.set()
            # Change icon to stop.svg
            self.speech_button.setIcon(
                QColoredSVGIcon("assets/stop.svg", QtGui.QColor("lightgray"))
            )
            logging.info("Voice recording started.")
        else:
            # Stop voice recording
            self.is_voice_recording = False
            self.stt_is_recording_event.clear()
            # Change icon back to microphone.svg
            self.speech_button.setIcon(
                QColoredSVGIcon("assets/microphone.svg", QtGui.QColor("lightgray"))
            )
            logging.info("Voice recording stopped.")

###############################################################################
# Main entry point
###############################################################################
def main():
    # Initialize SettingsManager
    settings_manager = SettingsManager()

    # Create shared queues
    agent_inbound_queue = queue.Queue()  # user messages from UI -> agent
    agent_outbound_queue = queue.Queue()  # agent responses from agent -> UI
    tts_inbound_queue = queue.Queue()  # text to speak from UI / agent -> TTS
    stt_outbound_queue = queue.Queue()  # speech to text from STT -> UI

    # Create control and shutdown events
    stt_is_recording_event = threading.Event()
    agent_shutdown_event = threading.Event()
    tts_shutdown_event = threading.Event()
    stt_shutdown_event = threading.Event()

    # Start the conversation agent thread
    _conversation_agent_thread = threading.Thread(
        target=conversation_agent_thread,
        args=(
            agent_inbound_queue,
            agent_outbound_queue,
            agent_shutdown_event,
            settings_manager
        ),
        daemon=False,  # Set to False to manage shutdown manually
    )
    _conversation_agent_thread.start()
    logging.info("Conversation Agent thread started.")

    # Start the TTS thread
    _tts_thread = threading.Thread(
        target=tts_thread,
        args=(
            tts_inbound_queue,
            tts_shutdown_event,
            settings_manager
        ),
        daemon=False,  # Set to False to manage shutdown manually
    )
    _tts_thread.start()
    logging.info("TTS thread started.")

    # Start the STT thread
    _stt_thread = threading.Thread(
        target=stt_thread,
        args=(
            stt_is_recording_event,
            stt_shutdown_event,
            stt_outbound_queue,
        ),
        kwargs={
            "vad_threshold": 0.75,
            "sample_rate": 16000,
            "silence_duration": 2.0,
            "pre_speech_length": 0.5
        },
        daemon=False,  # Set to False to manage shutdown manually
    )
    _stt_thread.start()
    logging.info("STT thread started.")

    # Initialize the QApplication
    app = QApplication(sys.argv)

    # Initialize the MainWindow with the settings manager and queues
    window = MainWindow(
        settings_manager=settings_manager,
        agent_inbound=agent_inbound_queue,
        agent_outbound=agent_outbound_queue,
        tts_inbound=tts_inbound_queue,
        stt_outbound=stt_outbound_queue,
        stt_is_recording_event=stt_is_recording_event,
    )
    window.setStyleSheet("background-color: #212121; color: white;")
    window.show()

    shutdown_lock = threading.Lock()
    shutdown_initiated = False

    def shutdown_threads():
        nonlocal shutdown_initiated
        with shutdown_lock:
            if shutdown_initiated:
                logging.debug("Shutdown already in progress. Skipping redundant call.")
                return
            shutdown_initiated = True

        logging.info("Initiating graceful shutdown...")

        # Signal the threads to stop
        agent_shutdown_event.set()
        tts_shutdown_event.set()
        stt_shutdown_event.set()

        # Send sentinel values to unblock queue.get()
        agent_inbound_queue.put(None)
        tts_inbound_queue.put(None)

        # Send sentinel to outbound_queue if another thread is consuming it
        agent_outbound_queue.put(None)
        stt_outbound_queue.put(None)

        # Wait for the threads to finish
        logging.info("Waiting for Conversation Agent thread to terminate...")
        _conversation_agent_thread.join(timeout=30)
        if _conversation_agent_thread.is_alive():
            logging.warning(
                "Conversation Agent thread did not terminate within the timeout."
            )

        logging.info("Waiting for TTS thread to terminate...")
        _tts_thread.join(timeout=30)
        if _tts_thread.is_alive():
            logging.warning("TTS thread did not terminate within the timeout.")

        logging.info("Waiting for STT thread to terminate...")
        _stt_thread.join(timeout=30)
        if _stt_thread.is_alive():
            logging.warning("STT thread did not terminate within the timeout.")

        logging.info("All threads have been shut down.")

    # Connect the shutdown function to the application's aboutToQuit signal
    app.aboutToQuit.connect(shutdown_threads)

    # Run the Qt event loop
    try:
        exit_code = app.exec()
    except Exception as e:
        logging.error(f"Exception during QApplication execution: {e}")
        exit_code = 1
    finally:
        # Ensure shutdown is called if app.exec() raises an exception
        shutdown_threads()
        sys.exit(exit_code)

if __name__ == "__main__":
    main()
