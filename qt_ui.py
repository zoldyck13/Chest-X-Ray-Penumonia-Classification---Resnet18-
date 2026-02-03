import sys
import torch
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QFileDialog, QFrame, QGraphicsDropShadowEffect)
from PyQt5.QtGui import QPixmap, QImage, QFont, QColor
from PyQt5.QtCore import Qt
from PIL import Image

from src.inference import load_model, get_transform, GradCAM, overlay_heatmap

class PneumoniaDetectionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = "runs/pneumonia_resnet18/best_resnet18.pt" 
        self.model = load_model(self.model_path, self.device)
        
        self.initUI()

    def apply_shadow(self, widget):
        """Add a professional shadow effect to widgets"""
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setXOffset(0)
        shadow.setYOffset(4)
        shadow.setColor(QColor(0, 0, 0, 50))
        widget.setGraphicsEffect(shadow)

    def initUI(self):
        # Window Settings
        self.setWindowTitle('Medical Vision AI - Pneumonia Diagnosis')
        self.resize(1100, 750)
        self.setStyleSheet("""
            QWidget {
                background-color: #f5f7fa;
                font-family: 'Segoe UI', Arial;
            }
            QFrame#ImageCard {
                background-color: white;
                border-radius: 15px;
                border: 1px solid #dcdfe6;
            }
            QLabel#TitleLabel {
                font-size: 28px;
                font-weight: bold;
                color: #2c3e50;
                margin-bottom: 10px;
            }
            QPushButton#UploadBtn {
                background-color: #3498db;
                color: white;
                font-size: 18px;
                font-weight: bold;
                border-radius: 10px;
                padding: 10px;
            }
            QPushButton#UploadBtn:hover {
                background-color: #2980b9;
            }
            QLabel#ResultLabel {
                background-color: #ebf5fb;
                border-radius: 8px;
                padding: 15px;
                color: #34495e;
            }
        """)

        # Main vertical layout
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(30, 30, 30, 30)
        self.main_layout.setSpacing(20)

        # Header Section
        self.title_label = QLabel("Chest X-Ray Analysis System")
        self.title_label.setObjectName("TitleLabel")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(self.title_label)

        # Content Layout (Images)
        self.content_layout = QHBoxLayout()
        self.content_layout.setSpacing(25)

        # Original Image Card
        self.orig_card = QFrame()
        self.orig_card.setObjectName("ImageCard")
        self.apply_shadow(self.orig_card)
        orig_vbox = QVBoxLayout(self.orig_card)
        self.label_orig = QLabel("Original Scan")
        self.label_orig.setAlignment(Qt.AlignCenter)
        self.label_orig.setFixedSize(480, 480)
        self.label_orig.setStyleSheet("color: #95a5a6; font-size: 16px;")
        orig_vbox.addWidget(self.label_orig)
        
        # Grad-CAM Card
        self.grad_card = QFrame()
        self.grad_card.setObjectName("ImageCard")
        self.apply_shadow(self.grad_card)
        grad_vbox = QVBoxLayout(self.grad_card)
        self.label_grad = QLabel("AI Heatmap Explanation")
        self.label_grad.setAlignment(Qt.AlignCenter)
        self.label_grad.setFixedSize(480, 480)
        self.label_grad.setStyleSheet("color: #95a5a6; font-size: 16px;")
        grad_vbox.addWidget(self.label_grad)

        self.content_layout.addWidget(self.orig_card)
        self.content_layout.addWidget(self.grad_card)
        self.main_layout.addLayout(self.content_layout)

        # Result Section
        self.label_result = QLabel("Status: Please upload an image to start diagnosis.")
        self.label_result.setObjectName("ResultLabel")
        self.label_result.setAlignment(Qt.AlignCenter)
        self.label_result.setFont(QFont('Segoe UI', 16))
        self.main_layout.addWidget(self.label_result)

        # Bottom Button
        self.btn_upload = QPushButton("  Open X-Ray Image  ")
        self.btn_upload.setObjectName("UploadBtn")
        self.btn_upload.setCursor(Qt.PointingHandCursor)
        self.btn_upload.clicked.connect(self.run_analysis)
        self.main_layout.addWidget(self.btn_upload, alignment=Qt.AlignCenter)

    def run_analysis(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg)")
        if not file_path: return

        # Set image with rounded corners look (simplified by fixed size)
        pixmap = QPixmap(file_path)
        self.label_orig.setPixmap(pixmap.scaled(480, 480, Qt.KeepAspectRatio, Qt.SmoothTransformation))

        # Model processing
        pil_img = Image.open(file_path).convert("RGB")
        input_tensor = get_transform()(pil_img).unsqueeze(0).to(self.device)
        original_bgr = cv2.imread(file_path)

        cam_engine = GradCAM(self.model, self.model.layer4)
        logits = self.model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
        pred_idx = int(np.argmax(probs))
        
        heatmap, _ = cam_engine(input_tensor, pred_idx)
        overlay, _ = overlay_heatmap(original_bgr, heatmap)
        
        cam_engine.hook_a.remove()
        cam_engine.hook_g.remove()

        # Update Result UI
        classes = {0: "NORMAL", 1: "PNEUMONIA"}
        res_text = classes[pred_idx]
        conf = probs[pred_idx] * 100
        color = "#e74c3c" if pred_idx == 1 else "#27ae60"
        
        self.label_result.setText(f"DIAGNOSIS: {res_text}   |   CONFIDENCE: {conf:.2f}%")
        self.label_result.setStyleSheet(f"background-color: {color}; color: white; border-radius: 8px; padding: 15px; font-weight: bold;")

        # Display Heatmap
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        h, w, ch = overlay_rgb.shape
        q_img = QImage(overlay_rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.label_grad.setPixmap(QPixmap.fromImage(q_img).scaled(480, 480, Qt.KeepAspectRatio, Qt.SmoothTransformation))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    # Set high DPI scaling for modern screens
    app.setAttribute(Qt.AA_EnableHighDpiScaling)
    window = PneumoniaDetectionApp()
    window.show()
    sys.exit(app.exec_())
