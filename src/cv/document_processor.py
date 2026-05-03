"""
Computer Vision Module — Resume Image Processing Pipeline
Handles PDF → Image conversion, preprocessing, deskewing, and OCR extraction.
"""

import io
import cv2
import numpy as np
import pytesseract
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
from pdf2image import convert_from_bytes, convert_from_path
from loguru import logger
from typing import Union
import fitz  # PyMuPDF


class DocumentProcessor:
    """
    End-to-end CV pipeline for resume documents.
    Supports scanned PDFs, digital PDFs, and images (PNG/JPG).
    """

    def __init__(self, dpi: int = 300):
        self.dpi = dpi
        self.tesseract_config = r"--oem 3 --psm 6 -l eng"

    # ──────────────────────────────────────────────────────────────────────────
    # Public entry point
    # ──────────────────────────────────────────────────────────────────────────

    def process(self, file_bytes: bytes, filename: str) -> dict:
        """
        Master method. Returns extracted text + image metadata.

        Args:
            file_bytes: raw bytes of uploaded file
            filename:   original filename (used to infer type)

        Returns:
            {
              "raw_text": str,
              "pages": int,
              "is_scanned": bool,
              "quality_score": float,   # 0-1
              "page_images": list[np.ndarray]
            }
        """
        suffix = Path(filename).suffix.lower()

        if suffix == ".pdf":
            return self._process_pdf(file_bytes)
        elif suffix in {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}:
            return self._process_image(file_bytes)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

    # ──────────────────────────────────────────────────────────────────────────
    # PDF handling
    # ──────────────────────────────────────────────────────────────────────────

    def _process_pdf(self, file_bytes: bytes) -> dict:
        """Detect digital vs scanned PDF and route accordingly."""
        # Try digital text extraction first (fast path)
        digital_text = self._extract_digital_pdf_text(file_bytes)
        is_scanned = len(digital_text.strip()) < 100  # heuristic

        if is_scanned:
            logger.info("Scanned PDF detected — running OCR pipeline")
            return self._ocr_pdf(file_bytes)
        else:
            logger.info("Digital PDF detected — using text layer")
            return {
                "raw_text": digital_text,
                "pages": self._count_pdf_pages(file_bytes),
                "is_scanned": False,
                "quality_score": 1.0,
                "page_images": [],
            }

    def _extract_digital_pdf_text(self, file_bytes: bytes) -> str:
        """Extract embedded text from a digital PDF using PyMuPDF."""
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        texts = []
        for page in doc:
            texts.append(page.get_text("text"))
        doc.close()
        return "\n".join(texts)

    def _count_pdf_pages(self, file_bytes: bytes) -> int:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        n = len(doc)
        doc.close()
        return n

    def _ocr_pdf(self, file_bytes: bytes) -> dict:
        """Convert scanned PDF pages to images and OCR each one."""
        pil_pages = convert_from_bytes(file_bytes, dpi=self.dpi)
        all_text = []
        page_imgs = []
        quality_scores = []

        for i, pil_img in enumerate(pil_pages):
            logger.debug(f"Processing page {i + 1}/{len(pil_pages)}")
            cv_img = self._pil_to_cv(pil_img)
            preprocessed, quality = self._preprocess(cv_img)
            text = self._ocr(preprocessed)
            all_text.append(text)
            page_imgs.append(preprocessed)
            quality_scores.append(quality)

        return {
            "raw_text": "\n\n".join(all_text),
            "pages": len(pil_pages),
            "is_scanned": True,
            "quality_score": float(np.mean(quality_scores)),
            "page_images": page_imgs,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Image handling
    # ──────────────────────────────────────────────────────────────────────────

    def _process_image(self, file_bytes: bytes) -> dict:
        """Process a single image file."""
        nparr = np.frombuffer(file_bytes, np.uint8)
        cv_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if cv_img is None:
            raise ValueError("Could not decode image file.")
        preprocessed, quality = self._preprocess(cv_img)
        text = self._ocr(preprocessed)
        return {
            "raw_text": text,
            "pages": 1,
            "is_scanned": True,
            "quality_score": quality,
            "page_images": [preprocessed],
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Core CV preprocessing pipeline
    # ──────────────────────────────────────────────────────────────────────────

    def _preprocess(self, img: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Full preprocessing pipeline:
          1. Grayscale
          2. Deskew
          3. Denoise
          4. Adaptive threshold (binarisation)
          5. Morphological cleanup
        Returns preprocessed image and quality score.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        deskewed = self._deskew(gray)
        denoised = cv2.fastNlMeansDenoising(deskewed, h=10)
        binary = cv2.adaptiveThreshold(
            denoised, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        # Morphological close to fix broken characters
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        quality = self._estimate_quality(cleaned)
        return cleaned, quality

    def _deskew(self, gray: np.ndarray) -> np.ndarray:
        """Correct skew angle using Hough line transform."""
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=100,
            minLineLength=100, maxLineGap=10
        )
        if lines is None:
            return gray

        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 != x1:
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                if -45 < angle < 45:
                    angles.append(angle)

        if not angles:
            return gray

        median_angle = np.median(angles)
        if abs(median_angle) < 0.5:
            return gray  # negligible skew

        (h, w) = gray.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(
            gray, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        logger.debug(f"Deskewed by {median_angle:.2f}°")
        return rotated

    def _estimate_quality(self, binary: np.ndarray) -> float:
        """
        Estimate image quality using Laplacian variance (sharpness proxy).
        Returns 0.0 (very blurry) to 1.0 (sharp).
        """
        lap_var = cv2.Laplacian(binary, cv2.CV_64F).var()
        # Clamp to sensible range for resume scans
        score = min(lap_var / 500.0, 1.0)
        return round(float(score), 4)

    # ──────────────────────────────────────────────────────────────────────────
    # OCR
    # ──────────────────────────────────────────────────────────────────────────

    def _ocr(self, preprocessed: np.ndarray) -> str:
        """Run Tesseract OCR on a preprocessed grayscale image."""
        pil_img = Image.fromarray(preprocessed)
        text = pytesseract.image_to_string(pil_img, config=self.tesseract_config)
        return self._clean_ocr_text(text)

    @staticmethod
    def _clean_ocr_text(text: str) -> str:
        """Post-process OCR output: remove artifacts, normalize whitespace."""
        import re
        # Remove non-printable characters
        text = re.sub(r"[^\x20-\x7E\n]", " ", text)
        # Collapse multiple spaces
        text = re.sub(r" {2,}", " ", text)
        # Collapse >2 consecutive newlines
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    # ──────────────────────────────────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _pil_to_cv(pil_img: Image.Image) -> np.ndarray:
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
