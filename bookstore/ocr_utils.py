import cv2
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import torch
from paddleocr import PaddleOCR, draw_ocr
import fitz  # PyMuPDF
from datetime import datetime as timezone

FONT = 'latin.ttf'

def noise_removal(image):
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, 3)
    return image

def thin_font(image):
    image = cv2.bitwise_not(image)
    kernel = np.ones((2, 2), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return image

def thick_font(image):
    image = cv2.bitwise_not(image)
    kernel = np.ones((2, 2), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return image

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    inverted = cv2.bitwise_not(gray)
    no_noise = noise_removal(inverted)
    thickened = thick_font(no_noise)
    _, thresh = cv2.threshold(thickened, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def init_ocr_models():
    """
    Khởi tạo các model OCR.
    :return: Tuple (detector, recognitor)
    """
    try:
        # Khởi tạo PaddleOCR cho detection
        detector = PaddleOCR(
            use_angle_cls=False,
            lang="vi",
            use_gpu=False  # Sử dụng CPU
        )

        # Khởi tạo VietOCR cho recognition
        config = Cfg.load_config_from_name('vgg_transformer')
        config['cnn']['pretrained'] = True
        config['predictor']['beamsearch'] = True
        config['device'] = 'cpu'  # Sử dụng CPU
        recognitor = Predictor(config)

        return detector, recognitor
    except Exception as e:
        print(f"Lỗi khởi tạo model OCR: {str(e)}")
        raise

def predict(recognitor, detector, img_path, save_path, padding=4, dpi=100):
    """
    Dự đoán và vẽ kết quả OCR lên ảnh.
    :param recognitor: Mô hình nhận diện ký tự
    :param detector: Mô hình phát hiện vùng văn bản
    :param img_path: Đường dẫn ảnh đầu vào
    :param save_path: Đường dẫn lưu ảnh kết quả
    :param padding: Padding quanh box
    :param dpi: Độ phân giải ảnh
    :return: Tuple (boxes, texts)
    """
    try:
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Không thể đọc ảnh từ đường dẫn: {img_path}")

        # Text detection
        result = detector.ocr(img_path, cls=False, det=True, rec=False)
        if not result or not result[0]:
            raise ValueError("Không tìm thấy vùng văn bản nào trong ảnh")
        result = result[:][:][0]

        # Filter Boxes
        boxes = []
        for line in result:
            boxes.append([[int(line[0][0]), int(line[0][1])], [int(line[2][0]), int(line[2][1])]])
        boxes = boxes[::-1]

        # Add padding to boxes
        h, w = img.shape[:2]
        for box in boxes:
            box[0][0] = max(0, box[0][0] - padding)
            box[0][1] = max(0, box[0][1] - padding)
            box[1][0] = min(w, box[1][0] + padding)
            box[1][1] = min(h, box[1][1] + padding)

        # Text recognition
        texts = []
        for box in boxes:
            try:
                cropped_image = img[box[0][1]:box[1][1], box[0][0]:box[1][0]]
                if cropped_image.size == 0:
                    continue
                    
                cropped_image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
                rec_result = recognitor.predict(cropped_image)
                texts.append(rec_result)
            except Exception as e:
                print(f"Lỗi khi nhận diện văn bản: {str(e)}")
                continue

        # Convert boxes to draw
        def get_rectangle_points(x1, y1, x2, y2):
            return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            
        _boxes = [get_rectangle_points(box[0][0], box[0][1], box[1][0], box[1][1]) for box in boxes]

        # Draw boxes and texts
        img = draw_ocr(img, _boxes, texts, scores=None, font_path=FONT)

        # Save image
        os.makedirs(save_path, exist_ok=True)
        img_name = os.path.basename(img_path)
        cv2.imwrite(os.path.join(save_path, img_name), img)

        # Display image
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        height, width = img.shape[:2]
        figsize = width / float(dpi), height / float(dpi)

        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        ax.imshow(img, cmap='gray')
        plt.show()

        return boxes, texts
    except Exception as e:
        print(f"Lỗi trong quá trình dự đoán: {str(e)}")
        return [], []

def extract_text_only(recognitor, detector, img_path, padding=4):
    """
    Trích xuất văn bản từ ảnh.
    :param recognitor: Mô hình nhận diện ký tự
    :param detector: Mô hình phát hiện vùng văn bản
    :param img_path: Đường dẫn ảnh hoặc ảnh numpy array
    :param padding: Padding quanh box
    :return: List văn bản
    """
    try:
        # Load image
        if isinstance(img_path, str):
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Không thể đọc ảnh từ đường dẫn: {img_path}")
        elif isinstance(img_path, np.ndarray):
            img = img_path.copy()
        else:
            raise ValueError(f"Đầu vào không hợp lệ: {type(img_path)}")

        # Text detection
        result = detector.ocr(img_path if isinstance(img_path, str) else img, cls=False, det=True, rec=False)
        if not result or not result[0]:
            raise ValueError("Không tìm thấy vùng văn bản nào trong ảnh")
        result = result[:][:][0]

        # Filter Boxes
        boxes = []
        for line in result:
            boxes.append([[int(line[0][0]), int(line[0][1])], [int(line[2][0]), int(line[2][1])]])
        boxes = boxes[::-1]

        # Add padding
        h, w = img.shape[:2]
        for box in boxes:
            box[0][0] = max(0, box[0][0] - padding)
            box[0][1] = max(0, box[0][1] - padding)
            box[1][0] = min(w, box[1][0] + padding)
            box[1][1] = min(h, box[1][1] + padding)

        # Recognition
        texts = []
        for box in boxes:
            try:
                cropped_image = img[box[0][1]:box[1][1], box[0][0]:box[1][0]]
                if cropped_image.size == 0:
                    continue
                    
                cropped_image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
                rec_result = recognitor.predict(cropped_image)
                if rec_result.strip():
                    texts.append(rec_result)
            except Exception as e:
                print(f"Lỗi khi nhận diện văn bản: {str(e)}")
                continue

        return texts
    except Exception as e:
        print(f"Lỗi trong quá trình trích xuất văn bản: {str(e)}")
        return []

def save_ocr_result(texts, output_dir, filename):
    """
    Lưu kết quả OCR vào file text.
    :param texts: List văn bản đã trích xuất
    :param output_dir: Thư mục lưu kết quả
    :param filename: Tên file đầu vào
    :return: Đường dẫn file kết quả
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(filename)[0]
        output_file = os.path.join(output_dir, f"{base_name}_ocr.txt")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=== Kết quả OCR ===\n\n")
            f.write(f"Thời gian: {timezone.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Tên file: {filename}\n")
            f.write("\n=== Nội dung ===\n\n")
            f.write("\n".join(texts))
            
        print(f"Đã lưu kết quả OCR vào file: {output_file}")
        return output_file
    except Exception as e:
        print(f"Lỗi khi lưu kết quả OCR: {str(e)}")
        return None

def process_image_for_ocr(image):
    """
    Xử lý ảnh trước khi OCR với các bước tối ưu.
    :param image: Ảnh đầu vào (numpy array).
    :return: Ảnh đã xử lý.
    """
    try:
        # Kiểm tra ảnh đầu vào
        if image is None:
            raise ValueError("Ảnh đầu vào không hợp lệ (None)")
        
        # Đảm bảo ảnh là numpy array
        if not isinstance(image, np.ndarray):
            raise ValueError("Ảnh đầu vào phải là numpy array")
            
        # Đảm bảo ảnh có 3 kênh màu
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif len(image.shape) == 3 and image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError(f"Định dạng ảnh không hợp lệ: shape={image.shape}")
        
        # Chuyển sang ảnh xám
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Tăng kích thước ảnh nếu quá nhỏ
        height, width = gray.shape
        if height < 1000 or width < 1000:
            scale = max(1000/height, 1000/width)
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        # Tăng độ tương phản với CLAHE
        try:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
        except Exception as e:
            print(f"Lỗi khi áp dụng CLAHE: {str(e)}")
            enhanced = gray
        
        # Khử nhiễu với nhiều phương pháp
        try:
            # 1. Gaussian blur để làm mịn
            blurred = cv2.GaussianBlur(enhanced, (3,3), 0)
            
            # 2. Non-local means denoising
            denoised = cv2.fastNlMeansDenoising(blurred, None, 10, 7, 21)
            
            # 3. Bilateral filter để giữ cạnh
            filtered = cv2.bilateralFilter(denoised, 9, 75, 75)
        except Exception as e:
            print(f"Lỗi khi khử nhiễu: {str(e)}")
            filtered = enhanced
        
        # Nhị phân hóa thích ứng
        try:
            binary = cv2.adaptiveThreshold(
                filtered,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11,
                2
            )
        except Exception as e:
            print(f"Lỗi khi nhị phân hóa: {str(e)}")
            _, binary = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Loại bỏ nhiễu nhỏ
        try:
            kernel = np.ones((2,2), np.uint8)
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            
            # Làm dày chữ
            thickened = cv2.dilate(cleaned, kernel, iterations=1)
        except Exception as e:
            print(f"Lỗi khi xử lý morphology: {str(e)}")
            thickened = binary
        
        # Chuyển về ảnh 3 kênh màu
        try:
            result = cv2.cvtColor(thickened, cv2.COLOR_GRAY2BGR)
        except Exception as e:
            print(f"Lỗi khi chuyển đổi màu: {str(e)}")
            result = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        
        return result
    except Exception as e:
        print(f"Lỗi khi xử lý ảnh: {str(e)}")
        # Trả về ảnh gốc nếu có lỗi
        if isinstance(image, np.ndarray):
            if len(image.shape) == 2:
                return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            return image
        raise ValueError("Không thể xử lý ảnh đầu vào") 