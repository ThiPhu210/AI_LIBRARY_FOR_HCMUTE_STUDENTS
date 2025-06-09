import cv2
from pyzbar.pyzbar import decode
import time
import re

def is_valid_isbn(code):
    # Remove any non-digit characters
    digits = re.sub(r'\D', '', code)
    # Check if it's a valid ISBN-10 or ISBN-13
    return len(digits) in [10, 13]

def preprocess_frame(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY, 11, 2)
    return thresh

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
used_codes = []

camera = True
while camera == True:
    success, frame = cap.read()
    
    if not success:
        print("Không thể kết nối với camera")
        continue
        
    if frame is None:
        print("Không nhận được hình ảnh")
        continue
        
    try:
        # Preprocess the frame
        processed_frame = preprocess_frame(frame)
        
        # Try to decode from both original and processed frames
        decoded_objects = decode(frame) or decode(processed_frame)
        
        for code in decoded_objects:
            try:
                isbn = code.data.decode('utf-8')
                if is_valid_isbn(isbn):
                    if isbn not in used_codes:
                        print('Mã ISBN hợp lệ!')
                        print(f'ISBN: {isbn}')
                        used_codes.append(isbn)
                        time.sleep(5)
                    else:
                        print('Mã ISBN này đã được sử dụng!')
                        time.sleep(5)
                else:
                    print('Không phải mã ISBN hợp lệ')
            except UnicodeDecodeError:
                print('Không thể đọc mã vạch')
                continue
    except Exception as e:
        print(f"Lỗi khi xử lý mã: {e}")
        
    cv2.imshow('Quét mã ISBN', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
# import cv2
# print(cv2.getBuildInformation())
