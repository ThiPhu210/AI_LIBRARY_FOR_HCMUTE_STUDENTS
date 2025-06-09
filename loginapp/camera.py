import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import os
from bookstore.models import User
from django.contrib import auth
from django.shortcuts import redirect

# Lấy đường dẫn thư mục hiện tại
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Thêm biến toàn cục
current_user = None
login_success = False

def recognize_face(frame):
    global current_user, login_success
    boxes, _ = mtcnn.detect(frame)
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = [int(b) for b in box]
            face = frame[y1:y2, x1:x2]
            try:
                # Chuyển đổi ảnh khuôn mặt thành tensor
                face_img = Image.fromarray(face).convert('RGB').resize((160, 160))
                face_tensor = torch.tensor(np.array(face_img)).permute(2, 0, 1).float() / 255
                face_tensor = (face_tensor - 0.5) / 0.5
                face_tensor = face_tensor.unsqueeze(0).to(device)

                # Trích xuất đặc trưng khuôn mặt
                with torch.no_grad():
                    current_embedding = resnet(face_tensor).cpu().numpy()

                # Load known embeddings và names từ file numpy
                known_embeddings = np.load(os.path.join(DATA_DIR, 'known_embeddings.npy'))
                known_names = np.load(os.path.join(DATA_DIR, 'known_names.npy'))

                # So sánh với tất cả embeddings đã biết
                similarities = cosine_similarity(current_embedding, known_embeddings)
                best_idx = np.argmax(similarities)
                
                if similarities[0][best_idx] > 0.85:
                    username = known_names[best_idx]
                    # Kiểm tra xem username có tồn tại trong database không
                    try:
                        user = User.objects.get(username=username)
                        current_user = user  # Lưu user để đăng nhập
                        name = username  # Hiển thị username thay vì tên đầy đủ
                        login_success = True  # Đánh dấu đăng nhập thành công
                    except User.DoesNotExist:
                        name = "Unknown"
                        login_success = False
                else:
                    name = "Unknown"
                    login_success = False

                # Vẽ khung và tên
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, name, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            except Exception as e:
                print("Face processing error:", e)
    return frame

def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)
        frame = recognize_face(frame)
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    cap.release()

def get_user_status():
    return current_user, login_success

def reset_user_status():
    global current_user, login_success
    current_user = None
    login_success = False
