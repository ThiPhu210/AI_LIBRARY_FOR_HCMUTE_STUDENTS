# extract_embeddings.py
import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from facenet_pytorch import InceptionResnetV1
import torch
from torchvision import transforms

# Đường dẫn thư mục chứa ảnh mỗi người
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, 'image')

# Tạo thư mục nếu chưa tồn tại
os.makedirs(IMAGE_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

embeddings = []
names = []

def parse_directory_name(dir_name):
    """Parse directory name to extract username"""
    # Split by underscore
    parts = dir_name.split('_')
    
    # All parts except the last one form the name
    name_parts = parts[:-1]
    
    # Join name parts and convert to lowercase for username
    username = ''.join(name_parts).lower()
    
    return username

# Kiểm tra xem thư mục có tồn tại và có ảnh không
if not os.path.exists(IMAGE_DIR):
    print(f"Thư mục {IMAGE_DIR} không tồn tại. Vui lòng tạo thư mục và thêm ảnh vào.")
    exit(1)

if not os.listdir(IMAGE_DIR):
    print(f"Thư mục {IMAGE_DIR} trống. Vui lòng thêm ảnh vào.")
    exit(1)

for person_name in os.listdir(IMAGE_DIR):
    person_path = os.path.join(IMAGE_DIR, person_name)
    if not os.path.isdir(person_path):
        continue

    # Chuyển đổi tên thư mục thành username
    username = parse_directory_name(person_name)
    print(f"\nĐang xử lý ảnh của {username}...")
    
    for image_name in tqdm(os.listdir(person_path), desc=username):
        image_path = os.path.join(person_path, image_name)
        try:
            img = Image.open(image_path).convert('RGB')
            img = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                embedding = model(img).cpu().numpy()
            embeddings.append(embedding[0])
            names.append(username)  # Lưu username thay vì tên thư mục
        except Exception as e:
            print(f"Lỗi xử lý {image_path}: {e}")

if not embeddings:
    print("Không tìm thấy ảnh hợp lệ nào để xử lý.")
    exit(1)

# Lưu file
output_dir = os.path.join(BASE_DIR, 'data')
os.makedirs(output_dir, exist_ok=True)

np.save(os.path.join(output_dir, "known_embeddings.npy"), np.array(embeddings))
np.save(os.path.join(output_dir, "known_names.npy"), np.array(names))
print(f"\nĐã lưu known_embeddings.npy và known_names.npy vào thư mục {output_dir}")
print("\nDanh sách username đã lưu:")
for name in set(names):
    print(f"- {name}")
