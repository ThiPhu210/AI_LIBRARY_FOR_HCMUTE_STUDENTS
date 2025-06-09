from django.shortcuts import render, redirect
from django.http import StreamingHttpResponse, HttpResponse, JsonResponse
from django.conf import settings
import os
import uuid
import qrcode
import numpy as np
import cv2
from .camera import gen_frames, get_user_status, reset_user_status
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages

# Trang chính (chọn giữa nhận diện khuôn mặt hoặc QR)
def index(request):
    return render(request, 'loginapp/index.html', {'show_qr': False})

# Khi bấm nút "Đăng nhập bằng mã QR"
def show_qr(request):
    if request.method == 'POST':
        token = str(uuid.uuid4())
        qr_url = f"http://localhost:8000/qr-login/{token}"
        qr_img = qrcode.make(qr_url)
        qr_path = os.path.join(settings.MEDIA_ROOT, 'qrcodes', f"{token}.png")
        os.makedirs(os.path.dirname(qr_path), exist_ok=True)
        qr_img.save(qr_path)
        return render(request, 'loginapp/index.html', {
            'show_qr': True,
            'qr_url': f"/media/qrcodes/{token}.png"
        })
    return redirect('index')

def video_feed(request):
    return StreamingHttpResponse(gen_frames(),
        content_type='multipart/x-mixed-replace; boundary=frame')

# Trang trung gian kiểm tra nếu nhận diện xong thì chuyển
def check_login_status(request):
    user, login_success = get_user_status()
    if user and login_success:
        try:
            # Kiểm tra user có active không
            if user.is_active:
                # Đăng nhập user bằng auth.login giống như loginView
                login(request, user)
                # Reset trạng thái sau khi login
                reset_user_status()
                
                # Trả về URL chuyển hướng dựa trên quyền của user
                if user.is_admin or user.is_superuser:
                    return HttpResponse('/dashboard/')
                elif user.is_librarian:
                    return HttpResponse('/librarian/')
                else:
                    return HttpResponse('/publisher/')
            else:
                messages.info(request, "User account is not active")
                return HttpResponse('/')
        except Exception as e:
            print(f"Login error: {str(e)}")
            messages.error(request, f"Login error: {str(e)}")
            return HttpResponse('/')
    return HttpResponse("")

# Khi người dùng quét QR
def qr_login(request, token):
    return redirect('/libraryview/')

# Trang chủ sau khi login thành công
def home(request):
    return render(request, 'bookstore/book_list.html')

# @csrf_exempt
# def register_face(request):
#     """Register face for a user"""
#     if request.method == 'POST':
#         try:
#             # Get current face features from session
#             current_face = np.array(request.session.get('current_face', []))
#             if len(current_face) == 0:
#                 return JsonResponse({'status': 'no_face'})
            
#             # Get user
#             user = request.user
#             if not user.is_authenticated:
#                 return JsonResponse({'status': 'not_authenticated'})
            
#             # Save face encoding
#             face_encoding, created = FaceEncoding.objects.get_or_create(user=user)
#             face_encoding.set_encoding(current_face)
#             face_encoding.save()
            
#             return JsonResponse({'status': 'success'})
            
#         except Exception as e:
#             print(f"Error in face registration: {e}")
#             return JsonResponse({'status': 'error', 'message': str(e)})
    
#     return JsonResponse({'status': 'failed'})

def face_login(request):
    """View for face login page"""
    return render(request, 'loginapp/face_login.html')
