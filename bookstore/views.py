from django.shortcuts import redirect, render
from django.contrib.messages.views import SuccessMessageMixin
from django.urls import reverse_lazy
from django.views import generic
from bootstrap_modal_forms.mixins import PassRequestMixin
from .recommend.transformer import RecTransformer
from .recommend.embeddings import get_book_embedding
from .models import User, Book, Chat, DeleteRequest, Feedback, UserBookInteraction, Department
from django.contrib import messages
from django.db.models import Sum
from django.views.generic import CreateView, DetailView, DeleteView, UpdateView, ListView
from .forms import ChatForm, BookForm, UserForm, ISBNForm
from .utils import fetch_book_info_from_isbn
from . import models
import operator
import itertools
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from django.contrib.auth import authenticate, logout, login
from django.contrib import auth, messages
from django.contrib.auth.hashers import make_password
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib.auth.decorators import login_required
from django.utils import timezone
from django.shortcuts import render, redirect
from django.contrib import messages
from .models import Book
import cv2
import numpy as np
import torch
from paddleocr import PaddleOCR
from PIL import Image
import requests
import tempfile
import qrcode
from django.http import HttpResponse, JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.cache import cache
import os
import uuid
from django.conf import settings
from .recommend.recommendation_service import get_recommendation_service
import json
import base64
from pyzbar.pyzbar import decode
from django.views.decorators.http import require_http_methods
from django.core.files import File
import fitz  # PyMuPDF
import io
import time
from .ocr_utils import init_ocr_models, extract_text_only, process_image_for_ocr, save_ocr_result

# Custom mixin để kiểm tra quyền truy cập
class RoleRequiredMixin:
    def dispatch(self, request, *args, **kwargs):
        if not request.user.is_authenticated:
            return redirect('home')
            
        if hasattr(self, 'required_role'):
            if self.required_role == 'admin' and not (request.user.is_admin or request.user.is_superuser):
                messages.error(request, 'Bạn không có quyền truy cập trang này')
                return redirect('home')
            elif self.required_role == 'librarian' and not request.user.is_librarian:
                messages.error(request, 'Bạn không có quyền truy cập trang này')
                return redirect('home')
            elif self.required_role == 'publisher' and not request.user.is_publisher:
                messages.error(request, 'Bạn không có quyền truy cập trang này')
                return redirect('home')
                
        return super().dispatch(request, *args, **kwargs)

# Khởi tạo model OCR một lần khi server khởi động
detector, recognitor = init_ocr_models()

def noise_removal(image):
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, 3)
    return image

def thin_font(image):
    image = cv2.bitwise_not(image)
    kernel = np.ones((2,2),np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    return cv2.bitwise_not(image)

def thick_font(image):
    image = cv2.bitwise_not(image)
    kernel = np.ones((2,2),np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    return cv2.bitwise_not(image)

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh



# Shared Views
def login_form(request):
    return render(request, 'bookstore/login.html')


def logoutView(request):
    logout(request)
    return redirect('home')


def loginView(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        if user is not None and user.is_active:
            auth.login(request, user)
            if user.is_admin or user.is_superuser:
                return redirect('dashboard')
            elif user.is_librarian:
                return redirect('librarian')
            elif user.is_publisher:
                return redirect('publisher')
            elif user.is_student:
                return redirect('library_home')
            else:
                messages.error(request, "Tài khoản của bạn không có quyền truy cập")
                return redirect('home')
        else:
            messages.error(request, "Tên đăng nhập hoặc mật khẩu không đúng")
            return redirect('home')


def register_form(request):
    return render(request, 'bookstore/register.html')


def registerView(request):
    if request.method == 'POST':
        username = request.POST['username']
        email = request.POST['email']
        password = request.POST['password']
        password = make_password(password)

        a = User(username=username, email=email, password=password)
        a.save()
        messages.success(request, 'Account was created successfully')
        return redirect('home')
    else:
        messages.error(request, 'Registration fail, try again later')
        return redirect('regform')



















            


# Publisher views
@login_required
def publisher(request):
    if not request.user.is_publisher:
        messages.error(request, 'Bạn không có quyền truy cập trang này')
        return redirect('home')
    return render(request, 'publisher/home.html')


@login_required
def uabook_form(request):
    return render(request, 'publisher/add_book.html')


@login_required
def uabookocr_form(request):
    return render(request, 'publisher/add_book_ocr.html')


@login_required
def uabook_ocr(request):
    print("🔍 Đang vào view uabook_ocr với path:", request.path)

    if request.method != 'POST':
        messages.warning(request, "Phương thức không hợp lệ. Vui lòng dùng form để gửi file PDF.")
        print("➡️ Redirecting: Method not POST")
        return redirect('uabookocr_form')

    if 'pdf' not in request.FILES:
        messages.error(request, 'Không có file PDF được tải lên.')
        print("➡️ Redirecting: No PDF file")
        return redirect('uabookocr_form')

    pdf_file = request.FILES['pdf']

    # Check if the request is AJAX
    is_ajax = request.headers.get('x-requested-with') == 'XMLHttpRequest'

    # Kiểm tra định dạng file
    if not pdf_file.name.lower().endswith('.pdf'):
        messages.error(request, 'Định dạng file không hợp lệ. Vui lòng tải lên file PDF.')
        print("➡️ Redirecting: Invalid file format")
        if is_ajax:
            return JsonResponse({'status': 'error', 'message': 'Định dạng file không hợp lệ. Vui lòng tải lên file PDF.'}, status=400)
        return redirect('uabookocr_form')

    try:
        # Đọc PDF file
        try:
            pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
        except Exception as e:
            raise ValueError(f"Không thể đọc file PDF: {str(e)}")
        
        # Kiểm tra số trang
        if pdf_document.page_count == 0:
            raise ValueError("File PDF không có trang nào")
        
        # Lấy trang đầu tiên
        first_page = pdf_document[0]
        
        # Chuyển trang đầu tiên thành ảnh với độ phân giải cao
        try:
            pix = first_page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
        except Exception as e:
            raise ValueError(f"Không thể chuyển đổi trang PDF thành ảnh: {str(e)}")
        
        # Chuyển đổi pixmap thành PIL Image
        try:
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        except Exception as e:
            raise ValueError(f"Không thể xử lý ảnh: {str(e)}")

        # Chuyển PIL Image sang OpenCV format
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # Xử lý ảnh trước khi OCR
        processed_img = process_image_for_ocr(img_cv)
        
        # Thực hiện OCR
        texts = extract_text_only(recognitor, detector, processed_img)
        
        if not texts:
            raise ValueError("Không thể nhận dạng được văn bản trong ảnh bìa. Vui lòng thử lại với ảnh rõ nét hơn.")

        # Lưu kết quả OCR
        ocr_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ocr_results')
        timestamp = timezone.now().strftime('%Y%m%d_%H%M%S')
        ocr_file = save_ocr_result(texts, ocr_dir, f"{timestamp}_{pdf_file.name}")

        # Gọi API để xử lý văn bản
        try:
            response = requests.post(
                "https://zep.hcmute.fit/7889/extract_book_info",
                json={"ocr_text": "\n".join(texts)},
                timeout=60
            )
            response.raise_for_status()
            data = response.json()
            title = data.get("title", "Không rõ")
            author = data.get("author", "Không rõ")

        except requests.exceptions.RequestException as e:
            raise ValueError(f"Lỗi khi gọi server LLM: {str(e)}")
        except Exception as e:
            raise ValueError(f"Lỗi không xác định khi xử lý văn bản: {str(e)}")

        # Tạo book object
        try:
            book = Book(
                title=title,
                author=author,
                uploaded_by=request.user.username,
                user_id=request.user.id
            )

            # Lưu ảnh bìa
            img_io = io.BytesIO()
            img.save(img_io, format='JPEG', quality=95)
            img_io.seek(0)
            book.cover.save(f"{title}_cover.jpg", File(img_io), save=False)

            # Lưu file PDF
            pdf_file.seek(0)
            book.pdf.save(f"{title}.pdf", File(pdf_file), save=False)

            # Lưu book vào database
            book.save()

            messages.success(request, 'Sách đã được thêm thành công')
            return redirect('publisher')

        except Exception as e:
            raise ValueError(f"Lỗi khi lưu thông tin sách: {str(e)}")

    except ValueError as ve:
        error_message = str(ve)
        print(f"❌ Lỗi xử lý: {error_message}")
        messages.error(request, f'Lỗi: {error_message}')
        if is_ajax:
            return JsonResponse({'status': 'error', 'message': error_message}, status=400)
        return redirect('uabookocr_form')
    except Exception as e:
        error_message = f"Lỗi không xác định: {str(e)}"
        print(f"❌ {error_message}")
        messages.error(request, error_message)
        if is_ajax:
            return JsonResponse({'status': 'error', 'message': error_message}, status=500)
        return redirect('uabookocr_form')


@login_required
def request_form(request):
    return render(request, 'publisher/delete_request.html')


@login_required
def feedback_form(request):
    return render(request, 'publisher/send_feedback.html')

@login_required
def about(request):
    return render(request, 'publisher/about.html')	


@login_required
def usearch(request):
    query = request.GET.get('query', '')
    search_type = request.GET.get('search_type', 'title')
    
    if not query:
        return redirect('publisher')
        
    try:
        if search_type == 'title':
            books = Book.objects.filter(title__icontains=query)
        elif search_type == 'author':
            books = Book.objects.filter(author__icontains=query)
        elif search_type == 'department':
            books = Book.objects.filter(department__name__icontains=query)
        else:
            books = Book.objects.none()
            
        # Phân trang kết quả
        page = request.GET.get('page', 1)
        paginator = Paginator(books, 10)
        try:
            books = paginator.page(page)
        except PageNotAnInteger:
            books = paginator.page(1)
        except EmptyPage:
            books = paginator.page(paginator.num_pages)
            
        word = f"Kết quả tìm kiếm cho '{query}'"
        return render(request, 'publisher/result.html', {
            'files': books,
            'word': word,
            'query': query,
            'search_type': search_type
        })
        
    except Exception as e:
        messages.error(request, f'Lỗi khi tìm kiếm: {str(e)}')
        return redirect('publisher')



@login_required
def delete_request(request):
    if request.method == 'POST':
        book_id = request.POST['delete_request']
        current_user = request.user
        user_id = current_user.id
        username = current_user.username
        user_request = username + " muốn xóa sách có mã " + book_id

        a = DeleteRequest(delete_request=user_request)
        a.save()
        messages.success(request, 'Yêu cầu đã được gửi')
        return redirect('request_form')
    else:
        messages.error(request, 'Yêu cầu không được gửi')
        return redirect('request_form')



@login_required
def send_feedback(request):
    if request.method == 'POST':
        feedback = request.POST['feedback']
        current_user = request.user
        user_id = current_user.id
        username = current_user.username
        feedback = username + " " + " says " + feedback

        a = Feedback(feedback=feedback)
        a.save()
        messages.success(request, 'Feedback was sent')
        return redirect('feedback_form')
    else:
        messages.error(request, 'Feedback was not sent')
        return redirect('feedback_form')


























class UBookListView(LoginRequiredMixin, RoleRequiredMixin, ListView):
    required_role = 'publisher'
    model = Book
    template_name = 'publisher/book_list.html'
    context_object_name = 'books'
    paginate_by = 100

    def get_queryset(self):
        return Book.objects.order_by('-id')

@login_required
def uabook(request):
    if request.method == 'POST':
        title = request.POST['title']
        author = request.POST['author']
        year = request.POST['year']
        publisher = request.POST['publisher']
        desc = request.POST['desc']
        cover = request.FILES['cover']
        pdf = request.FILES['pdf']
        current_user = request.user
        user_id = current_user.id
        username = current_user.username

        a = Book(title=title, author=author, year=year, publisher=publisher, 
            desc=desc, cover=cover, pdf=pdf, uploaded_by=username, user_id=user_id)
        a.save()
        messages.success(request, 'Book was uploaded successfully')
        return redirect('publisher')
    else:
        messages.error(request, 'Book was not uploaded successfully')
        return redirect('uabook_form')	



class UCreateChat(LoginRequiredMixin, CreateView):
    form_class = ChatForm
    model = Chat
    template_name = 'publisher/chat_form.html'
    success_url = reverse_lazy('ulchat')


    def form_valid(self, form):
        self.object = form.save(commit=False)
        self.object.user = self.request.user
        self.object.save()
        return super().form_valid(form)


class UListChat(LoginRequiredMixin, ListView):
    model = Chat
    template_name = 'publisher/chat_list.html'

    def get_queryset(self):
        return Chat.objects.filter(posted_at__lt=timezone.now()).order_by('posted_at')































# Librarian views
@login_required
def librarian(request):
    if not request.user.is_librarian:
        messages.error(request, 'Bạn không có quyền truy cập trang này')
        return redirect('home')
    book = Book.objects.all().count()
    user = User.objects.all().count()

    context = {'book':book, 'user':user}

    return render(request, 'librarian/home.html', context)


@login_required
def labook_form(request):
    return render(request, 'librarian/add_book.html')


@login_required
def labook(request):
    if request.method == 'POST':
        title = request.POST['title']
        author = request.POST['author']
        year = request.POST['year']
        publisher = request.POST['publisher']
        desc = request.POST['desc']
        cover = request.FILES['cover']
        pdf = request.FILES['pdf']
        current_user = request.user
        user_id = current_user.id
        username = current_user.username

        a = Book(title=title, author=author, year=year, publisher=publisher, 
            desc=desc, cover=cover, pdf=pdf, uploaded_by=username, user_id=user_id)
        a.save()
        messages.success(request, 'Book was uploaded successfully')
        return redirect('llbook')
    else:
        messages.error(request, 'Book was not uploaded successfully')
        return redirect('llbook')
    
@login_required
def labookisbn_form(request):
    return render(request, 'librarian/add_book_isbn.html')

@login_required
def labookisbn(request):
    if request.method == 'POST':
        form = ISBNForm(request.POST)
        if form.is_valid():
            isbn = form.cleaned_data['isbn']
            book_data = fetch_book_info_from_isbn(isbn)
            if book_data:
                # Tạo book object
                book = Book(
                    isbn=isbn,
                    title=book_data['title'],
                    author=book_data['author'],
                    publisher=book_data['publisher'],
                    year=book_data['year'],
                    desc=book_data['desc'],
                    uploaded_by=request.user.username,
                    user_id=request.user.id
                )
                
                # Nếu có URL ảnh bìa, tải về và lưu
                if book_data.get('cover_url'):
                    try:
                        response = requests.get(book_data['cover_url'])
                        if response.status_code == 200:
                            # Tạo tên file duy nhất
                            ext = book_data['cover_url'].split('.')[-1]
                            filename = f"{isbn}_cover.{ext}"
                            
                            # Lưu file tạm
                            with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{ext}') as temp_file:
                                temp_file.write(response.content)
                                temp_file.flush()
                                
                                # Lưu vào model
                                book.cover.save(filename, File(open(temp_file.name, 'rb')))
                                
                                # Xóa file tạm
                                os.unlink(temp_file.name)
                    except Exception as e:
                        print(f"Error downloading cover: {e}")
                        messages.warning(request, "Không thể tải ảnh bìa sách")
                
                book.save()
                messages.success(request, 'Sách đã được thêm thành công')
                return redirect('llbook')
            else:
                form.add_error('isbn', 'Không tìm thấy sách với ISBN này.')
            return redirect('llbook')
    else:
        form = ISBNForm()
    return render(request, 'librarian/add_book_isbn.html', {'form': form})

@login_required
def labookocr_form(request):
    return render(request, 'librarian/add_book_cover.html')

@login_required
def labook_ocr(request):
    print("🔍 Đang vào view ladd_book_ocr với path:", request.path)

    if request.method != 'POST':
        messages.warning(request, "Phương thức không hợp lệ. Vui lòng dùng form để gửi file PDF.")
        print("➡️ Redirecting: Method not POST")
        return redirect('labookocr_form')

    if 'pdf' not in request.FILES:
        messages.error(request, 'Không có file PDF được tải lên.')
        print("➡️ Redirecting: No PDF file")
        return redirect('labookocr_form')

    pdf_file = request.FILES['pdf']

    # Check if the request is AJAX
    is_ajax = request.headers.get('x-requested-with') == 'XMLHttpRequest'

    # Kiểm tra định dạng file
    if not pdf_file.name.lower().endswith('.pdf'):
        messages.error(request, 'Định dạng file không hợp lệ. Vui lòng tải lên file PDF.')
        print("➡️ Redirecting: Invalid file format")
        if is_ajax:
            return JsonResponse({'status': 'error', 'message': 'Định dạng file không hợp lệ. Vui lòng tải lên file PDF.'}, status=400)
        return redirect('labookocr_form')

    try:
        # Đọc PDF file
        try:
            pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
        except Exception as e:
            raise ValueError(f"Không thể đọc file PDF: {str(e)}")
        
        # Kiểm tra số trang
        if pdf_document.page_count == 0:
            raise ValueError("File PDF không có trang nào")
        
        # Lấy trang đầu tiên
        first_page = pdf_document[0]
        
        # Chuyển trang đầu tiên thành ảnh với độ phân giải cao
        try:
            pix = first_page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
        except Exception as e:
            raise ValueError(f"Không thể chuyển đổi trang PDF thành ảnh: {str(e)}")
        
        # Chuyển đổi pixmap thành PIL Image
        try:
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        except Exception as e:
            raise ValueError(f"Không thể xử lý ảnh: {str(e)}")

        # Chuyển PIL Image sang OpenCV format
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # Xử lý ảnh trước khi OCR
        processed_img = process_image_for_ocr(img_cv)
        
        # Thực hiện OCR
        texts = extract_text_only(recognitor, detector, processed_img)
        
        if not texts:
            raise ValueError("Không thể nhận dạng được văn bản trong ảnh bìa. Vui lòng thử lại với ảnh rõ nét hơn.")

        # Lưu kết quả OCR
        ocr_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ocr_results')
        timestamp = timezone.now().strftime('%Y%m%d_%H%M%S')
        ocr_file = save_ocr_result(texts, ocr_dir, f"{timestamp}_{pdf_file.name}")

        # Gọi API để xử lý văn bản
        try:
            response = requests.post(
                "https://zep.hcmute.fit/7889/extract_book_info",
                json={"ocr_text": "\n".join(texts)},
                timeout=60
            )
            response.raise_for_status()
            data = response.json()
            title = data.get("title", "Không rõ")
            author = data.get("author", "Không rõ")

        except requests.exceptions.RequestException as e:
            raise ValueError(f"Lỗi khi gọi server LLM: {str(e)}")
        except Exception as e:
            raise ValueError(f"Lỗi không xác định khi xử lý văn bản: {str(e)}")

        # Tạo book object
        try:
            book = Book(
                title=title,
                author=author,
                uploaded_by=request.user.username,
                user_id=request.user.id
            )

            # Lưu ảnh bìa
            img_io = io.BytesIO()
            img.save(img_io, format='JPEG', quality=95)
            img_io.seek(0)
            book.cover.save(f"{title}_cover.jpg", File(img_io), save=False)

            # Lưu file PDF
            pdf_file.seek(0)
            book.pdf.save(f"{title}.pdf", File(pdf_file), save=False)

            # Lưu book vào database
            book.save()

            messages.success(request, 'Sách đã được thêm thành công')
            return redirect('llbook')

        except Exception as e:
            raise ValueError(f"Lỗi khi lưu thông tin sách: {str(e)}")

    except ValueError as ve:
        error_message = str(ve)
        print(f"❌ Lỗi xử lý: {error_message}")
        messages.error(request, f'Lỗi: {error_message}')
        if is_ajax:
            return JsonResponse({'status': 'error', 'message': error_message}, status=400)
        return redirect('labookocr_form')
    except Exception as e:
        error_message = f"Lỗi không xác định: {str(e)}"
        print(f"❌ {error_message}")
        messages.error(request, error_message)
        if is_ajax:
            return JsonResponse({'status': 'error', 'message': error_message}, status=500)
        return redirect('labookocr_form')

@login_required
def scan_book_cover_view(request):
    if request.method == 'POST':
        is_ajax = request.headers.get('x-requested-with') == 'XMLHttpRequest'

        if 'cover' not in request.FILES:
             if is_ajax:
                return JsonResponse({'status': 'error', 'message': 'No cover file received.'}, status=400)
             messages.error(request, 'No cover file received.')
             return redirect('scan_book_cover_form')

        file = request.FILES['cover']

        try:
            # Đọc ảnh từ file
            image_bytes = file.read()
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                raise ValueError("Không thể đọc được ảnh")

            # Xử lý ảnh sử dụng hàm từ ocr_utils
            processed_img = process_image_for_ocr(img)
            
            # Thực hiện OCR
            texts = extract_text_only(recognitor, detector, processed_img)
            
            if not texts:
                raise ValueError("Không thể nhận dạng được văn bản trong ảnh")

            # Lọc và làm sạch text trước khi gửi
            cleaned_texts = []
            for text in texts:
                # Loại bỏ các ký tự đặc biệt và khoảng trắng thừa
                cleaned = ' '.join(text.split())
                if cleaned and len(cleaned) > 1:  # Chỉ giữ lại text có ý nghĩa
                    cleaned_texts.append(cleaned)
            
            if not cleaned_texts:
                raise ValueError("Không tìm thấy văn bản có ý nghĩa trong ảnh")

            ocr_text = "\n".join(cleaned_texts)

        except ValueError as ve:
            print("❌ Lỗi xử lý ảnh:", str(ve))
            messages.error(request, f"Lỗi xử lý ảnh: {str(ve)}")
            if is_ajax:
                return JsonResponse({'status': 'error', 'message': f'Lỗi xử lý ảnh: {str(ve)}'}, status=400)
            return redirect('scan_book_cover_form')
        except Exception as e:
            print("❌ Lỗi không xác định:", str(e))
            messages.error(request, f"Lỗi không xác định: {str(e)}")
            if is_ajax:
                return JsonResponse({'status': 'error', 'message': f'Lỗi không xác định: {str(e)}'}, status=500)
            return redirect('scan_book_cover_form')

        # Thêm cơ chế retry cho việc gọi API
        max_retries = 3
        retry_delay = 2  # seconds
        last_error = None

        for attempt in range(max_retries):
            try:
                # Chuẩn bị dữ liệu gửi đi
                payload = {
                    "ocr_text": ocr_text,
                    "max_length": 1000  # Giới hạn độ dài text
                }

                # Gọi API để xử lý văn bản
                response = requests.post(
                    "https://zep.hcmute.fit/7889/extract_book_info",
                    json=payload,
                    timeout=30,
                    headers={
                        'Content-Type': 'application/json',
                        'Accept': 'application/json',
                        'User-Agent': 'SmartLibrary/1.0'
                    }
                )

                # Kiểm tra status code
                if response.status_code == 500:
                    print(f"❌ Server error response: {response.text}")
                    raise requests.exceptions.RequestException(f"Server error: {response.text}")

                response.raise_for_status()
                
                # Kiểm tra response data
                try:
                    data = response.json()
                except ValueError as e:
                    print(f"❌ Invalid JSON response: {response.text}")
                    raise requests.exceptions.RequestException(f"Invalid JSON response: {str(e)}")

                # Validate response data
                if not isinstance(data, dict):
                    raise ValueError("Invalid response format")
                
                title = data.get("title")
                author = data.get("author")

                if not title or not author:
                    print(f"❌ Missing required fields in response: {data}")
                    raise ValueError("Missing required fields in response")

                # If it's an AJAX request, return JSON response
                if is_ajax:
                    return JsonResponse({
                        'status': 'success',
                        'ocr_text': ocr_text,
                        'title': title,
                        'author': author,
                    })
                break  # Thoát khỏi vòng lặp nếu thành công

            except requests.exceptions.RequestException as e:
                last_error = e
                print(f"❌ Lần thử {attempt + 1}/{max_retries} thất bại: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                
                # Nếu đã hết số lần thử
                error_message = f"Không thể kết nối đến server LLM sau {max_retries} lần thử. Vui lòng thử lại sau."
                print(f"❌ {error_message}")
                messages.error(request, error_message)
                if is_ajax:
                    return JsonResponse({
                        'status': 'error',
                        'message': error_message,
                        'ocr_text': ocr_text,
                        'raw_error': str(e)  # Thêm thông tin lỗi chi tiết
                    }, status=500)
                return redirect("scan_book_cover_form")

            except Exception as e:
                last_error = e
                print(f"❌ Lỗi không xác định trong lần thử {attempt + 1}: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                
                error_message = f"Lỗi không xác định: {str(e)}"
                messages.error(request, error_message)
                if is_ajax:
                    return JsonResponse({
                        'status': 'error',
                        'message': error_message,
                        'ocr_text': ocr_text,
                        'raw_error': str(e)
                    }, status=500)
                return redirect("scan_book_cover_form")

        messages.error(request, 'Unexpected error during processing.')
        return redirect('scan_book_cover_form')

    # Handle GET request - render the template with the camera interface
    return render(request, 'librarian/scan_book_cover.html')


class LBookListView(LoginRequiredMixin, RoleRequiredMixin, ListView):
    required_role = 'librarian'
    model = Book
    template_name = 'librarian/book_list.html'
    context_object_name = 'books'
    paginate_by = 3

    def get_queryset(self):
        return Book.objects.order_by('-id')


class LManageBook(LoginRequiredMixin, RoleRequiredMixin, ListView):
    required_role = 'librarian'
    model = Book
    template_name = 'librarian/manage_books.html'
    context_object_name = 'books'
    paginate_by = 3

    def get_queryset(self):
        return Book.objects.order_by('-id')

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        paginator = context['paginator']
        page_obj = context['page_obj']

        # Logic to generate the list of pages to display (1 2 3 ... cuối)
        num_pages = paginator.num_pages
        current_page = page_obj.number
        page_range = paginator.page_range

        # Define how many pages to show around the current page
        pages_around_current = 2

        # Define the number of initial and final pages to show
        pages_at_ends = 3

        # Calculate the range of pages to display
        # Start with initial pages
        pages_to_show = set(list(range(1, min(num_pages + 1, pages_at_ends + 1))))

        # Add pages around the current page
        pages_to_show.update(list(range(max(1, current_page - pages_around_current), min(num_pages + 1, current_page + pages_around_current + 1))))

        # Add final pages
        pages_to_show.update(list(range(max(1, num_pages - pages_at_ends + 1), num_pages + 1)))

        # Create a sorted list of unique page numbers
        pages_list = sorted(list(pages_to_show))

        # Add ellipsis where necessary
        final_pages_list = []
        last_page = 0
        for page in pages_list:
            if page > last_page + 1:
                final_pages_list.append('...')
            final_pages_list.append(page)
            last_page = page

        context['final_page_range'] = final_pages_list
        return context


class LDeleteRequest(LoginRequiredMixin,ListView):
    model = DeleteRequest
    template_name = 'librarian/delete_request.html'
    context_object_name = 'feedbacks'
    paginate_by = 3

    def get_queryset(self):
        return DeleteRequest.objects.order_by('-id')


class LViewBook(LoginRequiredMixin, RoleRequiredMixin, DetailView):
    required_role = 'librarian'
    model = Book
    template_name = 'librarian/book_detail.html'

    
class LEditView(LoginRequiredMixin, RoleRequiredMixin, UpdateView):
    required_role = 'librarian'
    model = Book
    form_class = BookForm
    template_name = 'librarian/edit_book.html'
    success_url = reverse_lazy('lmbook')
    success_message = 'Data was updated successfully'


class LDeleteView(LoginRequiredMixin, RoleRequiredMixin, DeleteView):
    required_role = 'librarian'
    model = Book
    template_name = 'librarian/confirm_delete.html'
    success_url = reverse_lazy('lmbook')
    success_message = 'Data was deleted successfully'


class LDeleteBook(LoginRequiredMixin,DeleteView):
    model = Book
    template_name = 'librarian/confirm_delete2.html'
    success_url = reverse_lazy('librarian')
    success_message = 'Data was dele successfully'



@login_required
def lsearch(request):
    query = request.GET['query']
    print(type(query))


    #data = query.split()
    data = query
    print(len(data))
    if( len(data) == 0):
        return redirect('publisher')
    else:
                a = data

                # Searching for It
                qs5 =models.Book.objects.filter(id__iexact=a).distinct()
                qs6 =models.Book.objects.filter(id__exact=a).distinct()

                qs7 =models.Book.objects.all().filter(id__contains=a)
                qs8 =models.Book.objects.select_related().filter(id__contains=a).distinct()
                qs9 =models.Book.objects.filter(id__startswith=a).distinct()
                qs10 =models.Book.objects.filter(id__endswith=a).distinct()
                qs11 =models.Book.objects.filter(id__istartswith=a).distinct()
                qs12 =models.Book.objects.all().filter(id__icontains=a)
                qs13 =models.Book.objects.filter(id__iendswith=a).distinct()




                files = itertools.chain(qs5, qs6, qs7, qs8, qs9, qs10, qs11, qs12, qs13)

                res = []
                for i in files:
                    if i not in res:
                        res.append(i)


                # word variable will be shown in html when user click on search button
                word="Searched Result :"
                print("Result")

                print(res)
                files = res




                page = request.GET.get('page', 1)
                paginator = Paginator(files, 10)
                try:
                    files = paginator.page(page)
                except PageNotAnInteger:
                    files = paginator.page(1)
                except EmptyPage:
                    files = paginator.page(paginator.num_pages)
   


                if files:
                    return render(request,'librarian/result.html',{'files':files,'word':word})
                return render(request,'librarian/result.html',{'files':files,'word':word})


class LCreateChat(LoginRequiredMixin, CreateView):
    form_class = ChatForm
    model = Chat
    template_name = 'librarian/chat_form.html'
    success_url = reverse_lazy('llchat')


    def form_valid(self, form):
        self.object = form.save(commit=False)
        self.object.user = self.request.user
        self.object.save()
        return super().form_valid(form)




class LListChat(LoginRequiredMixin, ListView):
    model = Chat
    template_name = 'librarian/chat_list.html'

    def get_queryset(self):
        return Chat.objects.filter(posted_at__lt=timezone.now()).order_by('posted_at')














# Admin views

def dashboard(request):
    book = Book.objects.all().count()
    user = User.objects.all().count()

    context = {'book':book, 'user':user}

    return render(request, 'dashboard/home.html', context)

def create_user_form(request):
    choice = ['1', '0', 'Publisher', 'Admin', 'Librarian']
    choice = {'choice': choice}

    return render(request, 'dashboard/add_user.html', choice)


class ADeleteUser(SuccessMessageMixin, DeleteView):
    model = User
    template_name='dashboard/confirm_delete3.html'
    success_url = reverse_lazy('aluser')
    success_message = "Data successfully deleted"


class AEditUser(SuccessMessageMixin, UpdateView): 
    model = User
    form_class = UserForm
    template_name = 'dashboard/edit_user.html'
    success_url = reverse_lazy('aluser')
    success_message = "Data successfully updated"

class ListUserView(generic.ListView):
    model = User
    template_name = 'dashboard/list_users.html'
    context_object_name = 'users'
    paginate_by = 4

    def get_queryset(self):
        return User.objects.order_by('-id')

def create_user(request):
    choice = ['1', '0', 'Publisher', 'Admin', 'Librarian', 'Student']
    choice = {'choice': choice}
    if request.method == 'POST':
            first_name=request.POST['first_name']
            last_name=request.POST['last_name']
            username=request.POST['username']
            userType=request.POST['userType']
            email=request.POST['email']
            password=request.POST['password']
            password = make_password(password)
            print("User Type")
            print(userType)
            if userType == "Publisher":
                a = User(first_name=first_name, last_name=last_name, username=username, email=email, password=password, is_publisher=True)
                a.save()
                messages.success(request, 'Member was created successfully!')
                return redirect('aluser')
            elif userType == "Admin":
                a = User(first_name=first_name, last_name=last_name, username=username, email=email, password=password, is_admin=True)
                a.save()
                messages.success(request, 'Member was created successfully!')
                return redirect('aluser')
            elif userType == "Librarian":
                a = User(first_name=first_name, last_name=last_name, username=username, email=email, password=password, is_librarian=True)
                a.save()
                messages.success(request, 'Member was created successfully!')
                return redirect('aluser') 
            elif userType == "Student":
                a = User(first_name=first_name, last_name=last_name, username=username, email=email, password=password, is_student=True)
                a.save()
                messages.success(request, 'Member was created successfully!')
                return redirect('aluser')   
            else:
                messages.success(request, 'Member was not created')
                return redirect('create_user_form')
    else:
        return redirect('create_user_form')


class ALViewUser(DetailView):
    model = User
    template_name='dashboard/user_detail.html'



class ACreateChat(LoginRequiredMixin, CreateView):
    form_class = ChatForm
    model = Chat
    template_name = 'dashboard/chat_form.html'
    success_url = reverse_lazy('alchat')


    def form_valid(self, form):
        self.object = form.save(commit=False)
        self.object.user = self.request.user
        self.object.save()
        return super().form_valid(form)




class AListChat(LoginRequiredMixin, ListView):
    model = Chat
    template_name = 'dashboard/chat_list.html'

    def get_queryset(self):
        return Chat.objects.filter(posted_at__lt=timezone.now()).order_by('posted_at')


@login_required
def aabook_form(request):
    return render(request, 'dashboard/add_book.html')


@login_required
def aabook(request):
    if request.method == 'POST':
        title = request.POST['title']
        author = request.POST['author']
        year = request.POST['year']
        publisher = request.POST['publisher']
        desc = request.POST['desc']
        cover = request.FILES['cover']
        pdf = request.FILES['pdf']
        current_user = request.user
        user_id = current_user.id
        username = current_user.username

        a = Book(title=title, author=author, year=year, publisher=publisher, 
            desc=desc, cover=cover, pdf=pdf, uploaded_by=username, user_id=user_id)
        a.save()
        messages.success(request, 'Book was uploaded successfully')
        return redirect('albook')
    else:
        messages.error(request, 'Book was not uploaded successfully')
        return redirect('aabook_form')


class ABookListView(LoginRequiredMixin,ListView):
    model = Book
    template_name = 'dashboard/book_list.html'
    context_object_name = 'books'
    paginate_by = 3

    def get_queryset(self):
        return Book.objects.order_by('-id')




class AManageBook(LoginRequiredMixin,ListView):
    model = Book
    template_name = 'dashboard/manage_books.html'
    context_object_name = 'books'
    paginate_by = 3

    def get_queryset(self):
        return Book.objects.order_by('-id')




class ADeleteBook(LoginRequiredMixin,DeleteView):
    model = Book
    template_name = 'dashboard/confirm_delete2.html'
    success_url = reverse_lazy('ambook')
    success_message = 'Data was dele successfully'


class ADeleteBookk(LoginRequiredMixin,DeleteView):
    model = Book
    template_name = 'dashboard/confirm_delete.html'
    success_url = reverse_lazy('dashboard')
    success_message = 'Data was dele successfully'


class AViewBook(LoginRequiredMixin,DetailView):
    model = Book
    template_name = 'dashboard/book_detail.html'




class AEditView(LoginRequiredMixin,UpdateView):
    model = Book
    form_class = BookForm
    template_name = 'dashboard/edit_book.html'
    success_url = reverse_lazy('ambook')
    success_message = 'Data was updated successfully'




class ADeleteRequest(LoginRequiredMixin,ListView):
    model = DeleteRequest
    template_name = 'dashboard/delete_request.html'
    context_object_name = 'feedbacks'
    paginate_by = 3

    def get_queryset(self):
        return DeleteRequest.objects.order_by('-id')



class AFeedback(LoginRequiredMixin,ListView):
    model = Feedback
    template_name = 'dashboard/feedback.html'
    context_object_name = 'feedbacks'
    paginate_by = 3

    def get_queryset(self):
        return Feedback.objects.order_by('-id')



@login_required
def asearch(request):
    query = request.GET['query']
    print(type(query))


    #data = query.split()
    data = query
    print(len(data))
    if( len(data) == 0):
        return redirect('dashborad')
    else:
                a = data

                # Searching for It
                qs5 =models.Book.objects.filter(id__iexact=a).distinct()
                qs6 =models.Book.objects.filter(id__exact=a).distinct()

                qs7 =models.Book.objects.all().filter(id__contains=a)
                qs8 =models.Book.objects.select_related().filter(id__contains=a).distinct()
                qs9 =models.Book.objects.filter(id__startswith=a).distinct()
                qs10 =models.Book.objects.filter(id__endswith=a).distinct()
                qs11 =models.Book.objects.filter(id__istartswith=a).distinct()
                qs12 =models.Book.objects.all().filter(id__icontains=a)
                qs13 =models.Book.objects.filter(id__iendswith=a).distinct()




                files = itertools.chain(qs5, qs6, qs7, qs8, qs9, qs10, qs11, qs12, qs13)

                res = []
                for i in files:
                    if i not in res:
                        res.append(i)


                # word variable will be shown in html when user click on search button
                word="Searched Result :"
                print("Result")

                print(res)
                files = res




                page = request.GET.get('page', 1)
                paginator = Paginator(files, 10)
                try:
                    files = paginator.page(page)
                except PageNotAnInteger:
                    files = paginator.page(1)
                except EmptyPage:
                    files = paginator.page(paginator.num_pages)
   


                if files:
                    return render(request,'dashboard/result.html',{'files':files,'word':word})
                return render(request,'dashboard/result.html',{'files':files,'word':word})

def library_home(request):
    user = request.user
    department_id = request.GET.get('department')
    view_type = request.GET.get('view')
    
    # Get all departments for sidebar
    departments = Department.objects.all()
    
    # Get selected department if any
    selected_department = None
    if department_id:
        selected_department = Department.objects.filter(id=department_id).first()
    
    if selected_department:
        # If department is selected, show books from that department
        recommend_books = Book.objects.filter(department=selected_department)
        discover_books = None  # Don't show discover books when filtering by department
    elif view_type == 'all':
        # Show all books
        recommend_books = Book.objects.all()
        discover_books = None
    elif view_type == 'recommended':
        # Show only recommended books
        interactions = UserBookInteraction.objects.filter(user=user).order_by('-timestamp')[:10]
        
        if interactions:
            try:
                book_embeddings = [get_book_embedding(inter.book) for inter in reversed(interactions)]
                input_tensor = torch.tensor(np.stack(book_embeddings)).unsqueeze(1)

                # Initialize model with the same number of books as the saved weights
                model = RecTransformer(n_books=250)  # Use the number of books the model was trained with
                model.load_state_dict(torch.load('bookstore/recommend/model.pt'))
                model.eval()

                with torch.no_grad():
                    logits = model(input_tensor)
                    top_indices = torch.topk(logits[0], 10).indices.tolist()

                # Map indices to actual books, handling the case where we have more books now
                all_books = list(Book.objects.all())
                recommend_books = []
                for idx in top_indices:
                    if idx < len(all_books):
                        recommend_books.append(all_books[idx])
                    if len(recommend_books) >= 10:
                        break
            except Exception as e:
                print(f"Error loading recommendation model: {e}")
                # Fallback to latest books if model loading fails
                recommend_books = Book.objects.order_by('-id')[:4]
        else:
            # If no interactions, get latest books as recommendations
            recommend_books = Book.objects.order_by('-id')[:4]
        discover_books = None
    else:
        # Default view: recommended books + discover books
        interactions = UserBookInteraction.objects.filter(user=user).order_by('-timestamp')[:10]
        
        if interactions:
            try:
                book_embeddings = [get_book_embedding(inter.book) for inter in reversed(interactions)]
                input_tensor = torch.tensor(np.stack(book_embeddings)).unsqueeze(1)

                # Initialize model with the same number of books as the saved weights
                model = RecTransformer(n_books=250)  # Use the number of books the model was trained with
                model.load_state_dict(torch.load('bookstore/recommend/model.pt'))
                model.eval()

                with torch.no_grad():
                    logits = model(input_tensor)
                    top_indices = torch.topk(logits[0], 5).indices.tolist()

                # Map indices to actual books, handling the case where we have more books now
                all_books = list(Book.objects.all())
                recommend_books = []
                for idx in top_indices:
                    if idx < len(all_books):
                        recommend_books.append(all_books[idx])
                    if len(recommend_books) >= 5:
                        break
            except Exception as e:
                print(f"Error loading recommendation model: {e}")
                # Fallback to latest books if model loading fails
                recommend_books = Book.objects.order_by('-id')[:4]
        else:
            # If no interactions, get latest books as recommendations
            recommend_books = Book.objects.order_by('-id')[:4]

        # Get discover books (next 12 books after recommendations)
        discover_books = Book.objects.order_by('-id')[4:16]

    return render(request, 'library_home.html', {
        'recommend_books': recommend_books,
        'discover_books': discover_books,
        'departments': departments,
        'selected_department': selected_department,
        'view_type': view_type,
    })

def book_grid_view(request):
    # Lấy tất cả sách hoặc lọc theo thể loại nếu có
    category = request.GET.get('category')
    if category:
        books = Book.objects.filter(category=category)
    else:
        books = Book.objects.all()
    # Lấy danh sách các thể loại (giả sử Book có trường category)
    categories = Book.objects.values_list('category', flat=True).distinct()
    return render(request, 'bookstore/book_grid.html', {
        'books': books,
        'categories': categories,
        'selected_category': category,
    })

def face_login(request):
    """Redirect to loginapp's face login page"""
    return redirect('loginapp:index')

def video_feed(request):
    """Redirect to loginapp's video feed"""
    return redirect('loginapp:video_feed')

@csrf_exempt
def check_login_status(request):
    """Check login status and redirect based on user role"""
    if request.user.is_authenticated:
        if request.user.is_admin or request.user.is_superuser:
            return redirect('dashboard')
        elif request.user.is_librarian:
            return redirect('librarian')
        elif request.user.is_publisher:
            return redirect('publisher')
        elif request.user.is_student:
            return redirect('library_home')
        else:
            messages.error(request, "Tài khoản của bạn không có quyền truy cập")
            return redirect('home')
    return redirect('loginapp:check_login_status')

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

def show_qr(request):
    """Redirect to loginapp's show QR"""
    return redirect('loginapp:show_qr')

def qr_login(request, token):
    """Redirect to loginapp's QR login"""
    return redirect('loginapp:qr_login', token=token)

def index(request):
    """Main page (choose between face recognition or QR)"""
    return render(request, 'bookstore/index.html', {'show_qr': False})

def record_interaction(request):
    """Ghi nhận tương tác mới và cập nhật model"""
    if request.method == 'POST':
        user_id = request.POST.get('user_id')
        book_id = request.POST.get('book_id')
        action = request.POST.get('action')  # 'view', 'ask', 'download'
        
        try:
            # Tạo tương tác mới
            interaction = UserBookInteraction.objects.create(
                user_id=user_id,
                book_id=book_id,
                action=action
            )
            
            # Thêm vào queue để update model
            service = get_recommendation_service()
            service.add_interaction(interaction)
            
            return JsonResponse({'status': 'success'})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)})
    
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'})

def get_recommendations(request):
    """Lấy recommendations cho user"""
    if request.method == 'GET':
        user_id = request.GET.get('user_id')
        n_recommendations = int(request.GET.get('n', 10))
        
        try:
            service = get_recommendation_service()
            recommendations = service.get_recommendations(user_id, n_recommendations)
            
            # Lấy thông tin chi tiết của sách
            books = []
            for rec in recommendations:
                book = Book.objects.get(id=rec['book_id'])
                books.append({
                    'id': book.id,
                    'title': book.title,
                    'author': book.author,
                    'cover_url': book.cover.url if book.cover else None,
                    'score': rec['score']
                })
            
            return JsonResponse({'status': 'success', 'books': books})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)})
    
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'})

@csrf_exempt
@require_http_methods(["POST"])
def scan_barcode(request):
    try:
        data = json.loads(request.body)
        image_data = data.get('image', '')
        if not image_data:
            return JsonResponse({'error': 'No image data provided'}, status=400)
            
        # Xử lý chuỗi base64
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        # Chuyển đổi bytes thành numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return JsonResponse({'error': 'Invalid image data'}, status=400)

        # Thử nhiều cách xử lý ảnh khác nhau để tăng khả năng nhận dạng
        processed_images = []
        
        # 1. Ảnh gốc
        processed_images.append(img)
        
        # 2. Ảnh xám
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        processed_images.append(gray)
        
        # 3. Ảnh xám với độ tương phản cao
        high_contrast = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)
        processed_images.append(high_contrast)
        
        # 4. Ảnh nhị phân
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed_images.append(binary)
        
        # 5. Ảnh đảo ngược
        processed_images.append(cv2.bitwise_not(binary))
        
        # Thử giải mã mã vạch với tất cả các phiên bản ảnh
        for processed_img in processed_images:
            decoded_objects = decode(processed_img)
            
            for obj in decoded_objects:
                try:
                    isbn = obj.data.decode('utf-8')
                    # Loại bỏ các ký tự không phải số
                    isbn = ''.join(filter(str.isdigit, isbn))
                    
                    # Kiểm tra ISBN (10 hoặc 13 chữ số)
                    if len(isbn) in [10, 13]:
                        return JsonResponse({'isbn': isbn})
                except:
                    continue
                
        return JsonResponse({'isbn': None})
    except Exception as e:
        print(f"Error in scan_barcode: {str(e)}")
        return JsonResponse({'error': str(e)}, status=400)














