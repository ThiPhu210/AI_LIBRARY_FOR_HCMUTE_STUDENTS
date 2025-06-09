from django.db import models
from django.contrib.auth.models import AbstractUser
from django.contrib.auth.models import User
from django.contrib.auth import get_user_model
from django.urls import reverse
import uuid
import numpy as np
from django.utils.translation import gettext_lazy as _


#User = get_user_model()


class User(AbstractUser):
    
    is_admin = models.BooleanField(default=False)
    is_publisher = models.BooleanField(default=False)
    is_librarian = models.BooleanField(default=False)
    is_student = models.BooleanField(default=False)

    face_embedding = models.BinaryField(null=True, blank=True)
    qr_code = models.CharField(max_length=255, null=True, blank=True)

    class Meta:
        swappable = 'AUTH_USER_MODEL'
        
class FaceLoginData(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    embedding = models.BinaryField(null=True, blank=True)
    
class BookIntro(models.Model):
    book = models.OneToOneField('Book', on_delete=models.CASCADE)
    intro_text = models.TextField()
    extracted_at = models.DateTimeField(auto_now_add=True)
    
class Department(models.Model):
    name = models.CharField(max_length=100, unique=True)

    def __str__(self):
        return self.name
    
class Book(models.Model):
    ISBN_SOURCE = 'isbn'
    OCR_SOURCE = 'ocr'
    MANUAL_SOURCE = 'manual'
    SOURCE_CHOICES = [
        (ISBN_SOURCE, 'ISBN'),
        (OCR_SOURCE, 'OCR'),
        (MANUAL_SOURCE, 'Manual')
    ]
    isbn = models.CharField(max_length=20, null=True, blank=True)
    title = models.CharField(max_length=200)
    author = models.CharField(max_length=100, null=True, blank=True)
    year = models.CharField(max_length=100, null=True, blank=True)
    publisher = models.CharField(max_length=200, null=True, blank=True)
    desc = models.CharField(max_length=1000, null=True)
    # intro = models.TextField(null=True, blank=True)  # lời mở đầu
    uploaded_by = models.CharField(max_length=100, null=True, blank=True)
    user_id = models.CharField(max_length=100, null=True, blank=True)
    pdf = models.FileField(upload_to='bookapp/pdfs/')
    cover = models.ImageField(upload_to='bookapp/covers/', null=True, blank=True)
    department = models.ForeignKey('Department', on_delete=models.SET_NULL, null=True, blank=True)  # new line
    input_source = models.CharField(max_length=10, choices=SOURCE_CHOICES, default=MANUAL_SOURCE, null=True, blank=True)
    def __str__(self):
        return self.title

    def delete(self, *args, **kwargs):
        self.pdf.delete()
        self.cover.delete()
        super().delete(*args, **kwargs)      
          
class QRCodeLoginToken(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    token = models.UUIDField(default=uuid.uuid4, unique=True, editable=False)
    created_at = models.DateTimeField(auto_now_add=True)
    is_used = models.BooleanField(default=False)
    
class BookQA(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    book = models.ForeignKey(Book, on_delete=models.SET_NULL, null=True, blank=True)
    question = models.TextField()
    answer = models.TextField()
    asked_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Q: {self.question[:50]} - by {self.user.username}"
    
class BookScanHistory(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    book = models.ForeignKey(Book, on_delete=models.CASCADE)
    scanned_at = models.DateTimeField(auto_now_add=True)
    
class Chat(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='bookstore_chats')
    message = models.TextField()
    posted_at = models.DateTimeField(auto_now=True, null=True)

    def __str__(self):
        return str(self.message)

class UserBookInteraction(models.Model):
    ACTION_CHOICES = [
        ('view', 'View'),
        ('ask', 'Ask Question'),
        ('download', 'Download'),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE)
    book = models.ForeignKey(Book, on_delete=models.CASCADE)
    action = models.CharField(max_length=10, choices=ACTION_CHOICES)
    timestamp = models.DateTimeField(auto_now_add=True)

    def get_reward(self):
        base_reward = {
        'view': 1.0,
        'ask': 2.0,
        'download': 3.0
    }.get(self.action, 0.0)

    # Tăng thưởng cho sách hiếm (ít người tương tác)
        total_interactions = UserBookInteraction.objects.filter(book=self.book).count()
        rarity_bonus = np.exp(-total_interactions/10.0) * 2.0  # Giảm dần theo hàm mũ
        
        # Cộng thưởng nếu người dùng đã từng tương tác sách này trước đó
        previous_interactions = UserBookInteraction.objects.filter(
            user=self.user, book=self.book
        ).exclude(id=self.id).count()
        repeat_bonus = np.log(1 + previous_interactions) * 0.5  # Tăng theo log
        
        # Thưởng cho hành động liên tiếp (sequential)
        last_action = UserBookInteraction.objects.filter(
            user=self.user
        ).order_by('-timestamp').first()
        sequential_bonus = 0.5 if last_action and (self.timestamp - last_action.timestamp).seconds < 3600 else 0.0
        
        return base_reward + rarity_bonus + repeat_bonus + sequential_bonus

class DeleteRequest(models.Model):
    delete_request = models.CharField(max_length=100, null=True, blank=True)


    def __str__(self):
        return self.delete_request


class Feedback(models.Model):
    feedback = models.CharField(max_length=100, null=True, blank=True)


    def __str__(self):
        return self.feedback

class StudentProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='student_profile')
    student_id = models.CharField(max_length=20, unique=True, verbose_name=_('Student ID'))
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = _('Student Profile')
        verbose_name_plural = _('Student Profiles')

    def __str__(self):
        return f"{self.user.get_full_name()} - {self.student_id}"













