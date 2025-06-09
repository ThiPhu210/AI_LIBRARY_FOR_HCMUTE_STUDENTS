from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from bootstrap_modal_forms.mixins import PopRequestMixin, CreateUpdateAjaxMixin
from django.forms import ModelForm
from bookstore.models import Chat, Book
from django import forms
import re


class ChatForm(forms.ModelForm):
    class Meta:
        model = Chat
        fields = ('message', )


class BookForm(forms.ModelForm):
    class Meta:
        model = Book
        fields = ('title', 'author', 'publisher', 'year', 'uploaded_by', 'desc', 'pdf', 'cover')
        
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Đặt required=False cho các trường file
        self.fields['pdf'].required = False
        self.fields['cover'].required = False
        # Đặt required=False cho các trường khác ngoài title và author
        self.fields['publisher'].required = False
        self.fields['year'].required = False
        self.fields['uploaded_by'].required = False
        self.fields['desc'].required = False


class UserForm(forms.ModelForm):
    class Meta:
        model = User
        fields = ('username', 'first_name', 'last_name', 'email', 'password')
        
class ISBNForm(forms.Form):
    isbn = forms.CharField(
        label='Mã ISBN',
        max_length=20,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Nhập mã ISBN (10 hoặc 13 chữ số)',
            'pattern': '[0-9]{10}|[0-9]{13}',
            'title': 'Mã ISBN phải có 10 hoặc 13 chữ số'
        })
    )

    def clean_isbn(self):
        isbn = self.cleaned_data['isbn']
        # Loại bỏ các ký tự không phải số
        isbn = re.sub(r'\D', '', isbn)
        
        # Kiểm tra độ dài
        if len(isbn) not in [10, 13]:
            raise forms.ValidationError('Mã ISBN phải có 10 hoặc 13 chữ số')
            
        # Kiểm tra tính hợp lệ của ISBN-10
        if len(isbn) == 10:
            total = 0
            for i in range(9):
                total += int(isbn[i]) * (10 - i)
            check_digit = (11 - (total % 11)) % 11
            if check_digit != int(isbn[9]):
                raise forms.ValidationError('Mã ISBN-10 không hợp lệ')
                
        # Kiểm tra tính hợp lệ của ISBN-13
        elif len(isbn) == 13:
            total = 0
            for i in range(12):
                total += int(isbn[i]) * (1 if i % 2 == 0 else 3)
            check_digit = (10 - (total % 10)) % 10
            if check_digit != int(isbn[12]):
                raise forms.ValidationError('Mã ISBN-13 không hợp lệ')
                
        return isbn