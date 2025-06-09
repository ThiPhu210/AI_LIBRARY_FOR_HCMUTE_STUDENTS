from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from . import views
from django.contrib.auth import views as auth_views

urlpatterns = [
    
    # Shared URLs
    path('', views.login_form, name='home'),
    path('login/', views.loginView, name='login'),
    path('logout/', views.logoutView, name='logout'),
    path('regform/', views.register_form, name='regform'),
    path('register/', views.registerView, name='register'),
    path('books/', views.book_grid_view, name='book_grid'),
    path('libraryview/', views.library_home, name='library_home'),
    
    # Face login URLs - redirect to loginapp
    path('face-login/', views.face_login, name='face_login'),
    path('video-feed/', views.video_feed, name='video_feed'),
    path('check-login-status/', views.check_login_status, name='check_login_status'),
    # path('register-face/', views.register_face, name='register_face'),
    
    # QR code login URLs - redirect to loginapp
    path('qr-login/', views.show_qr, name='show_qr'),
    path('qr-login/<str:token>/', views.qr_login, name='qr_login'),
    
    # Include loginapp URLs
    path('login/', include('loginapp.urls')),
    
    # # Book app URLs
    # path('book/', include('bookapp.urls')),
    
    # Librarian URLs
    path('librarian/', views.librarian, name='librarian'),
    path('labook_form/', views.labook_form, name='labook_form'),
    path('labookisbn_form/', views.labookisbn_form, name='labookisbn_form'),
    path('ladd_book_isbn/', views.labookisbn, name="labookisbn"),
    path('labookocr_form/', views.labookocr_form, name='labookocr_form'),
    path('ladd_book_ocr/', views.labook_ocr, name="ladd_book_ocr"),
    path('labook/', views.labook, name='labook'),
    path('llbook/', views.LBookListView.as_view(), name='llbook'),
    path('ldrequest/', views.LDeleteRequest.as_view(), name='ldrequest'),
    path('lsearch/', views.lsearch, name='lsearch'),
    path('ldbook/<int:pk>', views.LDeleteBook.as_view(), name='ldbook'),
    path('lmbook/', views.LManageBook.as_view(), name='lmbook'),
    path('ldbookk/<int:pk>', views.LDeleteView.as_view(), name='ldbookk'),
    path('lvbook/<int:pk>', views.LViewBook.as_view(), name='lvbook'),
    path('lebook/<int:pk>', views.LEditView.as_view(), name='lebook'),
    path('lcchat/', views.LCreateChat.as_view(), name='lcchat'),
    path('llchat/', views.LListChat.as_view(), name='llchat'),
    path('scan_book_cover/', views.scan_book_cover_view, name='scan_book_cover_form'),
    
    # Publisher URLs
    path('phome/', views.UBookListView.as_view(), name='phome'),
    path('publisher/', views.UBookListView.as_view(), name='publisher'),
    path('uabook_form/', views.uabook_form, name='uabook_form'),
    path('uadd_book_ocr/', views.uabookocr_form, name='uabookocr_form'),
    path('uadd_book_ocr/process/', views.uabook_ocr, name='uabook_ocr'),
    path('uabook/', views.uabook, name='uabook'),
    path('ucchat/', views.UCreateChat.as_view(), name='ucchat'),
    path('ulchat/', views.UListChat.as_view(), name='ulchat'),
    path('request_form/', views.request_form, name='request_form'),
    path('delete_request/', views.delete_request, name='delete_request'),
    path('feedback_form/', views.feedback_form, name='feedback_form'),
    path('send_feedback/', views.send_feedback, name='send_feedback'),
    path('about/', views.about, name='about'),
    path('usearch/', views.usearch, name='usearch'),
    
    # Admin URLs
    path('dashboard/', views.dashboard, name='dashboard'),
    path('acchat/', views.ACreateChat.as_view(), name='acchat'),
    path('alchat/', views.AListChat.as_view(), name='alchat'),
    path('aabook_form/', views.aabook_form, name='aabook_form'),
    path('aabook/', views.aabook, name='aabook'),
    path('albook/', views.ABookListView.as_view(), name='albook'),
    path('ambook/', views.AManageBook.as_view(), name='ambook'),
    path('adbook/<int:pk>', views.ADeleteBook.as_view(), name='adbook'),
    path('avbook/<int:pk>', views.AViewBook.as_view(), name='avbook'),
    path('aebook/<int:pk>', views.AEditView.as_view(), name='aebook'),
    path('adrequest/', views.ADeleteRequest.as_view(), name='adrequest'),
    path('afeedback/', views.AFeedback.as_view(), name='afeedback'),
    path('asearch/', views.asearch, name='asearch'),
    path('adbookk/<int:pk>', views.ADeleteBookk.as_view(), name='adbookk'),
    path('create_user_form/', views.create_user_form, name='create_user_form'),
    path('aluser/', views.ListUserView.as_view(), name='aluser'),
    path('create_use/', views.create_user, name='create_user'),
    path('alvuser/<int:pk>', views.ALViewUser.as_view(), name='alvuser'),
    path('aeuser/<int:pk>', views.AEditUser.as_view(), name='aeuser'),
    path('aduser/<int:pk>', views.ADeleteUser.as_view(), name='aduser'),
    path('scan-barcode/', views.scan_barcode, name='scan_barcode'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
