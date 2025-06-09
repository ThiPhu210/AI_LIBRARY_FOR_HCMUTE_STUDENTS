from django.urls import path
from . import views

app_name = 'loginapp'

urlpatterns = [
    path('', views.index, name='index'),
    path('face-login/', views.face_login, name='face_login'),
    path('video-feed/', views.video_feed, name='video_feed'),
    path('check-login-status/', views.check_login_status, name='check_login_status'),
    # path('register-face/', views.register_face, name='register_face'),
    path('show-qr/', views.show_qr, name='show_qr'),
    path('qr-login/<str:token>/', views.qr_login, name='qr_login'),
] 