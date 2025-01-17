# users/urls.py
from django.urls import path
from .views import register, home, profile, logout_view
from django.contrib.auth.views import LogoutView
from .views import CustomLoginView
from . import views
# from .views import CustomLogoutView

urlpatterns = [
    path("register/", register, name="register"),
    path("", home, name="home"),
    path("logout/", LogoutView.as_view(), name="logout"), #로그아웃 페이지
    path('profile/', profile, name='profile'),  # profile URL 추가
    path('login/', CustomLoginView.as_view(), name='login'), 
]

