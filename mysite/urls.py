from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from users.views import HomeView
from django.contrib.auth import views as auth_views
from . import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path("users/", include("users.urls")),  # users 앱의 URL 추가
    path("accounts/", include("django.contrib.auth.urls")),
    path('', include('users.urls')),   # Django 기본 로그인/로그아웃 URL 추가
    path('', views.home_view, name='home'),  # 홈 페이지 URL 연결
    path("", HomeView.as_view(), name="home"),
    path('login/', auth_views.LoginView.as_view(), name='login'),  # 로그인 URL
    path('logout/', auth_views.LogoutView.as_view(), name='logout'),  # 로그아웃 URL 연결
    path('chat/', views.chat_view, name='chat'),  # 채팅 페이지 URL 연결
]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
