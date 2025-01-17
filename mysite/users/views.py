# users/views.py
from django.shortcuts import render, redirect
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import login, logout, update_session_auth_hash
from django.urls import reverse
from .forms import CustomUserCreationForm, CustomAuthenticationForm, ProfileUpdateForm  # 커스텀 폼 가져오기
from django.views import View
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic import TemplateView
from django.contrib.auth import views as auth_views
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .models import User

# LoginRequiredMixin을 사용하여 로그인되지 않은 사용자는 로그인 페이지로 리디렉션
class HomeView(LoginRequiredMixin, TemplateView):
    template_name = "users/home.html"
    login_url = '/accounts/login/'  # 로그인 페이지로 리디렉션

# 기본 LoginView를 커스터마이징된 폼을 사용하도록 설정
class CustomLoginView(auth_views.LoginView):
    form_class = CustomAuthenticationForm

def register(request):
    if request.method == "POST":
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)  # 회원가입 후 자동 로그인
            return redirect("home")  # 회원가입 후 리다이렉트할 페이지 (홈페이지 등)
    else:
        form = CustomUserCreationForm()
    return render(request, "users/register.html", {"form": form})

def home(request):
    return render(request, "users/home.html")

def logout_view(request):
    logout(request)  # 로그아웃 실행
    return redirect(reverse("home"))  # 로그아웃 후 홈으로 리다이렉트

@login_required
def profile(request):
    user = request.user
    
    if request.method == "POST":
        form = ProfileUpdateForm(request.POST, instance=user)
        if form.is_valid():
            # 프로필 정보 업데이트
            user = form.save(commit=False)
            
            # 새 비밀번호가 입력된 경우 비밀번호 변경
            new_password = request.POST.get('new_password')
            if new_password:
                user.set_password(new_password)
                update_session_auth_hash(request, user)  # 세션 유지
                messages.success(request, '비밀번호가 성공적으로 변경되었습니다.')
            
            user.save()
            messages.success(request, "프로필이 성공적으로 업데이트되었습니다!")
            return redirect("profile")
    else:
        form = ProfileUpdateForm(instance=user)
    
    return render(request, "users/profile.html", {"form": form})






