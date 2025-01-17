# users/views.py
from django.shortcuts import render, redirect
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import login, logout
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
    user = request.user  # 현재 로그인한 사용자 가져오기
    
    if request.method == "POST":
        form = ProfileUpdateForm(request.POST, instance=user)
        if form.is_valid():
            form.save()
            messages.success(request, "프로필이 성공적으로 업데이트되었습니다!")
            return redirect("profile")  # 성공 후 마이페이지로 리디렉션
    else:
        form = ProfileUpdateForm(instance=user)  # 기존 사용자 정보 가져오기
    
    return render(request, "users/profile.html", {"form": form})
# class CustomLogoutView(View):
#     def get(self, request):
#         logout(request)  # 로그아웃 실행







