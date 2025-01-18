# users/views.py
from django.shortcuts import render, redirect
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import login, logout, update_session_auth_hash
from django.urls import reverse
from .forms import (
    CustomUserCreationForm,
    CustomAuthenticationForm,
    ProfileUpdateForm,
) 
from django.contrib.auth import views as auth_views
from django.contrib.auth.decorators import login_required
from django.contrib import messages



class CustomLoginView(auth_views.LoginView):
    form_class = CustomAuthenticationForm


def register(request):
    if request.method == "POST":
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect("home")
    else:
        form = CustomUserCreationForm()
    return render(request, "users/register.html", {"form": form})


def logout_view(request):
    logout(request)
    return redirect(reverse("home"))

@login_required
def profile(request):
    user = request.user
    form = ProfileUpdateForm(instance=user)

    return render(request, "users/profile.html", {"form": form})

@login_required
def profile_update(request):
    user = request.user
    if request.method == "POST":
        form = ProfileUpdateForm(request.POST, instance=user)
        if form.is_valid():
            user = form.save(commit=False)

            new_password = request.POST.get("new_password")
            if new_password:
                user.set_password(new_password)
                update_session_auth_hash(request, user)
                messages.success(request, "비밀번호가 성공적으로 변경되었습니다.")

            user.save()
            messages.success(request, "프로필이 성공적으로 업데이트되었습니다!")
            return redirect("profile")
    else:
        form = ProfileUpdateForm(instance=user)

    return render(request, "users/profile.html", {"form": form,"edit_mode": True})
