# users/urls.py
from django.urls import path
from .views import  profile, register

urlpatterns = [
    path("register/", register, name="register"),
    path('profile/', profile, name='profile'),
]

