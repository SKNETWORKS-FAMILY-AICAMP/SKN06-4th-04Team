# users/urls.py
from django.urls import path
from .views import  profile, register, profile_update

urlpatterns = [
    path("register/", register, name="register"),
    path('profile/', profile, name='profile'),
    path('profile/update', profile_update, name='profile_update'),
]

