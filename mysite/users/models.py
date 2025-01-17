# users/models.py
from django.contrib.auth.models import AbstractUser
from django.db import models

class User(AbstractUser):  # Django 기본 User 모델을 상속
    
    # 프로필 사진 추가
    # profile_picture = models.ImageField(upload_to='profile_pictures/', null=True, blank=True)

    name = models.CharField(max_length=100, null=True, blank=True)

    # 생년월일 추가
    birthdate = models.DateField(null=True, blank=True)

    # 휴대폰번호 추가
    phone_number = models.CharField(max_length=15, null=True, blank=True)
