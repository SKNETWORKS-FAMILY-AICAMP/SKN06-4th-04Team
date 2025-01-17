from django.contrib import admin
from .models import User

# 커스텀 User 모델을 관리자에 등록
admin.site.register(User)
