# users/models.py
from django.contrib.auth.models import AbstractUser
from django.core.validators import RegexValidator
from django.db import models


class User(AbstractUser):
    name = models.CharField(max_length=20, null=True, blank=True)

    birthdate = models.DateField(null=True, blank=True)

    phone_number = models.CharField(
        verbose_name="휴대폰번호",
        max_length=11,
        validators=[
            RegexValidator(
                regex=r"^\d{10,11}$", message="전화번호는 '-' 없이 입력해 주세요."
            )
        ],
        null=True,
        blank=True,
    )
