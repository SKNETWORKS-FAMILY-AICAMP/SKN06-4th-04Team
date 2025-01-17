from django import forms
from django.contrib.auth.forms import UserCreationForm, UserChangeForm, AuthenticationForm
from django.contrib.auth import get_user_model

class BaseUserForm(forms.ModelForm): # 공통 필드 정의
    email = forms.EmailField(label="이메일", required=True)
    name = forms.CharField(label="이름", max_length=30, required=True)
    birthdate = forms.DateField(label="생년월일", widget=forms.SelectDateWidget(years=range(1900, 2025)))  # 생년월일
    phone_number = forms.CharField(label="휴대폰 번호", max_length=15, required=True)  # 휴대폰번호

    class Meta:
        model = get_user_model()
        fields = ("email", "name", "birthdate", "phone_number")

    widgets = {
        'birthdate': forms.DateInput(attrs={'type': 'date'}),
        'phone_number': forms.TextInput(attrs={'placeholder': '010-1234-5678'}),
    }

    help_texts = {
        "email": "유효한 이메일 주소를 입력해주세요.",
        "name": "실명을 입력해주세요.",
        "birthdate": "생년월일을 선택해주세요.",
        "phone_number": "휴대폰 번호를 입력해주세요. (예: 010-1234-5678)",
    }

class CustomAuthenticationForm(AuthenticationForm):
    username = forms.CharField(label="아이디")
    password = forms.CharField(label="비밀번호", widget=forms.PasswordInput)

class CustomUserCreationForm(BaseUserForm, UserCreationForm):
    password1 = forms.CharField(
        label="비밀번호",
        widget=forms.PasswordInput,
        help_text="비밀번호는 6자 이상이어야 하며, 숫자만 포함되면 안 됩니다.",
    )
    password2 = forms.CharField(
        label="비밀번호 확인",
        widget=forms.PasswordInput,
        help_text="비밀번호를 다시 입력해주세요.",
    )

    class Meta(BaseUserForm.Meta):
        fields = BaseUserForm.Meta.fields + ("username", "password1", "password2")
        labels = {
            "username": "아이디",
        }
        help_texts = {
            "username": "10자 이하로 입력해주세요. 영어, 숫자 및 @/./+/-/_만 가능합니다.",
        }

class ProfileUpdateForm(BaseUserForm, UserChangeForm):
    class Meta(BaseUserForm.Meta):
        fields = BaseUserForm.Meta.fields + ("username",)
        labels = {
            "username": "아이디",
        }
        help_texts = {
            "username": "필수 항목. 150자 이하, 문자, 숫자 및 @/./+/-/_만 사용 가능합니다.",
        }
