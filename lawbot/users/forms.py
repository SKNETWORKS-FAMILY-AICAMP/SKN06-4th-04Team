from django import forms
from django.contrib.auth.forms import (
    UserCreationForm,
    UserChangeForm,
    AuthenticationForm,
)
from django.contrib.auth import get_user_model
from datetime import datetime
from django.core.exceptions import ValidationError


# 오류 메시지를 한 곳에 모음
ERROR_MESSAGES = {
    "invalid_login": "⚠️ 아이디 또는 비밀번호가 올바르지 않습니다.",
    "password_minlength": "⚠️ 비밀번호가 6자리 이상이어야 합니다.",
    "password_mismatch": "⚠️ 비밀번호가 일치하지 않습니다.",
    "password_confirmation_required": "⚠️ 비밀번호 확인을 입력해주세요.",
    "username_exists": "⚠️ 이미 사용 중인 아이디입니다.",
}


class BaseUserForm(forms.ModelForm):
    current_year = datetime.now().year

    name = forms.CharField(label="이름", max_length=20, required=True)
    email = forms.EmailField(label="이메일", required=True)
    birthdate = forms.DateField(
        label="생년월일", widget=forms.SelectDateWidget(years=range(1900, current_year))
    )
    phone_number = forms.CharField(label="휴대폰번호", max_length=11, required=True)


    class Meta:
        model = get_user_model()
        fields = ("email", "name", "birthdate", "phone_number", "profile_picture")
        widgets = {
            "birthdate": forms.DateInput(attrs={"type": "date"}),
            "phone_number": forms.TextInput(attrs={"placeholder": "01011112222"}),
        }
        help_texts = {
            "email": "유효한 이메일 주소를 입력해주세요.",
            "name": "이름을 입력해주세요.",
            "birthdate": "생년월일을 선택해주세요.",
            "phone_number": "휴대폰번호를 입력해주세요. (예: 01011112222)",
        }



class CustomAuthenticationForm(AuthenticationForm):
    username = forms.CharField(label="아이디")
    password = forms.CharField(label="비밀번호", widget=forms.PasswordInput)
    error_messages = {
        "invalid_login": ERROR_MESSAGES["invalid_login"],
    }


class CustomUserCreationForm(BaseUserForm, UserCreationForm):
    username = forms.CharField(
        label="아이디",
        widget=forms.TextInput,
        help_text="10자 이하로 입력해주세요. 영어, 숫자 및 @/./+/-/_만 가능합니다.",
    )
    password1 = forms.CharField(
        label="비밀번호",
        widget=forms.PasswordInput,
        help_text="비밀번호는 6자 이상이어야 합니다.",
    )
    password2 = forms.CharField(
        label="비밀번호 확인",
        widget=forms.PasswordInput,
        help_text="비밀번호를 다시 입력해주세요.",
        error_messages={
            'password_mismatch': ERROR_MESSAGES['password_confirmation_required']
        }
    )

    class Meta(BaseUserForm.Meta):
        fields = ("username", "password1", "password2") + BaseUserForm.Meta.fields

    def clean_username(self):
        username = self.cleaned_data.get("username")
        if (username and self._meta.model.objects.filter(username__iexact=username).exists()):
            self._update_errors(ValidationError({"username": ERROR_MESSAGES["username_exists"]}))
        else:
            return username

    
    def clean_password1(self):
        password1 = self.cleaned_data.get("password1")

        if password1 and len(password1) < 6:
            raise forms.ValidationError(ERROR_MESSAGES['password_minlength'])
        return password1
        
    def clean_password2(self):
        password1 = self.cleaned_data.get("password1")
        password2 = self.cleaned_data.get("password2")
        
        if password1 and password2 and password1 != password2:
            raise forms.ValidationError(ERROR_MESSAGES['password_mismatch'])
        if password2 and len(password2) < 6:
            raise forms.ValidationError(ERROR_MESSAGES['password_minlength'])
        return password2
        
class ProfileUpdateForm(BaseUserForm, UserChangeForm):
    password = None
    new_password = forms.CharField(
        label="새 비밀번호",
        widget=forms.PasswordInput,
        required=False,
        help_text="새 비밀번호를 입력해주세요.",
    )
    confirm_password = forms.CharField(
        label="새 비밀번호 확인",
        widget=forms.PasswordInput,
        required=False,
        help_text="새 비밀번호 확인을 입력해주세요.",
    )
    profile_picture = forms.ImageField(
        label="프로필 사진",
        required=False,
        widget=forms.FileInput(attrs={'class': 'form-control'})
    )
    class Meta(BaseUserForm.Meta):
        fields = ("profile_picture",) + BaseUserForm.Meta.fields

        
    def clean(self):
        cleaned_data = super().clean()
        new_password = cleaned_data.get("new_password")
        confirm_password = cleaned_data.get("confirm_password")

        if new_password and not confirm_password:
            raise forms.ValidationError(ERROR_MESSAGES["password_confirmation_required"])
        if new_password and len(new_password) < 6:
            raise forms.ValidationError(ERROR_MESSAGES['password_minlength'])
        if new_password and confirm_password and new_password != confirm_password:
            raise forms.ValidationError(ERROR_MESSAGES["password_mismatch"])

        return cleaned_data

    def save(self, commit=True):
        user = super().save(commit=False)
        new_password = self.cleaned_data.get("new_password")

        if new_password:
            user.set_password(new_password)

        # 프로필 사진이 None이면 기본 이미지로 설정
        if self.cleaned_data.get('profile_picture') is None:
            user.profile_picture = None


        if commit:
            user.save()
        return user

