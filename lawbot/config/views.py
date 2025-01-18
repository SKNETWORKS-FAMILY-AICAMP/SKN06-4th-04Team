# config/views.py
from django.shortcuts import render, redirect
from users.forms import CustomAuthenticationForm
from django.contrib.auth import login

    
def home_view(request):
    if request.user.is_authenticated:
        return render(request, 'chat/chatbot.html')
    
    if request.method == 'POST':
        form = CustomAuthenticationForm(data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user) 
            return redirect('chat')  
    else:
        form = CustomAuthenticationForm()

    return render(request, 'home.html', {'form': form})