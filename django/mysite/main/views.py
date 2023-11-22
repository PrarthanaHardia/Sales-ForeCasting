
from django.http import HttpResponse
from django.shortcuts import render, redirect
import requests

# views.py





# def signup(request):
#     if request.method == 'POST':
#         form = SignupForm(request.POST)
#         if form.is_valid():
#             # Get data from the form
#             username = form.cleaned_data['username']
#             email = form.cleaned_data['email']
#             password = form.cleaned_data['password']

#             # Send request to Flask API
#             api_url = 'http://localhost:5000/signup'  # Update with your Flask API URL
#             data = {'username': username, 'email': email, 'password': password}
#             response = requests.post(api_url, json=data)

#             # Check the response from the API
#             if response.status_code == 200 and response.json()['success']:
#                 # Registration successful
#                 return redirect('success_page')  # Redirect to a success page
#             else:
#                 # Registration failed
#                 form.add_error(None, 'Registration failed. Please try again.')
#     else:
#         form = SignupForm()

#     return render(request, 'signup.html', {'form': form})



def signup(request):
    return render(request,"signup.html")
  
def index(request):
    return render(request,"index.html")

def loginn(request):
    return render(request,"login.html")

def home(request):
    return render(request,"home.html")

def start(request):
    return render(request,"start.html")