
from django.http import HttpResponse
from django.shortcuts import render, redirect
import requests


def signup(request):
    return render(request,"signup.html")
  
def index(request):
    return render(request,"index.html")

def login(request):
    return render(request,"login2.html")

def home(request):
    return render(request,"home.html")

def start(request):
    return render(request,"start.html")
def saleforecasting(request):
    return render(request,"saleforecasting.html")
def time(request):
    return render(request,"timeseries2.html")
def output2(request):
    return render(request,"timeseires.html")