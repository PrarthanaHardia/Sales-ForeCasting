from django.urls import path
from . import views
# from .views import LoginView
from .views import signup
urlpatterns =[
    # path("signup/",views.signup,name="signup"),
    path("login/",views.login,name="login"),
    path("home/",views.home,name="home"),
    path("start/",views.start,name="start"),
    path("",views.index,name="index"),
   path('signup/', signup, name='signup'),
]