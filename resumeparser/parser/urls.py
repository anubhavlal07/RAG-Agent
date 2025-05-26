from django.urls import path
from .views import ResumeParserAPIView
urlpatterns = [
    path('parse/', ResumeParserAPIView.as_view(), name='resume-parse'),
]
