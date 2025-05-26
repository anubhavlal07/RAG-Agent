from django.contrib import admin
from .models import ParsedResume

@admin.register(ParsedResume)
class ParsedResumeAdmin(admin.ModelAdmin):
    list_display = ('name', 'email', 'phone', 'uploaded_at')
    search_fields = ('name', 'email', 'skills')
    readonly_fields = ('uploaded_at',)
