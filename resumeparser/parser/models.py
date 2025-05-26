from django.db import models

class ParsedResume(models.Model):
    name = models.CharField(max_length=255, blank=True)
    email = models.EmailField(blank=True)
    phone = models.CharField(max_length=20, blank=True)
    location = models.CharField(max_length=255, blank=True)
    experience_years = models.FloatField(default=0)
    skills = models.JSONField(default=list, blank=True)
    current_role = models.CharField(max_length=255, blank=True)
    company = models.CharField(max_length=255, blank=True)
    education = models.JSONField(default=list, blank=True)
    projects = models.JSONField(default=list, blank=True)
    work_experience = models.JSONField(default=list, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name or "Resume"
