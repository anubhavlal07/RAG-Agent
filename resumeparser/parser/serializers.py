from rest_framework import serializers
from .models import ParsedResume

class ResumeParseRequestSerializer(serializers.Serializer):
    file = serializers.FileField(required=False)
    text = serializers.CharField(required=False)

    def validate(self, data):
        if not data.get("file") and not data.get("text"):
            raise serializers.ValidationError("Provide either a resume file or resume text.")
        return data

class ParsedResumeSerializer(serializers.ModelSerializer):
    class Meta:
        model = ParsedResume
        fields = '__all__'
