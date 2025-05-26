import os
import fitz
import docx2txt
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .parser_utils.llm_parser import extract_resume_data
from .models import ParsedResume
from .serializers import ResumeParseRequestSerializer, ParsedResumeSerializer

class ResumeParserAPIView(APIView):
    def post(self, request):
        print("Received POST request for resume parsing.")
        print("Request data:", request.data)
        serializer = ResumeParseRequestSerializer(data=request.data)
        if serializer.is_valid():
            text = serializer.validated_data.get("text")
            if serializer.validated_data.get("file"):
                file = serializer.validated_data["file"]
                file_name = file.name.lower()
                print(f"Processing uploaded file: {file_name}")
                try:
                    if file_name.endswith(".pdf"):
                        print("Extracting text from PDF file.")
                        with fitz.open(stream=file.read(), filetype="pdf") as doc:
                            text = "\n".join([page.get_text() for page in doc])
                    elif file_name.endswith(".docx"):
                        print("Extracting text from DOCX file.")
                        temp_path = f"/tmp/{file.name}"
                        with open(temp_path, "wb+") as temp_file:
                            for chunk in file.chunks():
                                temp_file.write(chunk)
                        text = docx2txt.process(temp_path)
                        os.remove(temp_path)
                    else:
                        print("Unsupported file type uploaded.")
                        return Response(
                            {"error": "Unsupported file type. Upload PDF or DOCX."},
                            status=status.HTTP_400_BAD_REQUEST
                        )
                except Exception as e:
                    print(f"Failed to extract text from file: {e}")
                    return Response(
                        {"error": f"Failed to extract text: {str(e)}"},
                        status=status.HTTP_500_INTERNAL_SERVER_ERROR
                    )
            try:
                print("Extracting structured data from resume text using LLM.")
                extracted_data = extract_resume_data(text)
                parsed_resume = ParsedResume.objects.create(
                    name=extracted_data.get("name", ""),
                    email=extracted_data.get("email", ""),
                    phone=extracted_data.get("phone", ""),
                    location=extracted_data.get("location", ""),
                    experience_years=extracted_data.get("experience_years", 0),
                    skills=extracted_data.get("skills", []),
                    current_role=extracted_data.get("current_role", ""),
                    company=extracted_data.get("company", ""),
                    education=extracted_data.get("education", []),
                    projects=extracted_data.get("projects", []),
                    work_experience=extracted_data.get("work_experience", []),
                )
                response_data = ParsedResumeSerializer(parsed_resume).data
                print("Resume parsed and saved successfully.")
                return Response(response_data, status=status.HTTP_201_CREATED)
            except Exception as e:
                print(f"LLM parsing failed: {e}")
                return Response(
                    {"error": f"LLM parsing failed: {str(e)}"},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
        print("Serializer errors:", serializer.errors)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
