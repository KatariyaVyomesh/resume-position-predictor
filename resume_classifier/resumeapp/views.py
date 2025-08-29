# from django.shortcuts import render

# # Create your views here.
# from django.shortcuts import render
# import joblib

# # Load model
# model = joblib.load("resumeapp/resume_model.pkl")

# def home(request):
#     return render(request, "home.html")

# def predict(request):
#     if request.method == "POST":
#         resume_text = request.POST.get("resume_text")
#         prediction = model.predict([resume_text])[0]
#         return render(request, "home.html", {"prediction": prediction, "resume_text": resume_text})
#     return render(request, "home.html")
from django.shortcuts import render
import joblib
import os

# Load model only once
MODEL_PATH = os.path.join(os.path.dirname(__file__), "resume_model.pkl")
model = joblib.load(MODEL_PATH)

def home(request):
    return render(request, "home.html")

def predict(request):
    if request.method == "POST":
        resume_text = request.POST.get("resume_text")
        prediction = model.predict([resume_text])[0]
        return render(request, "home.html", {"prediction": prediction, "resume_text": resume_text})
    return render(request, "home.html")
