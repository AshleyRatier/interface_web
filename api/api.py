from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import re

# Initialisation de l'application FastAPI
app = FastAPI()

# Ajouter CORS pour autoriser le frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

# Détection du GPU si disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Chargement du modèle et du tokenizer 
model_path = "SOUMI23/generator_question_barthez"

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    return model, tokenizer

model, tokenizer = load_model()

# Définition des données envoyées par l'utilisateur
class QuestionRequest(BaseModel):
    text: str
    question_types: list[str]  

# Fonction pour découper un texte en phrases
def split_into_sentences(text):
    sentence_endings = re.compile(r'([.!?])\s*')
    sentences = sentence_endings.split(text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]

# Fonction pour générer des questions et éviter les doublons
def generate_multiple_questions(text, model, tokenizer, device, question_types):
    questions = []
    generated_questions_set = set()  

    # Découper le texte en plusieurs phrases 
    sentences = split_into_sentences(text)

    # Générer des questions pour chaque phrase
    for sentence in sentences:
        for q_type in question_types:
            input_text = f"<type:{q_type}> Contexte: {sentence}"

            inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)

            outputs = model.generate(
                **inputs,
                max_length=64,  
                do_sample=True,
                top_k=30,  
                top_p=0.90,
                temperature=0.6
            )

            generated_question = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # On évite les doublons
            if generated_question not in generated_questions_set:
                questions.append({"type": q_type, "question": generated_question})
                generated_questions_set.add(generated_question)  

    return questions

# Endpoint pour générer des questions
@app.post("/generate_questions")
def generate_question(request: QuestionRequest):
    generated_questions = generate_multiple_questions(request.text, model, tokenizer, device, request.question_types)
    return {"generated_questions": generated_questions}

# Endpoint de test
@app.get("/")
def home():
    return {"message": "API FastAPI pour la génération de questions"}
