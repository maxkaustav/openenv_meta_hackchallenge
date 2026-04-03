from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from functools import lru_cache
import timeit

@lru_cache(maxsize=None)
def train_tfidf():
    __data = [
    {"condition": "chest pain, heart attack, hypertension", "department": "Cardiology"},
    {"condition": "skin rash, acne, eczema", "department": "Dermatology"},
    {"condition": "headache, migraine, seizures", "department": "Neurology"},
    {"condition": "joint pain, arthritis, bone fracture", "department": "Orthopedics"},
    {"condition": "fever, infections, flu", "department": "General Medicine"},
    {"condition": "vision loss, cataract, glaucoma", "department": "Ophthalmology"},
    ]
    
    conditions = [item["condition"] for item in __data]
    departments = [item["department"] for item in __data]
    print("aaa")
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(conditions)

    return X,vectorizer,departments

def find_department_tfidf(query):
    model,vectorizer,departments = train_tfidf()
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, model)[0]

    best_idx = sims.argmax()
    
    return departments[best_idx]


find_department_tfidf("chest pain, heart attack, hypertension")
find_department_tfidf("chest pain, heart attack")
find_department_tfidf("headache, migraine, heart attack")

