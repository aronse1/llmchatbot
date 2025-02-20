import asyncio
from transformers import pipeline

class ClassifierManager:
    def __init__(self):
        self.lock = asyncio.Lock()
        self.intent_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        self.language_detector = pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection")
        self.translator =  pipeline("translation", model="facebook/m2m100_418M")
        #self.translator =  pipeline("translation", model="facebook/nllb-200-distilled-600M")
        self.intent_categories = {
            "study_topics": {
                "description": "Fragen zu Studium, Kursen und akademischen Anforderungen",
                "examples": [
                    "Wie viele Seiten brauche ich für die T1000?",
                    "Was sind die Anforderungen für die Bachelorarbeit?",
                    "Welche Voraussetzungen gibt es für das Praktikum?",
                    "Wie läuft die Prüfungsanmeldung ab?",
                    "Wo finde ich die Vorlesungsunterlagen?",
                    "Wann ist die Abgabe für die Projektarbeit?",
                    "Wie viele ECTS gibt es für den Kurs?",
                    "Was sind die Zulassungsvoraussetzungen?",
                    "Wie ist der Aufbau des Studiengangs?",
                    "Welche Fristen muss ich einhalten?",
                    "Wie erreiche ich Till Hänisch?"
                ]
            },
            "small_talk": {
                "description": "Allgemeine Konversation ohne akademischen Bezug",
                "examples": [
                    "Hallo",
                    "Wie geht es dir?",
                    "Was machst du am Wochenende?",
                    "Schönes Wetter heute",
                    "Kennst du einen guten Film?",
                    "Was ist dein Lieblingsessen?",
                    "Bist du ein Mensch?",
                    "Wie alt bist du?",
                    "Woher kommst du?",
                    "Was hältst du von...?",
                    "Magst du Sport?"
                ]
            }
        }

    # async def classify_intent(self, query):
    #     async with self.lock:  # Blockiert gleichzeitigen Zugriff
    #         intents = ["small_talk", "study_topics"]
    #         result = self.intent_classifier(query, candidate_labels=intents, hypothesis_template="This sentence belongs to the category: {}.")
    #         return result['labels'][0] if result['scores'][0] > 0.4 else "unclear"
    async def classify_intent(self, query: str, threshold: float = 0.4) -> str:
        """Klassifiziert den Intent mit verbesserter Kontextberücksichtigung"""
        async with self.lock:
            # Erstelle erweiterte Hypothesen für jede Kategorie
            hypotheses = []
            for intent, data in self.intent_categories.items():
                # Füge Beispiele als zusätzlichen Kontext hinzu
                hypothesis = f"This is a {intent} question. Similar examples: {'. '.join(data['examples'][:3])}"
                hypotheses.append(hypothesis)

           
            result = self.intent_classifier(
                query,
                candidate_labels=list(self.intent_categories.keys()),
                hypothesis_template="This input is similar to: {}"
            )

          
            print(f"Classification scores: {dict(zip(result['labels'], result['scores']))}")

            
            if "study_topics" in result['labels']:
                study_idx = result['labels'].index("study_topics")
                if result['scores'][study_idx] > 0.4:  
                    return "study_topics"

           
            return result['labels'][0] if result['scores'][0] > threshold else "unclear"
        
    async def detect_language(self, query):
        async with self.lock:
            result = self.language_detector(query)
            return result[0]['label']
        
    async def translate(self, text, source_lang, target_lang):
        async with self.lock:
            result = self.translator(text, 
                                src_lang=source_lang, 
                                tgt_lang=target_lang,
                                max_length=400)
            
            return result[0]['translation_text']
