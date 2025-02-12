import asyncio
from transformers import pipeline

class ClassifierManager:
    def __init__(self):
        self.lock = asyncio.Lock()
        self.intent_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        self.language_detector = pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection")
        self.translator =  pipeline("translation", model="facebook/m2m100_418M")
        #self.translator =  pipeline("translation", model="facebook/nllb-200-distilled-600M")


    async def classify_intent(self, query):
        async with self.lock:  # Blockiert gleichzeitigen Zugriff
            intents = ["small_talk", "study_topics", "people_questions"]
            result = self.intent_classifier(query, candidate_labels=intents, hypothesis_template="This sentence belongs to the category: {}.")
            return result['labels'][0] if result['scores'][0] > 0.4 else "unclear"

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
