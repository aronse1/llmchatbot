import json
import re
import os




def load_fachwoerter():
    script_dir = os.path.dirname(os.path.abspath(__file__))  
    json_path = os.path.join(script_dir, "helpers", "fachwörter.json")
    with open(json_path, "r", encoding="utf-8") as file:
        fachwoerter = json.load(file)

    return {key.lower(): [syn.lower() for syn in synonyms] for key, synonyms in fachwoerter.items()}

async def expand_query(query, fachwoerter): 
    query = query.lower()  
    for key, synonyms in fachwoerter.items():
        if re.search(rf"\b{re.escape(key)}\b", query):
            synonym_text = f"{key} ({', '.join(synonyms)})"
            query = re.sub(rf"\b{re.escape(key)}\b", synonym_text, query)
    return query

fachwoerter = load_fachwoerter()