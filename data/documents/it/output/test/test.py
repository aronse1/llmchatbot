
import re
# input = "Ein wesentliches Erfolgskriterium für ein Unternehmen ist die Güte der unternehmensinternen und -übergreifenden Geschäftsprozesse. Dazu gehört das Zusammenspiel der einzelnen Tätigkeiten innerhalb des Unternehmens sowie mit seinen Kunden und Lieferanten."
# keyworddstring = "unternehmensinterne und -übergreifende Geschäftsprozesse"

# keywords = [kw.strip() for kw in keyworddstring.split(",") if kw.strip()]

# # Suche nach ganzen Phrasen im Text (Case-Insensitive)
# found_keywords = [kw for kw in keywords if kw.lower() in input.lower()]

# print(found_keywords)

# percentage = (len(found_keywords) / len(keywords)) * 100
# print(percentage)

# import re

# def find_keywords(input_text, keyword_string):
#     # Zerlege die Keyword-Zeichenkette in eine Liste
#     keywords = [kw.strip() for kw in keyword_string.split(",") if kw.strip()]

#     # Erstelle eine Regex, die flexibel nach Wortformen sucht
#     pattern = r"\b(" + "|".join(re.escape(kw) for kw in keywords) + r")\b"

#     # Suche alle Vorkommen (case-insensitive)
#     matches = re.findall(pattern, input_text, re.IGNORECASE)

#     # Berechne den Prozentsatz der gefundenen Keywords
#     percentage = (len(matches) / len(keywords)) * 100 if keywords else 0

#     return matches, percentage

# # Beispiel
# input_text = "Ein wesentliches Erfolgskriterium für ein Unternehmen ist die Güte der unternehmensinternen und -übergreifenden Geschäftsprozesse."
# keyword_string = "unternehmensinterne -übergreifende Geschäftsprozesse"

# matches, percentage = find_keywords(input_text, keyword_string)

# print("Gefundene Keywords:", matches)
# print("Prozentsatz:", percentage)
