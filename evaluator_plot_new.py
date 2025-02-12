# import os
# import json
# import numpy as np
# import matplotlib.pyplot as plt
# import scipy.stats as stats


# folder_path = "C:\\Users\\arons\\Desktop\\Studienarbeit\\chatbot_poc\\data\\documents\\it\\output\\output_json\\"


# bleu_scores = []

# for filename in os.listdir(folder_path):
#     if filename.endswith(".json"):
#         file_path = os.path.join(folder_path, filename)
#         with open(file_path, "r", encoding="utf-8") as file:
#             data = json.load(file)
#             # BLEU-Scores extrahieren
#             for entry in data:
#                 try:
                    
#                     bleu_scores.append(entry["bleu_score"])
#                 except:
#                     continue

# # BLEU-Scores in NumPy-Array umwandeln   
# bleu_scores = np.array(bleu_scores)

# # Berechnung von Mittelwert und Standardabweichung
# mean_bleu = np.mean(bleu_scores)
# std_bleu = np.std(bleu_scores)

# # Werte für die Normalverteilung berechnen
# x = np.linspace(min(bleu_scores), max(bleu_scores), 100)
# y = stats.norm.pdf(x, mean_bleu, std_bleu)

# # Plot erstellen
# plt.figure(figsize=(10, 6))
# plt.hist(bleu_scores, bins=20, density=True, alpha=0.6, color='b', label="METEOR Score Histogram")
# plt.plot(x, y, 'r-', label="Normalverteilung")

# # Mittelwert und Standardabweichung als vertikale Linien einzeichnen
# plt.axvline(mean_bleu, color='g', linestyle='dashed', linewidth=1, label="Mittelwert")
# #plt.axvline(mean_bleu - std_bleu, color='orange', linestyle='dashed', linewidth=2, label="1 Std. Abw.")
# plt.axvline(mean_bleu + std_bleu, color='orange', linestyle='dashed', linewidth=1, label="1 Std. Abw.")

# # Optional: Text hinzufügen
# plt.text(mean_bleu, max(y) * 0.9, f"Ø = {mean_bleu:.2f}", color="g", ha="center", fontsize=12)
# #plt.text(mean_bleu - std_bleu, max(y) * 0.7, f"-1σ", color="orange", ha="center", fontsize=12)
# plt.text(mean_bleu + std_bleu, max(y) * 0.7, f"+1σ", color="orange", ha="center", fontsize=12)

# plt.title("Verteilung der BLEU-Scores mit Normalverteilung")
# plt.xlabel("BLEU-Score")
# plt.ylabel("Häufigkeit")
# plt.legend()

# plt.show()





# ##########################################################################################################################################

# json_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".json")]
# json_files[:5]


# # BLEU-Scores nach Schwierigkeitsgrad (difficulty) und Stage kategorisieren
# bleu_by_difficulty = {"leicht": [], "mittel": [], "schwer": []}
# bleu_by_stage = {}

# # Alle JSON-Dateien einlesen und BLEU-Scores extrahieren
# for file_path in json_files:
#     with open(file_path, "r", encoding="utf-8") as file:
#         data = json.load(file)
#         for entry in data:
#             bleu_score = entry.get("meteor_score", None)
#             if bleu_score is not None:
#                 bleu_score*= 100
#             difficulty = entry.get("difficulty", "unbekannt")
#             stage = entry.get("stage", "unbekannt")

#             if bleu_score is not None:
#                 # Nach Schwierigkeit kategorisieren
#                 if difficulty in bleu_by_difficulty:
#                     bleu_by_difficulty[difficulty].append(bleu_score)

#                 # Nach Stufe kategorisieren
#                 if stage not in bleu_by_stage:
#                     bleu_by_stage[stage] = []
#                 bleu_by_stage[stage].append(bleu_score)

# # Gemeinsame Achseneinstellungen berechnen
# all_bleu_scores = [score for scores in bleu_by_difficulty.values() for score in scores]
# for scores in bleu_by_stage.values():
#     all_bleu_scores.extend(scores)

# min_bleu = min(all_bleu_scores)
# max_bleu = max(all_bleu_scores)

# # Maximale Dichte für Normalisierung der y-Achse berechnen
# max_density = 0
# for scores in bleu_by_difficulty.values():
#     if scores:
#         kde = stats.gaussian_kde(scores)
#         max_density = max(max_density, max(kde(np.linspace(min_bleu, max_bleu, 100))))

# for scores in bleu_by_stage.values():
#     if scores:
#         kde = stats.gaussian_kde(scores)
#         max_density = max(max_density, max(kde(np.linspace(min_bleu, max_bleu, 100))))

# # Erstellen der Plots mit normalisierten Achsen
# # Erstellen der Plots NUR für Difficulty (ohne Stage-Diagramme)
# fig, axes = plt.subplots(1, len(bleu_by_difficulty), figsize=(15, 5))

# # Plot für Difficulty
# for idx, (difficulty, scores) in enumerate(bleu_by_difficulty.items()):
#     if scores:
#         mean_bleu = np.mean(scores)
#         std_bleu = np.std(scores)
#         x = np.linspace(min_bleu, max_bleu, 100)
#         y = stats.norm.pdf(x, mean_bleu, std_bleu)

#         ax = axes[idx]
#         ax.hist(scores, bins=20, density=True, alpha=0.6, color='b', label="METEOR Histogramm")
#         ax.plot(x, y, 'r-', label="Normalverteilung")
#         ax.set_title(f"METEOR-Verteilung für {difficulty}")
#         ax.set_xlabel("METEOR-Score")
#         ax.set_ylabel("Häufigkeit")
#         ax.legend()
#         ax.set_xlim(min_bleu, max_bleu)
#         ax.set_ylim(0, max_density)

# plt.tight_layout()
# plt.show()

##################################################################################################################


# import numpy as np
# import matplotlib.pyplot as plt
# import scipy.stats as stats
# import json
# import os

# # Datei laden
# folder_path = "C:\\Users\\arons\\Desktop\\Studienarbeit\\chatbot_poc\\data\\documents\\it\\output\\output_json\\"



# # ROUGE-Scores extrahieren
# rouge_r = []
# rouge_p = []
# rouge_f1 = []

# for filename in os.listdir(folder_path):
#     if filename.endswith(".json"):
#         file_path = os.path.join(folder_path, filename)
#         with open(file_path, "r", encoding="utf-8") as file:
#             data = json.load(file)
#             for entry in data:
#                 try:
#                     rouge_scores = entry["rouge_scores"][0]  # Da es eine Liste mit einem Element ist
#                     #print(rouge_scores)
#                     for metric in rouge_scores.keys():
#                         if metric == "rouge-1":
#                             rouge_r.append(rouge_scores[metric]["f"])
#                         if metric == "rouge-2":
#                             rouge_p.append(rouge_scores[metric]["f"])
#                         if metric == "rouge-l":
#                             rouge_f1.append(rouge_scores[metric]["f"])
#                 except:
#                     continue


# # # Konvertieren in NumPy-Arrays
# rouge_r = np.array(rouge_r)
# rouge_p = np.array(rouge_p)
# rouge_f1 = np.array(rouge_f1)

# # Mittelwerte und Standardabweichungen berechnen
# mean_r, std_r = np.mean(rouge_r), np.std(rouge_r)
# mean_p, std_p = np.mean(rouge_p), np.std(rouge_p)
# mean_f1, std_f1 = np.mean(rouge_f1), np.std(rouge_f1)

# # Werte für Normalverteilungen berechnen
# x_r = np.linspace(min(rouge_r), max(rouge_r), 100)
# y_r = stats.norm.pdf(x_r, mean_r, std_r)

# x_p = np.linspace(min(rouge_p), max(rouge_p), 100)
# y_p = stats.norm.pdf(x_p, mean_p, std_p)

# x_f1 = np.linspace(min(rouge_f1), max(rouge_f1), 100)
# y_f1 = stats.norm.pdf(x_f1, mean_f1, std_f1)

# # Subplots erstellen
# # Gemeinsame Achsenskalierung für besseren Vergleich
# common_x_min = min(np.min(rouge_r), np.min(rouge_p), np.min(rouge_f1))
# common_x_max = max(np.max(rouge_r), np.max(rouge_p), np.max(rouge_f1))
# common_y_max = max(max(y_r), max(y_p), max(y_f1)) 
# common_y_max += 4.0

# # Subplots mit normalisierten Achsen erstellen
# fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)

# # Plot für Recall
# axes[0].hist(rouge_r, bins=20, density=True, alpha=0.6, color='b', label="Rouge-1 F1-Score")
# axes[0].plot(x_r, y_r, 'r-', label="Normalverteilung")
# axes[0].axvline(mean_r, color='g', linestyle='dashed', linewidth=1, label="Mittelwert")
# axes[0].axvline(mean_r + std_r, color='orange', linestyle='dashed', linewidth=1, label="1 Std. Abw.")
# axes[0].set_title("Verteilung des ROUGE-1 F1 Score")
# axes[0].set_xlabel("F1-Score")
# axes[0].set_ylabel("Häufigkeit")
# axes[0].legend()
# axes[0].set_xlim(common_x_min, common_x_max)
# axes[0].set_ylim(0, common_y_max)

# # Plot für Precision
# axes[1].hist(rouge_p, bins=20, density=True, alpha=0.6, color='b', label="Rouge-2 F1-Score")
# axes[1].plot(x_p, y_p, 'r-', label="Normalverteilung")
# axes[1].axvline(mean_p, color='g', linestyle='dashed', linewidth=1, label="Mittelwert")
# axes[1].axvline(mean_p + std_p, color='orange', linestyle='dashed', linewidth=1, label="1 Std. Abw.")
# axes[1].set_title("Verteilung des ROUGE-2 F1 Scores")
# axes[1].set_xlabel("F1-Score")
# axes[1].legend()
# axes[1].set_xlim(common_x_min, common_x_max)
# axes[1].set_ylim(0, common_y_max)

# # Plot für F1-Score
# axes[2].hist(rouge_f1, bins=20, density=True, alpha=0.6, color='b', label="Rouge-l F1-Score")
# axes[2].plot(x_f1, y_f1, 'r-', label="Normalverteilung")
# axes[2].axvline(mean_f1, color='g', linestyle='dashed', linewidth=1, label="Mittelwert")
# axes[2].axvline(mean_f1 + std_f1, color='orange', linestyle='dashed', linewidth=1, label="1 Std. Abw.")
# axes[2].set_title("Verteilung der ROUGE-l F1-Scores")
# axes[2].set_xlabel("F1-Score")
# axes[2].legend()
# axes[2].set_xlim(common_x_min, common_x_max)
# axes[2].set_ylim(0, common_y_max)

# plt.tight_layout()
# plt.show()




import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import json
import os
# Datei laden
folder_path = "C:\\Users\\arons\\Desktop\\Studienarbeit\\chatbot_poc\\data\\documents\\it\\output\\output_json\\"


# ROUGE-Scores nach Schwierigkeitsgrad (difficulty) kategorisieren
rouge_by_difficulty = {"leicht": {"r": [], "p": [], "f": []}, 
                       "mittel": {"r": [], "p": [], "f": []}, 
                       "schwer": {"r": [], "p": [], "f": []}}

# JSON-Daten durchgehen und ROUGE-Scores extrahieren

for filename in os.listdir(folder_path):
    if filename.endswith(".json"):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
            for entry in data:
                try:
                    difficulty = entry.get("difficulty", "unbekannt")
                    if difficulty in rouge_by_difficulty:
                        rouge_scores = entry["rouge_scores"][0]
                        for metric in rouge_scores.keys():
                            if metric == "rouge-1":
                                rouge_by_difficulty[difficulty]["r"].append(rouge_scores[metric]["f"])
                            if metric == "rouge-2":
                                rouge_by_difficulty[difficulty]["p"].append(rouge_scores[metric]["f"])
                            if metric == "rouge-l":
                                rouge_by_difficulty[difficulty]["f"].append(rouge_scores[metric]["f"])
                except:
                    continue



# Gemeinsame Achseneinstellungen berechnen
all_rouge_scores = []
for scores in rouge_by_difficulty.values():
    all_rouge_scores.extend(scores["r"])
    all_rouge_scores.extend(scores["p"])
    all_rouge_scores.extend(scores["f"])

min_rouge = min(all_rouge_scores)
max_rouge = max(all_rouge_scores)

# Maximale Dichte für Normalisierung der y-Achse berechnen
max_density = 0
for scores in rouge_by_difficulty.values():
    for key in ["r", "p", "f"]:
        if scores[key]:
            kde = stats.gaussian_kde(scores[key])
            max_density = max(max_density, max(kde(np.linspace(min_rouge, max_rouge, 100))))

# Erstellen der Plots für jeden ROUGE-Wert (Recall, Precision, F1) und jede Schwierigkeit
fig, axes = plt.subplots(3, len(rouge_by_difficulty), figsize=(18, 12), sharex=True, sharey=True)

score_types = ["r", "p", "f"]
score_labels = ["Rouge-1 F1", "Rouge-2 F1", "Rouge-l F1"]
colors = ["b", "g", "r"]

for row, (score_type, score_label, color) in enumerate(zip(score_types, score_labels, colors)):
    for col, (difficulty, scores) in enumerate(rouge_by_difficulty.items()):
        ax = axes[row, col]

        if scores[score_type]:
            mean_score = np.mean(scores[score_type])
            std_score = np.std(scores[score_type])
            x = np.linspace(min_rouge, max_rouge, 100)
            y = stats.norm.pdf(x, mean_score, std_score)

            ax.hist(scores[score_type], bins=20, density=True, alpha=0.6, color=color, label=f"{score_label} Histogramm")
            ax.plot(x, y, color + "-", label=f"{score_label} Normalverteilung")
            ax.axvline(mean_score, color="black", linestyle="dashed", linewidth=1, label="Mittelwert")

        ax.set_title(f"{score_label} für {difficulty}")
        ax.set_xlabel("ROUGE-Score")
        ax.set_ylabel("Häufigkeit")
        ax.legend()
        ax.set_xlim(min_rouge, max_rouge)
        ax.set_ylim(0, max_density)

plt.tight_layout()
plt.show()
##############################################################################################################################################

# import json
# import matplotlib.pyplot as plt
# import os

# # Datei laden
# folder_path = "C:\\Users\\arons\\Desktop\\Studienarbeit\\chatbot_poc\\data\\documents\\it\\output\\output_json\\"

# categories = {
#     "Kontext Stimmt, aber Antwort unschön": 0,
#     "Falsche Antwort": 0,
#     "Gute Antwort, richtige Informationen": 0,
#     "Gute Antwort, aber Falschinformationen": 0
# }

# bleu_threshold = 20
# meteor_threshold = 0.3
# rouge_threshold = 0.3

# # Daten durchgehen und 
# for filename in os.listdir(folder_path):
#     if filename.endswith(".json"):
#         file_path = os.path.join(folder_path, filename)
#         with open(file_path, "r", encoding="utf-8") as file:
#             data = json.load(file)
#             for entry in data:
#                 try: 
#                     bleu = entry["bleu_score"]
#                     meteor = entry["meteor_score"]
#                     rouge_f1 = entry["rouge_scores"][0]["rouge-1"]["f"]
#                     faithfulness = entry["passed"]


#                     high_metrics = (bleu >= bleu_threshold) and (meteor >= meteor_threshold) and (rouge_f1 >= rouge_threshold)
#                 except:
#                     continue
#                 if not high_metrics and faithfulness:
#                     categories["Kontext Stimmt, aber Antwort unschön"] += 1
#                     print(entry["query"])
#                     print(entry["response"])
#                     print(entry["reference"])
#                     print("----------------------------------------------------------------")
#                 elif not high_metrics and not faithfulness:
#                     categories["Falsche Antwort"] += 1
#                 elif high_metrics and faithfulness:
#                     categories["Gute Antwort, richtige Informationen"] += 1
#                 elif high_metrics and not faithfulness:
#                     categories["Gute Antwort, aber Falschinformationen"] += 1


# plt.figure(figsize=(10, 6))
# plt.bar(categories.keys(), categories.values(), color=["blue", "red", "green", "orange"])
# plt.ylabel("Anzahl der Fälle")
# plt.title("Kategorisierung der Antworten im RAG-System")
# plt.xticks(rotation=10)
# plt.grid(axis="y", linestyle="--", alpha=0.7)

# # Diagramm anzeigen
# plt.show()


###########################################################################################################################

# import os
# import json
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Ordner mit den JSON-Dateien
# extract_folder ="C:\\Users\\arons\\Desktop\\Studienarbeit\\chatbot_poc\\data\\documents\\it\\output\\output_json\\"


# # Alle JSON-Dateien einlesen
# data = []
# for file_name in os.listdir(extract_folder):
#     if file_name.endswith(".json"):
#         file_path = os.path.join(extract_folder, file_name)
#         with open(file_path, 'r', encoding='utf-8') as file:
#             data.extend(json.load(file))

# # Daten in DataFrame umwandeln
# df = pd.DataFrame(data)

# # Relevante Metriken extrahieren
# df_scores = df[['query', 'bleu_score', 'meteor_score', 'keyword-percentage']].copy()
# df_scores['meteor_score'] *= 100
# # Durchschnittswerte pro Frage berechnen
# df_avg_scores = df_scores.groupby('query').mean()

# # Heatmap erstellen
# plt.figure(figsize=(15, 8))
# avg_scores = df_scores.groupby('query')[['bleu_score', 'meteor_score', 'keyword-percentage']].mean()
# sns.heatmap(avg_scores, annot=True, cmap='YlOrRd', fmt='.3f')
# plt.title('Average Scores per Question')
# plt.xticks(rotation=45)
# plt.yticks(fontsize=50)
# plt.tight_layout()
# plt.show()





############################################################################################################



# import zipfile
# import os
# import json
# import pandas as pd
# import matplotlib.pyplot as plt

# # Pfad zur hochgeladenen Datei
# extract_folder ="C:\\Users\\arons\\Desktop\\Studienarbeit\\chatbot_poc\\data\\documents\\it\\output\\output_json\\"
# data = []
# for file in os.listdir(extract_folder):
#     if file.endswith(".json"):
#         file_path = os.path.join(extract_folder, file)
#         with open(file_path, "r", encoding="utf-8") as f:
#             json_data = json.load(f)
#             data.extend(json_data)

# # In DataFrame umwandeln
# df = pd.DataFrame(data)

# # Durchschnittliche Scores pro Stage und Schwierigkeitsgrad berechnen
# df_avg_scores = df.groupby(["stage", "difficulty"]).agg(
#     bleu_score=("bleu_score", "mean"),
#     meteor_score=("meteor_score", "mean"),
#     keyword_percentage=("keyword-percentage", "mean")
# ).reset_index()

# # Extrahieren des ROUGE-1-Scores aus den verschachtelten Dictionaries
# df["rouge_1_score"] = df["rouge_scores"].apply(
#     lambda x: x[0]["rouge-1"]["f"] if isinstance(x, list) and len(x) > 0 else None
# )
# df_avg_scores["rouge_1_score"] = df.groupby(["stage", "difficulty"])["rouge_1_score"].mean().values

# # Balkendiagramme erstellen
# fig, axes = plt.subplots(4, 1, figsize=(12, 24))
# metrics = ["bleu_score", "meteor_score", "rouge_1_score", "keyword_percentage"]
# titles = ["Durchschnittlicher BLEU-Score", "Durchschnittlicher METEOR-Score", "Durchschnittlicher ROUGE-1-Score", "Durchschnittliche Keyword Treffer"]

# for i, metric in enumerate(metrics):
#     df_pivot = df_avg_scores.pivot(index="stage", columns="difficulty", values=metric)
#     df_pivot.plot(kind="bar", ax=axes[i])
    
#     axes[i].set_xlabel("Stage")
#     axes[i].set_ylabel("Score")
#     axes[i].set_title(titles[i])
#     axes[i].legend(title="Schwierigkeit")
#     axes[i].grid(axis="y")

# plt.tight_layout()
# plt.show()
