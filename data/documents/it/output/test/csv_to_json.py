import json
folder = "C:\\Users\\arons\\Desktop\\Studienarbeit\\chatbot_poc\\data\\documents\\it\\output\\test\\"
allquestionpairs = []
with open(folder + "fragen.csv", "r", encoding="utf-8") as file:
    currentstage = -1
    currentdifficulty = ""
    i = 0
    for line in file:
        if i == 0: 
            i+=1
            continue
        qa_pair = {}
        entries = line.strip().split(";")
        try:
            
            if "Stufe" in entries[0]:
                currentstage = entries[0].split()[1]
            keywords = ["leicht", "mittel", "schwer"]
            current_keyword = next((kw for kw in keywords if kw in entries[1]), None)
            if current_keyword:
                currentdifficulty = current_keyword
            if entries[2] != "" and entries[3] != "" and entries[4] != "" :
                qa_pair['stage'] = currentstage
                qa_pair['difficulty'] = currentdifficulty
                qa_pair['question'] = entries[2]
                qa_pair['reference'] = entries[3]
                qa_pair['keywords'] = entries[4]
                allquestionpairs.append(qa_pair)
            i+=1
        except:
            i+=1
            continue
with open(folder + "testset_wi.json", "w", encoding="utf-8") as f:
    json.dump(allquestionpairs, f, ensure_ascii=False, indent=4)
      
    
    