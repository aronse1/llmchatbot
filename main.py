import time
import warnings
import dotenv
from llama_index.core.evaluation import FaithfulnessEvaluator
#from src.ChatBot import ChatBot, Course
from src.logger import (chatbot_logger, message_logger,
                        unanswered_questions_logger)
from src.Pipeline import *
from src.evaluator import evaluate
import json
import asyncio
warnings.filterwarnings(
    "ignore", message=".*Torch was not compiled with flash attention.*")
from colorama import Fore, Back, Style

dotenv.load_dotenv()


def loadTestset(file):
    data = []
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


import os
async def makeEvaluation(iterations : int, course, chat_bot, name: str):
    if course != Course.WI:
        print(Fore.RED + "Wrong Course selected for eval" + Fore.RESET)
        return
    
    testset = loadTestset("./data/documents/it/output/test/testset_wi.json")
    
    for a in range(iterations):
        allevaluations = []
        i = 1
        for item in testset: 
            query = item['question']
            print(Fore.BLUE + f"\nIteration {a} of {iterations} Evaluating question {i} of {len(testset)}..." + Fore.RESET)
            try:
                response = await chat_bot.run(query=query)
                print(Fore.MAGENTA + response + Fore.RESET)
            except:
                response = "Das konnte ich nicht beantworten"
            allevaluations.append(evaluate(item, response))
            i+=1
        for item in allevaluations:
            print(item)
        output_path = f"./data/documents/it/output/output_json/{name}_evaluation_results{a+16}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(allevaluations, f, ensure_ascii=False, indent=4)


async def main():
    initialise()
    course = Course.WI
    #c = AdvancedRAGWorkflow(timeout=3600, verbose=True, course=course)
    #await makeEvaluation(20, course=course, chat_bot=c, name="thinker_other_index")
    d = AdvancedRAGWorkflow3(timeout=3600, verbose=True, course=course)
    await makeEvaluation(15, course=course, chat_bot=d, name="no_context_query_qdrant")
    #e = AdvancedRAGWorkflow2(timeout=3600, verbose=True, course=course)
    #await makeEvaluation(10, course=course, chat_bot=e, name="no_context_react_other_index")

    #os.system("sudo shutdown now")
    while True:
        user_input = input("Frage: ")
        if user_input.lower() in ["exit", "quit", "q"]:
            print("Beende den Chat...")
            break
    
        result = await d.run(query=user_input)
        print("Antwort:", result)

if __name__ == "__main__":
    asyncio.run(main())