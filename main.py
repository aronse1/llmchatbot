import time
import warnings
import dotenv
from llama_index.core.evaluation import FaithfulnessEvaluator
from src.ChatBot import ChatBot, Course
from src.logger import (chatbot_logger, message_logger,
                        unanswered_questions_logger)
from src.Pipeline import *
from evaluator import evaluate
import json
import asyncio
warnings.filterwarnings(
    "ignore", message=".*Torch was not compiled with flash attention.*")

dotenv.load_dotenv()
# def loadTestset(file):
#     data = []
#     with open(file, "r", encoding="utf-8") as f:
#         data = json.load(f)
#     return data



# def makeEvaluation(iterations : int, course, chat_bot):
#     testset = loadTestset("C:\\Users\\arons\\Desktop\\Studienarbeit\\chatbot_poc\\data\\documents\\it\\output\\test\\testset_wi.json")
    
#     for a in range(iterations):
#         allevaluations = []
#         i = 1
#         for item in testset: 
#             query = item['question']
#             response = chat_bot.perform_query(query, course)
            
#             print(f"\nIteration {a} of {iterations} Evaluating question {i} of {len(testset)}...")
#             allevaluations.append(evaluate(item, response))
#             i+=1

#         for item in allevaluations:
#             print(item)
#         output_path = f"C:\\Users\\arons\\Desktop\\Studienarbeit\\chatbot_poc\\data\\documents\\it\\output\\output_json\\wi_evaluation_results{a}.json"
#         with open(output_path, "w", encoding="utf-8") as f:
#             json.dump(allevaluations, f, ensure_ascii=False, indent=4)






# if __name__ == "__main__":
    
    
#     # Loggers
#     chatbot_logger = chatbot_logger(logLevel=10)
#     message_logger = message_logger(logLevel=10)
#     unanswered_questions_logger = unanswered_questions_logger(logLevel=10)

#     # Start time
#     #start_time = time.time()

#     # Set course
#     course = Course.IT

#     # setup Bot
#     chat_bot = ChatBot()

#     start_time = time.time()
#     #makeEvaluation(1, course, chat_bot)
#     #end_time = time.time()
#     #elapsed_time = end_time - start_time
#     #print(f"Time elapsed for 10 Questions: {elapsed_time}")
#     # # Perform RAG query
#     # query = "In welcher Straße befindet sich die DHBW?"
#     # result = chat_bot.perform_query(query, course)
#     # print(result)
    
#     # Calculate and print the elapsed time
#     end_time = time.time()
#     elapsed_time = end_time - start_time
#     chatbot_logger.debug(f"Elapsed time: {elapsed_time:.2f} seconds")

#     # # Loop for chat
#     while True:
#         response_text =""
#         query = input("\nFrage: ")

#         result = chat_bot.perform_query(query, course)
#         print(result.response)
#         #for  result in chat_bot.perform_query(query, course,response):
#         #    response += result
#         #    print(result, end="")
        
#         #print("\nEvaluating....")
#         #evaluate(response['response'])
#         time.sleep(1)







def loadTestset(file):
    data = []
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data



async def makeEvaluation(iterations : int, course, chat_bot):
    testset = loadTestset("./data/documents/it/output/test/testset_wi.json")
    
    for a in range(iterations):
        allevaluations = []
        i = 1
        for item in testset: 
            query = item['question']
            print(f"\nIteration {a} of {iterations} Evaluating question {i} of {len(testset)}...")
            try:
                response = await chat_bot.run(query=query)
            except:
                response = "Das konnte ich nicht beantworten"
            allevaluations.append(evaluate(item, response))
            i+=1
        for item in allevaluations:
            print(item)
        output_path = f"./data/documents/it/output/output_json/wi_new_valuation_results{a+1}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(allevaluations, f, ensure_ascii=False, indent=4)


async def main():
    initialise()
    c = AdvancedRAGWorkflow(timeout=3600, verbose=True, course=Course.WI)
    await makeEvaluation(10, course=Course.WI, chat_bot=c)
    while True:
        user_input = input("Frage: ")
        if user_input.lower() in ["exit", "quit", "q"]:
            print("Beende den Chat...")
            break
    
        result = await c.run(query=user_input)
        print("Antwort:", result)

if __name__ == "__main__":
    asyncio.run(main())