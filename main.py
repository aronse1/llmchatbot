import time
import warnings
import dotenv
from llama_index.core.evaluation import FaithfulnessEvaluator
from src.ChatBot import ChatBot, Course
from src.logger import (chatbot_logger, message_logger,
                        unanswered_questions_logger)
#from src.ChatBotExtension import ChatBotExtension, Course
from evaluator import evaluate
import json
import asyncio
warnings.filterwarnings(
    "ignore", message=".*Torch was not compiled with flash attention.*")

dotenv.load_dotenv()
def loadTestset(file):
    data = []
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data



def makeEvaluation(iterations : int, course, chat_bot):
    testset = loadTestset("C:\\Users\\arons\\Desktop\\Studienarbeit\\chatbot_poc\\data\\documents\\it\\output\\test\\testset_wi.json")
    
    for a in range(iterations):
        allevaluations = []
        i = 1
        for item in testset: 
            query = item['question']
            response = chat_bot.perform_query(query, course)
            
            print(f"\nIteration {a} of {iterations} Evaluating question {i} of {len(testset)}...")
            allevaluations.append(evaluate(item, response))
            i+=1

        for item in allevaluations:
            print(item)
        output_path = f"C:\\Users\\arons\\Desktop\\Studienarbeit\\chatbot_poc\\data\\documents\\it\\output\\output_json\\wi_evaluation_results{a}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(allevaluations, f, ensure_ascii=False, indent=4)






if __name__ == "__main__":
    
    
    # Loggers
    chatbot_logger = chatbot_logger(logLevel=10)
    message_logger = message_logger(logLevel=10)
    unanswered_questions_logger = unanswered_questions_logger(logLevel=10)

    # Start time
    #start_time = time.time()

    # Set course
    course = Course.IT

    # setup Bot
    chat_bot = ChatBot()

    start_time = time.time()
    #makeEvaluation(1, course, chat_bot)
    #end_time = time.time()
    #elapsed_time = end_time - start_time
    #print(f"Time elapsed for 10 Questions: {elapsed_time}")
    # # Perform RAG query
    # query = "In welcher Straße befindet sich die DHBW?"
    # result = chat_bot.perform_query(query, course)
    # print(result)
    
    # Calculate and print the elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    chatbot_logger.debug(f"Elapsed time: {elapsed_time:.2f} seconds")

    # # Loop for chat
    while True:
        response_text =""
        query = input("\nFrage: ")

        result = chat_bot.perform_query(query, course)
        print(result.response)
        #for  result in chat_bot.perform_query(query, course,response):
        #    response += result
        #    print(result, end="")
        
        #print("\nEvaluating....")
        #evaluate(response['response'])
        time.sleep(1)







    