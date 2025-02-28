from llama_index.core.evaluation import FaithfulnessEvaluator, SemanticSimilarityEvaluator
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
import torch
import os
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
import re


def evaluate(testsetitem : dict, response ):
    responseText = response.response
    referenceText = testsetitem['reference']
    keywords = testsetitem["keywords"].split()
    thisEvaluation = {}
    thisEvaluation["query"] =  testsetitem['question']
    thisEvaluation["response"] = response.response
    thisEvaluation["reference"] = testsetitem['reference']
    thisEvaluation["stage"] = testsetitem["stage"]
    thisEvaluation["difficulty"] = testsetitem["difficulty"]
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    # llm = Ollama(
    #          model="llama3.1", device=device, temperature=0.0, request_timeout=360.0)
    # evaluator = FaithfulnessEvaluator(llm=llm)
    # eval_result = evaluator.evaluate_response(response=response)
    # thisEvaluation["trust_score"] = eval_result.score
    # thisEvaluation["passed"] = eval_result.passing

    bleu_score = sentence_bleu(
        [referenceText.split()],  
        responseText.split(),
        smoothing_function=SmoothingFunction().method1      
    )
    thisEvaluation["bleu_score"] = bleu_score * 100
    meteor = meteor_score([referenceText.split()], responseText.split())
    thisEvaluation["meteor_score"] = meteor
    rouge = Rouge()
    rouge_scores = rouge.get_scores(responseText, referenceText)
    thisEvaluation["rouge_scores"] = rouge_scores
    words = set(re.findall(r'\b\w+\b', responseText.lower()))

    found_keywords = [word.lower() for word in keywords if word.lower() in words]

    percentage = (len(found_keywords) / len(keywords)) * 100
    thisEvaluation['keyword-percentage'] = percentage
    return thisEvaluation
   

