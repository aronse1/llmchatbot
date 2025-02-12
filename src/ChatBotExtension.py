import os
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage
)
from llama_index.core.agent.react import ReActAgent
from llama_index.core.workflow import (
    step,
    Context,
    Workflow,
    Event,
    StartEvent,
    StopEvent
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama
from llama_index.core.postprocessor.rankGPT_rerank import RankGPTRerank
from llama_index.core.postprocessor.llm_rerank import LLMRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.core import Settings
from llama_index.core.tools import FunctionTool, QueryEngineTool, ToolMetadata
from helpers.PriorityNodeScoreProcessor import PriorityNodeScoreProcessor
import torch
from llama_index.core.agent import ReActAgent
from llama_index.core.query_engine import CitationQueryEngine
from enum import Enum
from llama_index.core.postprocessor import SimilarityPostprocessor
from helpers.SystemMessage import system_message
import asyncio

class JudgeEvent(Event):
    query: str

class BadQueryEvent(Event):
    query: str

class NaiveRAGEvent(Event):
    query: str

class AgentEvent(Event):
    query: str
class HighTopKEvent(Event):
    query: str

class RerankEvent(Event):
    query: str

class ResponseEvent(Event):
    query: str
    response: str

class SummarizeEvent(Event):
    query: str
    response: str

DATA_DIR = ""
PERSIST_DIR = ""
class Course(Enum):
    WI = "wi"
    IT = "it"

    def data_dir(self) -> str:
        return DATA_DIR + "/" + self.value

    def persist_dir(self) -> str:
        return PERSIST_DIR + "/" + self.value


class ComplicatedWorkflow(Workflow):
    def log_unanswered_question(self, question: str):
        """
        Call this method when the question cant be answered using the rag_tool, and then inform the user politely that you cannot answer this question.
        Logs the question that cant be answered using given information to improve the bot in future
        :param question: The question asked by user
        :return:
        """
        print(f"LOG: Following question could not be answered {question}")
        return "Answer: Ich kann diese Frage leider nicht beantworten."

    def create_agent(self, course: Course, chat_history=None, index=None):
        """
        Create chatbot agent and set up tools to be called trough ai. Each user needs his own agent to have its own context
        :param course: the desired course
        :return: agent to chat with
        """
        tools = []
        
        #if intent == "instructions":
            # Load index and set up the RAG query engine
        postprocessor = SimilarityPostprocessor(similarity_cutoff=0.7)
        query_engine = CitationQueryEngine.from_args(
            index,
            similarity_top_k=5,
            citation_chunk_size=1024,
            node_postprocessors=[PriorityNodeScoreProcessor(), postprocessor],
            streaming=True
        )
        

        rag_tool = QueryEngineTool(
            query_engine=query_engine,
            metadata=ToolMetadata(
                name="rag_tool",
                description=(
                    "This tool provides several information about the course. Use the complete user prompt question as input!"
                ),
            )
        )
        tools.append(rag_tool)
        
        # Add log tool for unanswered questions
        log_tool = FunctionTool.from_defaults(fn=self.log_unanswered_question)
        tools.append(log_tool)

        # Combine system messages with chat history
        messages = system_message + (chat_history or [])
        
        # Return the agent with the relevant tools
        return ReActAgent.from_tools(
            chat_history=messages,
            tools=tools,
            verbose=True,
            max_iterations=10
        )

    def load_or_create_index(self, directory_path, persist_dir):
        # Check if the index already exists

        if os.path.exists(persist_dir):
            print("Loading existing index...")
            # Load the index from disk
            storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
            index = load_index_from_storage(storage_context)
        else:
            print("Creating new index...")
            # Load documents from the specified directory
            documents = SimpleDirectoryReader(directory_path).load_data()

            # Create a new index from the documents
            index = VectorStoreIndex.from_documents(documents)

            # Persist the index to disk
            index.storage_context.persist(persist_dir=persist_dir)

        return index

    @step(pass_context=True)
    async def judge_query(self, ctx: Context, ev: StartEvent | JudgeEvent ) -> BadQueryEvent | NaiveRAGEvent | HighTopKEvent | RerankEvent | AgentEvent:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # initialize
        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
        Settings.llm = Ollama(
            model="llama3.1", request_timeout=360.0, device=device)
        if not hasattr(ctx.data, "llm"):
            ctx.data["llm"] = Ollama(model="llama3.1", device=device, request_timeout=360.0)#OpenAI(model="gpt-4o",temperature=0.1)
            ctx.data["index"] = self.load_or_create_index(
                "./data/documents/it",
                "./data/index/it"
            )
            # we use a chat engine so it remembers previous interactions
            ctx.data["judge"] = SimpleChatEngine.from_defaults()

        response = ctx.data["judge"].chat(f"""
                    Gegeben eine Benutzeranfrage, bestimme, ob diese voraussichtlich gute Ergebnisse von einem RAG-System liefern wird. 
                    Falls sie gut ist, gib 'gut' zurück, falls sie schlecht ist, gib 'schlecht' zurück.  
                    Gute Anfragen enthalten viele relevante Schlüsselwörter und sind detailliert.  
                    Schlechte Anfragen sind vage oder mehrdeutig.

                    Hier ist die Anfrage: {ev.query}
                    """)
        if response == "schlecht":
            # try again
            return BadQueryEvent(query=ev.query)
        else:
            # send query to all 3 strategies
            self.send_event(NaiveRAGEvent(query=ev.query))
            self.send_event(HighTopKEvent(query=ev.query))
            #self.send_event(RerankEvent(query=ev.query))
            self.send_event(AgentEvent(query=ev.query))

    @step(pass_context=True)
    async def improve_query(self, ctx: Context, ev: BadQueryEvent) -> JudgeEvent:
        response = ctx.data["llm"].complete(f"""
            Dies ist eine Anfrage an ein RAG-System: {ev.query}

            Die Anfrage ist schlecht, weil sie zu vage ist. Bitte gib eine detailliertere Anfrage an, die spezifische Schlüsselwörter enthält und jegliche Mehrdeutigkeit beseitigt.
        """)
        return JudgeEvent(query=str(response))

    @step(pass_context=True)
    async def citationAgent(self, ctx: Context, ev: AgentEvent) -> ResponseEvent:
        index = ctx.data["index"]
        agent = self.create_agent(course=Course.IT, index=index)
        response = agent.chat(ev.query)

        return ResponseEvent(query=ev.query, source="Agent", response=response.response)

    
    @step(pass_context=True)
    async def naive_rag(self, ctx: Context, ev: NaiveRAGEvent) -> ResponseEvent:
        index = ctx.data["index"]
        engine = index.as_query_engine(similarity_top_k=5)
        response = engine.query(ev.query)
        print("Naive response:", response)
        return ResponseEvent(query=ev.query, source="Naive", response=str(response))

    @step(pass_context=True)
    async def high_top_k(self, ctx: Context, ev: HighTopKEvent) -> ResponseEvent:
        index = ctx.data["index"]
        engine = index.as_query_engine(similarity_top_k=20)
        response = engine.query(ev.query)
        print("High top k response:", response)
        return ResponseEvent(query=ev.query, source="High top k", response=str(response))

    @step(pass_context=True)
    async def rerank(self, ctx: Context, ev: RerankEvent) -> ResponseEvent:
        index = ctx.data["index"]
        #reranker = RankGPTRerank(
        #    top_n=5,
        #    llm=ctx.data["llm"]
        #)
        reranker = LLMRerank(top_n=5, llm=ctx.data["llm"])
        retriever = index.as_retriever(similarity_top_k=20)
        engine = RetrieverQueryEngine.from_args(
            retriever=retriever,
            node_postprocessors=[reranker],
        )
        response = engine.query(ev.query)
        print("Reranker response:", response)
        return ResponseEvent(query=ev.query, source="Reranker", response=str(response))

    @step(pass_context=True)
    async def judge(self, ctx: Context, ev: ResponseEvent) -> StopEvent:
        ready = ctx.collect_events(ev, [ResponseEvent]*3)
        if ready is None:
            return None

        response = ctx.data["judge"].chat(f"""
                    Ein Benutzer hat eine Anfrage gestellt, und es wurden drei verschiedene Strategien verwendet,  
                    um die Anfrage zu beantworten. Deine Aufgabe ist es zu entscheiden, welche Strategie die Anfrage am besten beantwortet hat.  
                    Die Anfrage war: {ev.query}

                    Antwort 1 ({ready[0].source}): {ready[0].response}  
                    Antwort 2 ({ready[1].source}): {ready[1].response}  
                    Antwort 3 ({ready[2].source}): {ready[2].response}  

                    Bitte gib die Nummer der besten Antwort an (1, 2 oder 3).  
                    Gib nur die Nummer aus, ohne weiteren Text oder Einleitung.
                """)

        best_response = int(str(response))
        print(f"Best response was number {best_response}, which was from {ready[best_response-1].source}")
        return StopEvent(result=str(ready[best_response-1].response))
    

async def main():
    c = ComplicatedWorkflow(timeout=360, verbose=True)
    result = await c.run(
    #query="How has spending on police changed in San Francisco's budgets from 2016 to 2018?"
    query="Wer ist der Studiengangsleiter?"
    #query="How has spending changed?"
    )
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
