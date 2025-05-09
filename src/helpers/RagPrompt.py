from llama_cloud import InputMessage, MessageRole
from llama_index.core import ChatPromptTemplate

additional_kwargs = {}

rag_messages = [
    InputMessage(
        id=1,
        index=1,
        role=MessageRole.SYSTEM,
        content=(
            """
                Beantworte die Frage anhand der gegebenen RAG Dokumente.
                Beginne deine Antwort mit "Answer:"
            """
        ),
        additional_kwargs=additional_kwargs
    ),
    InputMessage(
        id="user",
        index=1,
        role=MessageRole.USER,
        content=(
            """
            Kontext-Informationen:
            {context_str}
            Frage:
            {query_str}
            """
        ),
        additional_kwargs=additional_kwargs
    )
]

rag_template = ChatPromptTemplate(rag_messages)
