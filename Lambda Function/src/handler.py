from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import ConfigurableFieldSpec
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_pinecone import PineconeVectorStore
from langchain_community.chat_message_histories import DynamoDBChatMessageHistory
import json
import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

def lambda_handler(event, context):
    body = json.loads(event['body'])

    question = body["question"]
    user = body["user"]
    session = body["session"]
    pinecone_index = body["pinecone_index"]
    k_value = int(body["k_value"])

    os.environ["LANGCHAIN_API_KEY"] = body["langchain_api_key"]
    os.environ["OPENAI_API_KEY"] = body["openai_api_key"]
    os.environ["PINECONE_API_KEY"] = body["pinecone_api_key"]

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = PineconeVectorStore(index_name=pinecone_index, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k_value})

    llm = ChatOpenAI(model="gpt-3.5-turbo", streaming=True)

    ### Contextualize question ###
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    template = """
    You are a Car Recommender chatbot. You are helping a user find a car that fits their needs.
    The context provided to you is a mixture of car specs data as well as reviews and opinions on cars. Make sure you understand this before answering the user's question.
    You should answer the question based only on the following context provided to you. If you don't have enough information to answer the question, you should say so.

    Give the output in a nice markdown format.

    Context:
    {context}
    ####----####
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", template),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        lambda user_id, session_id: DynamoDBChatMessageHistory(
        table_name="lchain-ddb",
        session_id=session_id,
        key={"user_id": user_id, "session_id": session_id}
        ),
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
        history_factory_config=[
                        ConfigurableFieldSpec(
                            id="user_id",
                            annotation=str,
                            name="User ID",
                            description="Unique identifier for the user.",
                            default="",
                            is_shared=True,
                        ),
                        ConfigurableFieldSpec(
                            id="session_id",
                            annotation=str,
                            name="Conversation ID",
                            description="Unique identifier for the conversation.",
                            default="",
                            is_shared=True,
                        ),
                    ],
    )

    config = {"configurable": {"user_id": user, "session_id": session}}

    for answer in conversational_rag_chain.stream({"input": question}, config=config):
        # Process and stream the output here
        for key in answer:
                if key == "answer":
                        yield answer['answer']
