from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
import re
import os
import gradio as gr

base_url = os.environ.get("BASE_URL")
api_key = os.environ.get("API_KEY")
model = os.environ.get("MODEL_NAME")
embed_model = os.environ.get("EMBED_MODEL_NAME")

llm = ChatOpenAI(openai_api_key = api_key,
                openai_api_base = base_url,
                model_name = model,
                temperature=0.6)
embeddings = OpenAIEmbeddings(model= embed_model, base_url=base_url, api_key = api_key, tiktoken_enabled=False) 

def qa_chain(question):

    # Build prompt
    template = """You are a highly knowledgeable assistant with access to a vast external knowledge base. 
    Your role is to provide accurate, detailed, and well-referenced answers to a variety of queries.
    You are not allowed to make up answers if it's not stated included in your knowledge, just say that you are not sure for the answer.
    You can ask for more information if the question can not be answered base on current provided knowledge. 
    {context}
    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    # Load vector store
    vector_store = FAISS.load_local(
    "vector_store",  # created by chunking_and_embedding.ipynb
    embeddings,
    allow_dangerous_deserialization=True  # Required for security reasons
    )
    
    # Run chain
    create_qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 10}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    result = create_qa_chain.invoke({"query": question})
    
    return result["result"]

def remove_thinking(llm_output):
    return re.sub(r'<think>.*?</think>', '', llm_output, flags=re.DOTALL).strip()

def gradio_qa(question):
    answer = qa_chain(question)
    answer = remove_thinking(answer)
    return answer

iface = gr.Interface(
    fn=gradio_qa,
    inputs=gr.Textbox(lines=2, placeholder="Enter your question here..."),
    outputs=gr.Textbox(label="Answer"),
    title="LLM QA Assistant"
)

if __name__ == "__main__":
    iface.launch()