import os
import asyncio
from dotenv import load_dotenv

from browser_use import Agent, Browser
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# Load .env variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
RESUME_PATH = os.getenv("RESUME_PATH", "Abhinav Raj.pdf")

def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.2
    )

def load_and_chunk_resume(path):
    loader = PyPDFLoader(path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)

def vectorize_resume(docs, persist_dir="chroma_db"):
    embedding = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )
    vectordb = Chroma.from_documents(docs, embedding, persist_directory=persist_dir)
    vectordb.persist()
    return vectordb


print("🔍 Parsing and vectorizing resume...")
chunks = load_and_chunk_resume(RESUME_PATH)
vectordb = vectorize_resume(chunks)

relevant_chunks = vectordb.similarity_search("software engineering jobs", k=4)
context = "\n---\n".join(doc.page_content for doc in relevant_chunks)

# Agent task description
task = (
    "Go to IBM Careers: https://ibmglobal.avature.net/en_US/careers . "
    "go to explore oppurtunities and start Find software-related roles in india with early professional and after updating search filter click search button below that match the following resume:\n\n"
    f"{context}\n\n"
    "Return a list of job titles and their links. "
    "click on the job title or arrow ro the right and if the new tab shows error return to previous tab and contintue search Make sure you open each job description page in a **new tab** (don't reuse the original tab) to verify the match. "
    "Filter out internships unless highly relevant. Prioritize backend, Python, FastAPI, and software engineering roles."
)
browser=Browser()
print("🧠 Running the AI agent on IBM Careers with Gemini Flash...")
agent = Agent(
    task=task,
    llm=get_llm(),
    browser=browser,
)
async def main():
    await agent.run()
	# await browser.close()


if __name__ == '__main__':
	asyncio.run(main())

