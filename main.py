import os
import asyncio
from dotenv import load_dotenv

from browser_use import Agent, Browser
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
RESUME_PATH = os.getenv("RESUME_PATH", "Abhinav Raj.pdf")
SIMILARITY_THRESHOLD = 0.8  # 80% similarity threshold

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

async def process_job_page(browser, url, title, vectordb):
    try:
        await browser.go_to(url, new_tab=True)
        
        if await browser.is_error_page():
            with open("error_internal.txt", "a") as f:
                f.write(f"{title} | {url}\n")
            await browser.close_tab()
            return False
        
        job_text = await browser.get_page_text()
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        job_docs = splitter.split_documents([Document(page_content=job_text)])
        max_score = 0
        
        for doc in job_docs:
            similar = vectordb.similarity_search_with_score(doc.page_content, k=1)
            score = 1 - abs(similar[0][1])
            if score > max_score:
                max_score = score
        
        if max_score >= SIMILARITY_THRESHOLD:
            with open("good_matches.txt", "a") as f:
                f.write(f"{title} | {url} | Score: {max_score:.2f}\n")
        else:
            with open("low_matches.txt", "a") as f:
                f.write(f"{title} | {url} | Score: {max_score:.2f}\n")
        
        await browser.close_tab()
        return True
    
    except Exception as e:
        with open("error_internal.txt", "a") as f:
            f.write(f"{title} | {url}\n")
        await browser.close_tab()
        return False

print("🔍 Parsing and vectorizing resume...")
chunks = load_and_chunk_resume(RESUME_PATH)
vectordb = vectorize_resume(chunks)

# Updated task description with improved pagination handling
task = (
    "Go to IBM Careers: https://ibmglobal.avature.net/en_US/careers\n"
    "1. Navigate to 'Explore Opportunities'\n"
    "2. Search for software roles in India for early professionals\n"
    "3. Apply filters and search\n"
    "4. While there are more pages:\n"
    "   a. Extract all job titles and URLs from current page\n"
    "   b. Look for a 'Next' page button or similar\n"
    "   c. If next page exists, click it and repeat\n"
    "5. For each job:\n"
    "   a. Open URL in new tab\n"
    "   b. Check if the posting matches the resume\n"
    "6. Handle errors gracefully\n"
    "Final output should ONLY contain job entries in format: 'TITLE | URL'"
)

browser = Browser()
agent = Agent(
    task=task,
    llm=get_llm(),
    browser=browser,
)

async def main():
    open("good_matches.txt", "w").close()
    open("low_matches.txt", "w").close()
    open("error_internal.txt", "w").close()

    result = await agent.run()
    
    # Extract valid job entries from agent's history
    job_entries = []
    for entry in result:
        if entry['step_type'] == 'action' and 'TITLE | URL' in entry['content']:
            job_entries.extend(entry['content'].split('\n'))
    
    for line in job_entries:
        if '|' not in line:
            continue
        title, url = [part.strip() for part in line.split('|', 1)]
        await process_job_page(browser, url, title, vectordb)
    
    await browser.close()

if __name__ == '__main__':
    asyncio.run(main())