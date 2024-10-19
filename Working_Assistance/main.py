# from fastapi.middleware.cors import CORSMiddleware
# import os
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from pymongo import MongoClient
# from langchain.text_splitter import CharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_groq import ChatGroq
# import os
# from dotenv import load_dotenv
# from azure.devops.connection import Connection
# from msrest.authentication import BasicAuthentication

# app = FastAPI()
# load_dotenv(dotenv_path=".env")


# # CORS middleware setup
# origins = [
#     "http://localhost:5173",  
#     "http://localhost:3000"    
# ]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# client = MongoClient(os.getenv("MONGO_URI"))
# db = client.get_database(os.getenv("DBDB_NAME"))
# students_collection = db.get_collection("users")

# # Azure DevOps PAT and organization URL
# personal_access_token = os.getenv("AZURE_DEVOPS_PAT")
# organization_url = 'https://dev.azure.com/shwetambhosale18'

# # Authenticate using the personal access token
# credentials = BasicAuthentication('', personal_access_token)
# connection = Connection(base_url=organization_url, creds=credentials)

# # Initialize the Git client
# git_client = connection.clients.get_git_client()

# # Initialize FastAPI
# app = FastAPI()

# # Define data models
# class Repository(BaseModel):
#     project_name: str
#     repo_name: str
#     github_url: str

# # Function to fetch code from a repository
# def get_code_from_repo(repo_name, project_name, file_types=['.py', '.js']):
#     repo = git_client.get_repository(project=project_name, repository_id=repo_name)
    
#     # Get items (files and directories) from the repository root
#     items = git_client.get_items(project=project_name, repository_id=repo.id, recursion_level='Full')

#     if not items:
#         print("No items found in the repository.")
    
#     code = ""
#     for item in items:
#         if item.is_folder:
#             print(f"Skipping folder: {item.path}")
#         else:
#             print(f"Found file: {item.path}")
            
#             if any(item.path.endswith(ext) for ext in file_types):  # Filter by file type
#                 print(f"Fetching content of: {item.path}")
                
#                 # Get the content of each item (handle as a generator)
#                 file_content_generator = git_client.get_blob_content(project=project_name, repository_id=repo.id, sha1=item.object_id)
                
#                 # Collect content from the generator and decode bytes
#                 file_content = ''.join([chunk.decode('utf-8') for chunk in file_content_generator])

#                 if file_content:
#                     code += file_content  # Append the content to the code string
#                 else:
#                     print(f"No content found for file: {item.path}")
    
#     return code

# # Function to split text into chunks
# def text_to_chunks(raw_text):
#     text_splitter = CharacterTextSplitter(
#         chunk_size=1000,
#         chunk_overlap=200,
#         separator="\n",
#         length_function=len
#     )
#     chunks = text_splitter.split_text(raw_text)
#     return chunks

# # Function to create a vector store from chunks
# def get_vectorstore(chunks):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
#     return vectorstore

# conversation = None
# chat_history = []
# app.secret_key = b'\xac\xaa\xd6\xa2\x83\xf7h\xf4\xc9\xd7Q/\xf4@\x12\xfb\xff*\xf8\xe6b{\xc5\xc0'

# # Function to get the conversation chain
# def get_conversation_chain(vectorstore):
#     llm = ChatGroq(
#         model="llama-3.1-70b-versatile",
#         temperature=0,
#         max_tokens=None,
#         timeout=None,
#         max_retries=2,
#         api_key=os.getenv("CHAT_GROQ_API_KEY")  # Ensure your API key is stored in an environment variable
#     )

#     memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

#     conversation_chain = ConversationalRetrievalChain.from_llm(
#         llm=llm,
#         retriever=vectorstore.as_retriever(),
#         memory=memory,
#     )
#     return conversation_chain

# # Function to handle user questions and provide custom responses
# def custom_response_handler(user_question):
#     if "code" in user_question.lower():
#         prompt = f"Retrieve code snippets related to: {user_question}. Please return the code only, without any explanations."
#     elif "explain" in user_question.lower() or "optimize" in user_question.lower():
#         prompt = f"Explain or optimize the following code: {user_question}."
#     else:
#         prompt = f"Provide a response to: {user_question}."

#     response = conversation({'question': prompt})  # Call the conversation chain with the custom prompt
#     return response

# # Endpoint to process repository information
# @app.post("/process")
# async def process_repository(repo: Repository):
#     code = get_code_from_repo(repo.repo_name, repo.project_name)
#     chunks = text_to_chunks(code)
#     vectorstore = get_vectorstore(chunks)
#     global conversation
#     conversation = get_conversation_chain(vectorstore)
#     return {"message": "Repository processed successfully."}

# # Endpoint to ask a question
# @app.post("/ask")
# async def ask_question(question: str):
#     if not conversation:
#         raise HTTPException(status_code=400, detail="No conversation chain available. Please process a repository first.")
#     response = custom_response_handler(question)
#     return {"response": response}

# @app.get("/")
# async def read_root():
#     return {"message": "Helo, Checking connection "}

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from htmlTemp import css, bot_template, user_template
import google.generativeai as genai
from azure.devops.connection import Connection
from msrest.authentication import BasicAuthentication
from langchain_groq import ChatGroq
import os
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify


from dotenv import load_dotenv
load_dotenv()


#PAT
personal_access_token = os.getenv("AZURE_DEVOPS_PAT")  
organization_url = 'https://dev.azure.com/shwetambhosale18'

# Authenticate using the personal access token
credentials = BasicAuthentication('', personal_access_token)
connection = Connection(base_url=organization_url, creds=credentials)

# Initialize the Git client
git_client = connection.clients.get_git_client()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")    

genai.configure(api_key=GOOGLE_API_KEY)



app = Flask(__name__)

def get_code_from_repo(repo_name, project_name, file_types=['.py', '.js']):
    repo = git_client.get_repository(project=project_name, repository_id=repo_name)
    
    # Get items (files and directories) from the repository root
    items = git_client.get_items(project=project_name, repository_id=repo.id, recursion_level='Full')

    if not items:
        print("No items found in the repository.")
    
    code = ""
    for item in items:
        if item.is_folder:
            print(f"Skipping folder: {item.path}")
        else:
            print(f"Found file: {item.path}")  # Debug print to show found file paths
            
            if any(item.path.endswith(ext) for ext in file_types):  # Filter by file type
                print(f"Fetching content of: {item.path}")  # Debug print to show file being processed
                
                # Get the content of each item (handle as a generator)
                file_content_generator = git_client.get_blob_content(project=project_name, repository_id=repo.id, sha1=item.object_id)
                
                # Collect content from the generator and decode bytes
                file_content = ''.join([chunk.decode('utf-8') for chunk in file_content_generator])

                if file_content:
                    code += file_content  # Append the content to the code string
                else:
                    print(f"No content found for file: {item.path}")  # Debug print if no content is found
    
    return code

def text_to_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200,
    separator = "\n",
    length_function = len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks

def get_vectorstore(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatGroq(
            model="llama-3.1-70b-versatile",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            api_key = "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"
            )

    memory = ConversationBufferMemory(memory_key = "chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever=vectorstore.as_retriever(),
        memory = memory,
    )
    return conversation_chain

conversation = None
chat_history = []
app.secret_key = b'\xac\xaa\xd6\xa2\x83\xf7h\xf4\xc9\xd7Q/\xf4@\x12\xfb\xff*\xf8\xe6b{\xc5\xc0'

@app.route('/create_project', methods=['POST'])
def create_project():
    """Endpoint to create a project (if needed for API functionality)."""
    data = request.get_json()
    repository_name = data.get('repository_name')
    project_name = data.get('project_name')
    
    return jsonify({"repository": repository_name, "project": project_name})

@app.route('/', methods=['GET'])
def index():
    """Render the main chat interface."""
    global chat_history
    return render_template('index.html', chat_history=chat_history)

@app.route('/ask', methods=['POST'])
def ask():
    """Handle user questions."""
    global conversation, chat_history
    user_question = request.form['user_question']

    if conversation is None:
        flash("Please process the repository first.", "warning")
        return redirect(url_for('index'))

    # Process user question and update chat history
    response = conversation({'question': user_question})

    # Format the chatbot response, escaping HTML where necessary
    formatted_response = format_chatbot_response(response['answer'])  # Ensure you replace 'response['answer']' with the actual key if needed

    chat_history.append(formatted_response)
    return redirect(url_for('index'))

def format_chatbot_response(answer):
    """Formats the chatbot response to include HTML for code snippets."""
    # Replace code snippets with <pre><code> blocks for proper formatting
    formatted_answer = answer.replace('```', '<pre><code>').replace('```', '</code></pre>')

    return formatted_answer

@app.route('/process', methods=['POST'])
def process_repo():
    global conversation, chat_history
    # project_name = request.form['project_name']
    # repo_name = request.form['repo_name']
    project_name = 'SPARK'
    repo_name = 'SPARK'
    if project_name and repo_name:
        raw_text = get_code_from_repo(repo_name, project_name)
        chunks = text_to_chunks(raw_text)
        vectorstore = get_vectorstore(chunks)
        conversation = get_conversation_chain(vectorstore)
        chat_history = []
        flash("Repository processed successfully!", "success")
    else:
        flash("Please provide both the project name and repository name.", "warning")

    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)