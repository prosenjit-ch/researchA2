import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from telegram import Update, InputFile
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from fastapi import FastAPI, Request
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
import os
from io import BytesIO

load_dotenv()


TOKEN = os.environ.get('TOKEN')
BOT_USERNAME = os.environ.get('BOT_USERNAME')
WEBHOOK_URL = "https://researcha2.onrender.com/webhook"

os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    embeddings_list = embeddings.embed_documents(text_chunks)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings_list)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """You are a virtual Research Assistant. Your task is to answer questions related to research papers, 
    including details such as the title, abstract, keywords, field of research, and summary. Provide a 
    thorough and accurate response based on the provided context. If the answer is not in the provided context, 
    just say, "answer is not available in the context", and do not provide a wrong answer.

    Context:\n {context}?\n
    Question:\n{question}\n
    Answer:"""

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def process_pdf_and_ask_question(pdf_files, user_question):
    raw_text = get_pdf_text(pdf_files)
    text_chunks = get_text_chunks(raw_text)
    get_vector_store(text_chunks)
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)
    
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    
    return response["output_text"]





# Initialize FastAPI and bot application as before
app = FastAPI()

# Add your refactored functions here

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hello! Thanks for chatting with me!")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Send a PDF and ask a question to analyze the document.")

async def handle_pdf(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Handle PDF files
    pdf_files = []
    for document in update.message.document:
        file = await document.get_file()
        file_content = await file.download_as_bytearray()
        pdf_files.append(BytesIO(file_content))

    # Notify the user that the processing has started
    await update.message.reply_text("Processing your PDF...")

    # Save the files and then ask a question
    user_question = "Provide the summary of this paper."
    response_text = process_pdf_and_ask_question(pdf_files, user_question)

    await update.message.reply_text(response_text)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_question = update.message.text
    # Assuming that PDF was already processed and the vector store is ready
    response_text = process_pdf_and_ask_question([], user_question)
    await update.message.reply_text(response_text)

async def error(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f"Update {update} caused error {context.error}")

# Register the bot handlers
bot_app = Application.builder().token(TOKEN).build()

bot_app.add_handler(CommandHandler('start', start_command))
bot_app.add_handler(CommandHandler('help', help_command))
bot_app.add_handler(MessageHandler(filters.Document.PDF, handle_pdf))
bot_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
bot_app.add_error_handler(error)

# FastAPI routes for webhook setup

# @app.get("/")
# async def root():
#     return {"message": "Service is live! Visit the bot at https://t.me/researchA_bot"}

@app.get("/")
async def root():
    return RedirectResponse(url="https://t.me/researchA2_bot")

@app.head("/")
async def root_head():
    return {"message": "HEAD requests are allowed."}

@app.get("/webhook")
async def get_webhook():
    return {"message": "This is the webhook endpoint. Please use POST requests to send updates."}

@app.post('/webhook')
async def webhook_handler(request: Request):
    update = await request.json()
    update = Update.de_json(update, bot_app.bot)
    await bot_app.process_update(update)
    return 'OK'

@app.on_event("startup")
async def on_startup():
    await bot_app.initialize()
    await bot_app.bot.set_webhook(WEBHOOK_URL)

@app.on_event("shutdown")
async def on_shutdown():
    await bot_app.shutdown()

