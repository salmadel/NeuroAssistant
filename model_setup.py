from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

pdf_path = "./Diseases_of_The_Brain.pdf"

loader = PyPDFLoader(pdf_path)
pages = loader.load()

text_splitter = CharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50
)
docs = text_splitter.split_documents(pages)

model_id = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
qa_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embedding_model)

retriever = vectorstore.as_retriever()

all = ["retriever", "qa_pipeline"]