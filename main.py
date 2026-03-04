import os
import sys
import time
from urllib.parse import unquote
from dotenv import load_dotenv

# Azure Libraries
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from openai import AzureOpenAI

# -------------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------------
load_dotenv()

SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")
SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")

OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

STORAGE_ACCOUNT_URL = os.getenv("AZURE_STORAGE_ACCOUNT_URL")
CONTAINER_NAME = os.getenv("BLOB_CONTAINER_NAME")
MY_FOLDER = "Kenneth"


# -------------------------------------------------------------------------
# HELPER: Status Printer
# -------------------------------------------------------------------------
def print_status(message):
    """Prints a status message in a clean, distinct way."""
    print(f"   [Processing] {message}")


def print_success(message):
    """Prints a success message."""
    print(f"   [✓] {message}")


# -------------------------------------------------------------------------
# 1. RETRIEVAL
# -------------------------------------------------------------------------
def search_documents(query):
    print_status(f"Connecting to Azure AI Search (Index: {SEARCH_INDEX_NAME})...")

    search_client = SearchClient(
        endpoint=SEARCH_ENDPOINT,
        index_name=SEARCH_INDEX_NAME,
        credential=AzureKeyCredential(SEARCH_API_KEY)
    )

    folder_prefix = f"{STORAGE_ACCOUNT_URL}/{CONTAINER_NAME}/{MY_FOLDER}/"
    filter_expression = (
        f"metadata_storage_path ge '{folder_prefix}' and "
        f"metadata_storage_path lt '{folder_prefix}~'"
    )

    try:
        results = search_client.search(
            search_text=query,
            filter=filter_expression,
            top=3,
            select=["content", "metadata_storage_path"]
        )

        documents = []
        for result in results:
            # Extract clean filename from the long URL for citation
            full_path = result.get("metadata_storage_path")
            filename = unquote(full_path.split("/")[-1]) if full_path else "Unknown File"

            documents.append({
                "content": result.get("content"),
                "source": filename,
                "full_url": full_path
            })

        print_success(f"Search complete. Found {len(documents)} relevant document chunks.")
        return documents

    except Exception as e:
        print(f"   [!] Search Failed: {e}")
        return []


# -------------------------------------------------------------------------
# 2. GENERATION (With Citations & Streaming)
# -------------------------------------------------------------------------
def stream_chat_response(user_query, context_docs, chat_history):
    print_status("Generating answer with memory...")

    client = AzureOpenAI(
        azure_endpoint=OPENAI_ENDPOINT,
        api_key=OPENAI_API_KEY,
        api_version=OPENAI_API_VERSION
    )

    # 1. Prepare Context
    context_text = ""
    if context_docs:
        # We format it so the model clearly sees where the document starts/ends
        context_text = "\n\n".join([
            f"--- Document: {doc['source']} ---\n{doc['content']}\n----------------"
            for doc in context_docs
        ])
    else:
        context_text = "No relevant documents found."

    # 2. System Prompt (RESTORED CITATION LOGIC)
    system_prompt = """You are a helpful corporate assistant.

    Guidelines:
    1. Answer the user's question using ONLY the provided Context Excerpts.
    2. CITATIONS ARE MANDATORY. At the end of every distinct claim or paragraph, you MUST include a citation.
    3. Format: [Source: filename, Page X] or [Source: filename].
    4. LOOK FOR PAGE NUMBERS: If the text chunk mentions "Page X" or "Section Y", you MUST include that in the citation.
    5. If the answer is not in the context, say "I don't know".
    6. Be professional and concise."""

    # 3. Prepare Messages
    messages = [
        {"role": "system", "content": system_prompt}
    ]

    # Add History (Last 4 turns)
    messages.extend(chat_history[-4:])

    # Add Current Question + Context
    messages.append({
        "role": "user",
        "content": f"Context Excerpts:\n{context_text}\n\nUser Question: {user_query}"
    })

    # 4. Stream Response
    try:
        response = client.chat.completions.create(
            model=OPENAI_DEPLOYMENT,
            messages=messages,
            stream=True
        )

        print_success("Streaming answer...\n")
        print("🤖 Bot: ", end="", flush=True)

        full_answer = ""
        for chunk in response:
            if chunk.choices:
                delta = chunk.choices[0].delta.content
                if delta:
                    sys.stdout.write(delta)
                    sys.stdout.flush()
                    full_answer += delta
                    time.sleep(0.01)

        print("\n")
        return full_answer

    except Exception as e:
        print(f"\n   [!] OpenAI Error: {e}")
        return "I encountered an error generating the response."


# -------------------------------------------------------------------------
# MAIN LOOP
# -------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print(f"💬 CORPORATE Q&A BOT (Target Folder: {MY_FOLDER})")
    print("   commands: type 'exit' or 'quit' to stop.")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("You: ")

            if user_input.strip().lower() in ["exit", "quit"]:
                print("Exiting bot. Goodbye!")
                break

            if not user_input.strip():
                continue

            # Visual Separation for the processing block
            print("-" * 60)

            # 1. Search Phase
            docs = search_documents(user_input)

            # 2. Generation Phase
            if docs:
                stream_chat_response(user_input, docs)
            else:
                print("\n🤖 Bot: No relevant documents found matching your keywords.\n")

            print("=" * 60 + "\n")

        except KeyboardInterrupt:
            print("\n\nForce Quit detected. Goodbye!")
            break