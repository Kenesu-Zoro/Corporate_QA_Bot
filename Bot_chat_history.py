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

# Environment Variables
SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")
SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")

OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

STORAGE_ACCOUNT_URL = os.getenv("AZURE_STORAGE_ACCOUNT_URL")
CONTAINER_NAME = os.getenv("BLOB_CONTAINER_NAME")
MY_FOLDER = "Kenneth"  # <--- Target Folder


# -------------------------------------------------------------------------
# HELPER: Status Printer
# -------------------------------------------------------------------------
def print_status(message):
    """Prints a processing status message."""
    print(f"   [Processing] {message}")


def print_success(message):
    """Prints a success message."""
    print(f"   [✓] {message}")


# -------------------------------------------------------------------------
# 1. MEMORY & REWRITING (The "Brain")
# -------------------------------------------------------------------------
def check_and_rewrite_query(user_query, chat_history):
    """
    Uses GPT to rewrite follow-up questions (e.g., "Is it paid?")
    into standalone search queries (e.g., "Is sick leave paid?").
    """
    # If no history, no need to rewrite
    if not chat_history:
        return user_query

    print_status("Checking if query needs context from history...")

    client = AzureOpenAI(
        azure_endpoint=OPENAI_ENDPOINT,
        api_key=OPENAI_API_KEY,
        api_version=OPENAI_API_VERSION
    )

    # Prompt to merge history into the new question
    system_prompt = """You are a query reformulator. 
    Rewrite the user's last question to be a standalone keyword search query based on the chat history.
    - If the user says "it", "they", or "that", replace it with the specific noun from history.
    - Keep it short and keyword-focused.
    - Do NOT answer the question. Just rewrite the query."""

    # Create a minimal history string (Last 2 turns)
    history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history[-2:]])

    try:
        response = client.chat.completions.create(
            model=OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"History:\n{history_text}\n\nNew Question: {user_query}"}
            ]
        )

        rewritten_query = response.choices[0].message.content.strip()

        # Log if the query changed
        if rewritten_query.lower() != user_query.lower():
            print(f"   [Context] Rewrote query: '{user_query}' -> '{rewritten_query}'")
            return rewritten_query

        return user_query

    except Exception as e:
        print(f"   [!] Query Rewrite Failed: {e}")
        return user_query


# -------------------------------------------------------------------------
# 2. RETRIEVAL (Search)
# -------------------------------------------------------------------------
def search_documents(query):
    print_status(f"Searching Handbook for: '{query}'...")

    search_client = SearchClient(
        endpoint=SEARCH_ENDPOINT,
        index_name=SEARCH_INDEX_NAME,
        credential=AzureKeyCredential(SEARCH_API_KEY)
    )

    # Filter strictly for your folder
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
            full_path = result.get("metadata_storage_path")
            # Extract clean filename from URL
            filename = unquote(full_path.split("/")[-1]) if full_path else "Unknown File"

            documents.append({
                "content": result.get("content"),
                "source": filename
            })

        print_success(f"Found {len(documents)} relevant chunks.")
        return documents

    except Exception as e:
        print(f"   [!] Search Failed: {e}")
        return []


# -------------------------------------------------------------------------
# 3. GENERATION (Streaming + Citations)
# -------------------------------------------------------------------------
def stream_chat_response(user_query, context_docs, chat_history):
    print_status("Generating answer with memory...")

    client = AzureOpenAI(
        azure_endpoint=OPENAI_ENDPOINT,
        api_key=OPENAI_API_KEY,
        api_version=OPENAI_API_VERSION
    )

    # Prepare Context Block
    context_text = ""
    if context_docs:
        context_text = "\n\n".join([
            f"--- Document: {doc['source']} ---\n{doc['content']}\n----------------"
            for doc in context_docs
        ])
    else:
        context_text = "No relevant documents found."

    # System Prompt with STRICT Citation Rules
    system_prompt = """You are a helpful corporate assistant.

    Guidelines:
    1. Answer the user's question using ONLY the provided Context Excerpts.
    2. CITATIONS ARE MANDATORY. At the end of every distinct claim or paragraph, you MUST include a citation.
    3. Format: [Source: filename, Page X] or [Source: filename].
    4. LOOK FOR PAGE NUMBERS: If the text chunk mentions "Page X" or "Section Y", you MUST include that in the citation.
    5. If the answer is not in the context, say "I don't know".
    6. Be professional and concise."""

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
                    time.sleep(0.01)  # Typing effect

        print("\n")
        return full_answer

    except Exception as e:
        print(f"\n   [!] OpenAI Error: {e}")
        return "I encountered an error generating the response."


# -------------------------------------------------------------------------
# MAIN LOOP
# -------------------------------------------------------------------------
if __name__ == "__main__":
    # Initialize Memory
    chat_history = []

    print("\n" + "=" * 60)
    print(f"💬 CORPORATE BOT WITH MEMORY (Target: {MY_FOLDER})")
    print("   Type 'exit' to quit.")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("You: ")

            if user_input.strip().lower() in ["exit", "quit"]:
                print("Exiting. Goodbye!")
                break

            if not user_input.strip():
                continue

            print("-" * 60)

            # Step 1: Rewrite Query (The "Memory" Step)
            search_query = check_and_rewrite_query(user_input, chat_history)

            # Step 2: Search with the Rewritten Query
            docs = search_documents(search_query)

            # Step 3: Generate Answer (Pass history for tone/flow)
            bot_answer = stream_chat_response(user_input, docs, chat_history)

            # Step 4: Update Memory
            # We append the ORIGINAL question, not the rewritten one (keeps history natural)
            chat_history.append({"role": "user", "content": user_input})
            chat_history.append({"role": "assistant", "content": bot_answer})

            print("=" * 60 + "\n")

        except KeyboardInterrupt:
            print("\nForce Quit.")
            break