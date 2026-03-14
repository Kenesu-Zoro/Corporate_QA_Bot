import os
import sys
import time
from urllib.parse import unquote

from dotenv import load_dotenv
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
OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

STORAGE_ACCOUNT_URL = os.getenv("AZURE_STORAGE_ACCOUNT_URL")
CONTAINER_NAME = os.getenv("BLOB_CONTAINER_NAME")

# Change this to the folder you want to target inside the blob container
MY_FOLDER = "Kenneth"

# Validate required environment variables
required_vars = {
    "AZURE_SEARCH_ENDPOINT": SEARCH_ENDPOINT,
    "AZURE_SEARCH_INDEX_NAME": SEARCH_INDEX_NAME,
    "AZURE_SEARCH_API_KEY": SEARCH_API_KEY,
    "AZURE_OPENAI_ENDPOINT": OPENAI_ENDPOINT,
    "AZURE_OPENAI_API_KEY": OPENAI_API_KEY,
    "AZURE_OPENAI_DEPLOYMENT_NAME": OPENAI_DEPLOYMENT,
    "AZURE_OPENAI_API_VERSION": OPENAI_API_VERSION,
    "AZURE_STORAGE_ACCOUNT_URL": STORAGE_ACCOUNT_URL,
    "BLOB_CONTAINER_NAME": CONTAINER_NAME,
}

missing = [key for key, value in required_vars.items() if not value]
if missing:
    raise ValueError(
        f"Missing required environment variables: {', '.join(missing)}"
    )

# -------------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------------
def print_status(message: str) -> None:
    print(f"   [Processing] {message}")


def print_success(message: str) -> None:
    print(f"   [✓] {message}")


def print_error(message: str) -> None:
    print(f"   [!] {message}")


# -------------------------------------------------------------------------
# AZURE CLIENTS
# -------------------------------------------------------------------------
def get_search_client() -> SearchClient:
    return SearchClient(
        endpoint=SEARCH_ENDPOINT,
        index_name=SEARCH_INDEX_NAME,
        credential=AzureKeyCredential(SEARCH_API_KEY),
    )


def get_openai_client() -> AzureOpenAI:
    return AzureOpenAI(
        azure_endpoint=OPENAI_ENDPOINT,
        api_key=OPENAI_API_KEY,
        api_version=OPENAI_API_VERSION,
    )


# -------------------------------------------------------------------------
# RETRIEVAL
# -------------------------------------------------------------------------
def search_documents(query: str, use_folder_filter: bool = True, top_k: int = 5) -> list:
    print_status(f"Connecting to Azure AI Search (Index: {SEARCH_INDEX_NAME})...")
    search_client = get_search_client()

    search_kwargs = {
        "search_text": query,
        "top": top_k,
        "select": ["content", "metadata_storage_path"],
    }

    if use_folder_filter:
        folder_prefix = f"{STORAGE_ACCOUNT_URL}/{CONTAINER_NAME}/{MY_FOLDER}/"
        filter_expression = (
            f"metadata_storage_path ge '{folder_prefix}' and "
            f"metadata_storage_path lt '{folder_prefix}~'"
        )
        search_kwargs["filter"] = filter_expression

    try:
        results = search_client.search(**search_kwargs)

        documents = []
        for result in results:
            full_path = result.get("metadata_storage_path") or ""
            filename = (
                unquote(full_path.split("/")[-1]) if full_path else "Unknown File"
            )
            content = result.get("content") or ""

            if content.strip():
                documents.append(
                    {
                        "content": content,
                        "source": filename,
                        "full_url": full_path,
                    }
                )

        print_success(f"Search complete. Found {len(documents)} relevant document chunks.")
        return documents

    except Exception as e:
        print_error(f"Search failed: {e}")
        return []


# -------------------------------------------------------------------------
# GENERATION
# -------------------------------------------------------------------------
def build_context_text(context_docs: list) -> str:
    if not context_docs:
        return "No relevant documents found."

    formatted_docs = []
    for i, doc in enumerate(context_docs, start=1):
        formatted_docs.append(
            f"[Document {i}: {doc['source']}]\n"
            f"{doc['content']}\n"
            f"[End of Document {i}]"
        )

    return "\n\n".join(formatted_docs)


def stream_chat_response(user_query: str, context_docs: list, chat_history: list) -> str:
    print_status("Generating answer with memory...")
    client = get_openai_client()

    context_text = build_context_text(context_docs)

    system_prompt = """
You are a helpful corporate assistant.

Rules:
1. Answer using ONLY the provided Context Excerpts.
2. If the answer is not in the context, say: "I don't know based on the provided documents."
3. Add citations at the end of each paragraph or claim.
4. Use citation format exactly like:
   [Source: filename]
   or
   [Source: filename, Page X]
5. If page numbers are not present in the context, use only:
   [Source: filename]
6. Be concise, accurate, and professional.
"""

    messages = [{"role": "system", "content": system_prompt.strip()}]

    # Keep only the last 4 messages for short memory
    messages.extend(chat_history[-4:])

    messages.append(
        {
            "role": "user",
            "content": f"Context Excerpts:\n{context_text}\n\nUser Question: {user_query}",
        }
    )

    try:
        response = client.chat.completions.create(
            model=OPENAI_DEPLOYMENT,
            messages=messages,
            stream=True,
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
        return full_answer.strip()

    except Exception as e:
        print_error(f"OpenAI error: {e}")
        return "I encountered an error generating the response."


# -------------------------------------------------------------------------
# OPTIONAL CONNECTION TESTS
# -------------------------------------------------------------------------
def test_search_connection() -> bool:
    print_status("Testing Azure AI Search connection...")
    try:
        search_client = get_search_client()
        results = search_client.search(
            search_text="test",
            top=1,
            select=["content", "metadata_storage_path"],
        )
        list(results)
        print_success("Azure AI Search connection is working.")
        return True
    except Exception as e:
        print_error(f"Azure AI Search test failed: {e}")
        return False


def test_openai_connection() -> bool:
    print_status("Testing Azure OpenAI connection...")
    try:
        client = get_openai_client()
        response = client.chat.completions.create(
            model=OPENAI_DEPLOYMENT,
            messages=[{"role": "user", "content": "Reply with: connection ok"}],
            stream=False,
        )
        text = response.choices[0].message.content
        print_success(f"Azure OpenAI connection is working. Response: {text}")
        return True
    except Exception as e:
        print_error(f"Azure OpenAI test failed: {e}")
        return False


# -------------------------------------------------------------------------
# MAIN LOOP
# -------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print(f"💬 CORPORATE Q&A BOT (Target Folder: {MY_FOLDER})")
    print("Commands:")
    print("  - type 'exit' or 'quit' to stop")
    print("  - type 'test' to test Azure connections")
    print("=" * 70 + "\n")

    chat_history = []

    while True:
        try:
            user_input = input("You: ").strip()

            if user_input.lower() in {"exit", "quit"}:
                print("Exiting bot. Goodbye!")
                break

            if not user_input:
                continue

            if user_input.lower() == "test":
                print("-" * 70)
                test_search_connection()
                test_openai_connection()
                print("=" * 70 + "\n")
                continue

            print("-" * 70)

            # First try with folder filter
            docs = search_documents(user_input, use_folder_filter=True, top_k=5)

            # Optional fallback: if nothing found, try without folder filter
            if not docs:
                print_status("No results found in target folder. Trying broader search...")
                docs = search_documents(user_input, use_folder_filter=False, top_k=5)

            if docs:
                answer = stream_chat_response(user_input, docs, chat_history)

                chat_history.append({"role": "user", "content": user_input})
                chat_history.append({"role": "assistant", "content": answer})
            else:
                print("\n🤖 Bot: No relevant documents found matching your keywords.\n")

            print("=" * 70 + "\n")

        except KeyboardInterrupt:
            print("\n\nForce quit detected. Goodbye!")
            break

        except Exception as e:
            print_error(f"Unexpected error: {e}")
            print("=" * 70 + "\n")