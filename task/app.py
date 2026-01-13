import os

from task._constants import API_KEY
from task.chat.chat_completion_client import DialChatCompletionClient
from task.embeddings.embeddings_client import DialEmbeddingsClient
from task.embeddings.text_processor import TextProcessor, SearchMode
from task.models.conversation import Conversation
from task.models.message import Message
from task.models.role import Role


SYSTEM_PROMPT = """You are a RAG-powered microwave assistant. Your purpose is to help users with questions about microwave oven usage based on the provided manual context.

IMPORTANT RULES:
1. You will receive messages with two sections: RAG Context and User Question.
2. Use ONLY the information from the RAG Context to answer the User Question.
3. If the RAG Context does not contain relevant information to answer the question, politely explain that you don't have that information in the manual.
4. Only answer questions related to microwave oven usage, safety, features, and maintenance.
5. For questions unrelated to microwaves or outside the scope of the provided context, politely decline and explain that you can only help with microwave-related questions.
6. Be concise and helpful in your responses.
"""

USER_PROMPT = """## RAG Context:
{context}

## User Question:
{question}
"""


DB_CONFIG = {
    'host': 'localhost',
    'port': 5433,
    'database': 'vectordb',
    'user': 'postgres',
    'password': 'postgres'
}

EMBEDDING_MODEL = 'text-embedding-3-small-1'
CHAT_MODEL = 'gpt-4o-mini'
MANUAL_PATH = os.path.join(os.path.dirname(__file__), 'embeddings', 'microwave_manual.txt')


def main():
    """Main function to run the RAG-powered microwave assistant"""

    print("Initializing RAG Microwave Assistant...")

    embeddings_client = DialEmbeddingsClient(EMBEDDING_MODEL, API_KEY)
    chat_client = DialChatCompletionClient(CHAT_MODEL, API_KEY)
    text_processor = TextProcessor(embeddings_client, DB_CONFIG)

    print("Processing microwave manual...")
    text_processor.process_text_file(
        file_path=MANUAL_PATH,
        chunk_size=300,
        overlap=40,
        dimensions=1536,
        truncate=True
    )

    conversation = Conversation()
    conversation.add_message(Message(Role.SYSTEM, SYSTEM_PROMPT))

    print("\n" + "=" * 60)
    print("RAG Microwave Assistant Ready!")
    print("Ask questions about your microwave oven.")
    print("Type 'exit' or 'quit' to end the conversation.")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['exit', 'quit']:
                print("Goodbye!")
                break

            # Retrieval: Search for relevant context
            context_chunks = text_processor.search(
                query=user_input,
                mode=SearchMode.COSINE_DISTANCE,
                top_k=5,
                min_score=0.5,
                dimensions=1536
            )

            # Augmentation: Combine context with user question
            if context_chunks:
                context = "\n\n".join(context_chunks)
            else:
                context = "No relevant information found in the manual."

            augmented_prompt = USER_PROMPT.format(
                context=context,
                question=user_input
            )

            # Add user message to conversation
            conversation.add_message(Message(Role.USER, augmented_prompt))

            # Generation: Get response from LLM
            response = chat_client.get_completion(conversation.get_messages())

            # Add assistant response to conversation
            conversation.add_message(response)

            print(f"\nAssistant: {response.content}\n")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()