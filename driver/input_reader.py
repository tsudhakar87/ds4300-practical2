from embedding_db_setup.embedder import Embedder
from embedding_db_setup.redis_instantiator import RedisInstantiator
from embedding_db_setup.chroma_instantiator import ChromaInstantiator
from embedding_db_setup.milvus_instantiator import MilvusInstantiator
from text_preprocessing.preprocessor import Preprocessor
from timer.timer import timer
from memory_profiler import profile


def read_input():
    """Reads inputs to make the pipeline with default values if none are provided."""
    use_defaults = input("Use defaults? (y/n, default y): ") or 'y'

    if use_defaults.lower() == 'y':
        chunk_size = 300
        overlap = 50
        text_prep = 'all'
        embedding_model = 'sentence-transformers/all-MiniLM-L6-v2'
        database = 'redis'
        local_llm = 'mistral'
        print("Using default settings.")
    else:
        chunk_size = int(input("Enter chunk size (default 300): ") or 300)
        overlap = int(input("Enter overlap size (default 50): ") or 50)
        text_prep = input("Enter text prep strategy ('whitespace removal', 'punctuation removal', 'all', default all): ") or 'all'
        embedding_model = input("Enter embedding model (default sentence-transformers/all-MiniLM-L6-v2): ") or 'sentence-transformers/all-MiniLM-L6-v2'
        database = input("Enter database (default Redis): ") or 'Redis'
        local_llm = input("Enter local LLM (default mistral): ") or 'mistral'

    print(f"Chunk size: {chunk_size}, Overlap: {overlap}, Text prep: {text_prep}, Model: {embedding_model}, DB: {database}, LLM: {local_llm}")

    return chunk_size, overlap, text_prep, embedding_model, database, local_llm

def process_and_store(preprocessor, redis_instance: Embedder):
    print("Processing PDFs and storing embeddings...")

    all_chunks = preprocessor.process_pdfs()

    for file_name, page_num, chunk_index, chunk in all_chunks:
        print(f"Storing Chunk {chunk_index+1} from {file_name}, Page {page_num}")
        embedding = redis_instance.get_embedding(chunk)
        redis_instance.store_embedding(file_name, page_num, chunk, embedding)

@profile
@timer
def create_pipeline():
    chunk_size, overlap, text_prep, embedding_model, database, local_llm = read_input()

    # Initialize Preprocessor
    preprocessor = Preprocessor(data_dir="./class_materials/slides/", chunk_size=chunk_size, overlap=overlap, text_prep=text_prep)
    print(f"Preprocessor initialized with text prep strategy: {text_prep}")

    if database.lower() == 'redis':
        print("Using Redis database.")
        redis_instance = RedisInstantiator()
        redis_instance.change_embedding_model(embedding_model)
        redis_instance.create_hnsw_index()

        print("Database and model initialized.")
        
        # Process PDFs and store embeddings
        process_and_store(preprocessor, redis_instance)
        
        generate_responses(redis_instance, local_llm)

    elif database.lower() == 'chroma':
        print("Using Chroma database.")
        chroma_instance = ChromaInstantiator()
        chroma_instance.change_embedding_model(embedding_model)

        print("Database and model initialized.")

        # Process PDFs and store embeddings
        process_and_store(preprocessor, chroma_instance)

        generate_responses(chroma_instance, local_llm)

    elif database.lower() == 'milvus':
        print("Using Chroma database.")
        chroma_instance = MilvusInstantiator()
        chroma_instance.change_embedding_model(embedding_model)

        print("Database and model initialized.")

        # Process PDFs and store embeddings
        process_and_store(preprocessor, chroma_instance)

        generate_responses(chroma_instance, local_llm)
   
    else:
        print(f"Database {database} not supported yet.")

@timer
def generate_responses(instantiator: Embedder, llm_model: str):
    print("Generating responses using LLM...")
    instantiator.llm_model = llm_model


    while True:
        question = input("\nEnter your question (or type 'exit' to quit): ").strip()
        if question.lower() == 'exit':
            print("Exiting query mode.")
            break
        response = instantiator.chat_with_model(question)


@timer
def main():
    input_str = int(input("What would you like to do? \n 1. Run a pipeline \n 2. Query the model \n"))

    if input_str == 1:
        create_pipeline()
    elif input_str == 2:
        pass
    else:
        print("Invalid option or Query model logic not implemented yet.")


if __name__ == "__main__":
    main()
