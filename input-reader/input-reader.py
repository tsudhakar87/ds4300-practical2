def read_input():
    """Reads in inputs to make the pipeline."""
    chunk_size = int(input("Enter chunk size: "))
    overlap = int(input("Enter overlap size: "))
    text_prep = input("Enter text prep strategy ('whitespace removal', 'punctuation removal', 'all'): ")
    embedding_model = input("Enter embedding model: ")
    database = input("Enter database: ")
    local_llm = input("Enter local LLM: ")

    print(f"Chunk size: {chunk_size}, Overlap: {overlap}, Text prep: {text_prep}, Model: {embedding_model}, DB: {database}, LLM: {local_llm}")

def main():
    input_str = int(input("What would you like to do? \n 1. Run a pipeline \n 2. Query the model\n"))
    
    if input_str == 1:
        read_input()
    else:
        print("Invalid option or Query model logic not implemented yet.")

if __name__ == "__main__":
    main()
    
    