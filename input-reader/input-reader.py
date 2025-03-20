 # This file is in charge of user input to run pipelines and query the model.
 # Since we the variables we want to test are:
    # chunk size
    # overlap
    # text prep strategy
    # embedding model
    # database type
    # local llm
# the user will be able to input all of these to run a specific pipeline.
# the runtime

def read_input():
    """Reads in inputs to make the pipeline."""
    chunk_size = int(input("Enter chunk size: "))
    overlap = int(input("Enter overlap size: "))
    text_prep = input("Enter text prep strategy ('whitespace removal', 'punctuation removal', 'all'): ")
    embedding_model = input("Enter embedding model: ")
    database = input("Enter database: ")
    local_llm = input("Enter local LLM: ")

    print(f"Chunk size: {chunk_size}, Overlap: {overlap}, Text prep: {text_prep}, Model: {embedding_model}, DB: {database}, LLM: {local_llm}")
    
    return chunk_size, overlap, text_prep, embedding_model, database, local_llm

def create_pipeline():
    pass

def main():
    input_str = int(input("What would you like to do? \n 1. Run a pipeline \n 2. Query the model\n"))
    
    if input_str == 1:
        read_input()
    else:
        print("Invalid option or Query model logic not implemented yet.")

if __name__ == "__main__":
    main()