import csv
import os
import time
import tracemalloc
from memory_profiler import memory_usage
from driver.old_input_reader import create_pipeline, main

# Function to log results into CSV file
def log_results(filename, headers, data, write_header=False):
    filepath = os.path.join("results", filename)
    file_exists = os.path.isfile(filepath)

    
    
    with open(filepath, mode='a', newline='') as file:
        writer = csv.writer(file)
        if write_header and not file_exists:
            writer.writerow(headers)
        writer.writerow(data)
 
# Function to run experiments (existing logic for pipeline, memory, etc.)
def run_experiments():
    base_config = {
        "chunk_size": 400,
        "overlap": 50,
        "text_prep": "all",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "database": "redis",
        "local_llm": "llama"
    }
    
    available_params = list(base_config.keys())
    print("Available parameters to modify:")
    for i, param in enumerate(available_params, 1):
        print(f"{i}. {param} (default: {base_config[param]})")
    
    selected_params = input("Enter the numbers of the parameters you want to modify (comma-separated): ")
    selected_params = [available_params[int(i) - 1] for i in selected_params.split(',') if i.strip().isdigit() and 0 < int(i) <= len(available_params)]
    
    for param in selected_params:
        new_value = input(f"Enter new value for {param} (current: {base_config[param]}): ")
        base_config[param] = type(base_config[param])(new_value) if isinstance(base_config[param], int) else new_value
    
    print(f"Running experiment with config: {base_config}")
    
    headers = ["Chunk Size", "Overlap", "Text Prep", "Embedding Model", "Database", "Local LLM", "Time (s)", "Memory (MB)", "Peak Memory (MB)"]
    
    # Write headers once before the loop
    log_results("custom_pipeline.csv", headers, [], write_header=True)

    for _ in range(1):  # Run each experiment once (can change to 10 as needed)
        tracemalloc.start() 
        mem_before = memory_usage()[0]
        start_time = time.time()
        
        create_pipeline(
            chunk_size=base_config["chunk_size"],
            overlap=base_config["overlap"],  # Make sure to check for correct key usage
            text_prep=base_config["text_prep"],
            embedding_model=base_config["embedding_model"],
            database=base_config["database"],
            local_llm=base_config["local_llm"]
        )
        
        total_time = time.time() - start_time
        total_mem = memory_usage()[0] - mem_before
        peak_mem = tracemalloc.get_traced_memory()[1] / 1024 / 1024  # Convert to MB
        tracemalloc.stop()
        
        print(f"Run completed: Time = {total_time:.2f}s,  Memory = {peak_mem:.2f}MB")
        
        result_data = [base_config["chunk_size"], base_config["overlap"], base_config["text_prep"], base_config["embedding_model"], base_config["database"], base_config["local_llm"], total_time, total_mem, peak_mem]
        log_results("custom_pipeline.csv", headers, result_data)

# New function to handle question and answer logging
def log_qa(question, answer):

    qa_headers = ["Question", "Answer"]
    result_data = [question, answer]
    log_results("qa_log.csv", qa_headers, result_data, write_header=True)

# New loop for interacting with the model
def query_model():
    while True:
        question = input("Enter your question (or type 'exit' to quit): ")
        if question.lower() == 'exit':
            break
        # Here you would call your model to get the response
        answer = "This is where the model's answer would go."  # Placeholder
        print(f"Model: {answer}")
        log_qa(question, answer)  # Log question and answer to CSV

if __name__ == "__main__":
    choice = input("What would you like to do?\n1. Run a pipeline\n2. Query the model\n3. Run experiments\n")
    if choice == "1":
        create_pipeline()
    elif choice == "2":
        query_model()  # Call the function for querying the model and logging questions/answers
    elif choice == "3":
        run_experiments()
    else:
        print("Invalid choice.")
