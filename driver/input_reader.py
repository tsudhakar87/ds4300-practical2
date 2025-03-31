import csv
import os
import time
import tracemalloc
from memory_profiler import memory_usage
from driver.old_input_reader import create_pipeline, main

def log_results(filename, headers, data):
    filepath = os.path.join("results", filename)
    file_exists = os.path.isfile(filepath)
    
    with open(filepath, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(headers)
        writer.writerow(data)

def run_experiments():
    base_config = {
        "chunk_size": 300,
        "overlap": 50,
        "text_prep": "all",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "database": "redis",
        "local_llm": "mistral"
    }
    
    variations = {
        "chunk_size": [200, 300, 400],
        "overlap": [30, 50, 70],
        "text_prep": ["none", "whitespace removal", "punctuation removal", "all"],
        "embedding_model": ["sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-mpnet-base-v2", "InstructorXL"],
        "database": ["redis", "chroma", "milvus"],
        "local_llm": ["mistral", "llama"]
    }
    
    for param, values in variations.items():
        for value in values:
            config = {**base_config, param: value}
            print(f"Running experiment: {param} = {value}")
            tracemalloc.start()
            mem_before = memory_usage()[0]
            start_time = time.time()
            
            create_pipeline(
                chunk_size=config["chunk_size"],
                overlap=config["overlap"],
                text_prep=config["text_prep"],
                embedding_model=config["embedding_model"],
                database=config["database"],
                local_llm=config["local_llm"]
            )
            
            total_time = time.time() - start_time
            total_mem = memory_usage()[0] - mem_before
            peak_mem = tracemalloc.get_traced_memory()[1] / 1024 / 1024  # Convert to MB
            tracemalloc.stop()
            
            result_data = [param, value, total_time, total_mem, peak_mem]
            log_results("all_pipelines.csv", ["Parameter", "Value", "Time (s)", "Memory (MB)", "Peak Memory (MB)"], result_data)
            log_results(f"{param}.csv", ["Value", "Time (s)", "Memory (MB)", "Peak Memory (MB)"], result_data[1:])

if __name__ == "__main__":
    choice = input("What would you like to do?\n1. Run a pipeline\n2. Query the model\n3. Run experiments\n")
    if choice == "1":
        create_pipeline()
    elif choice == "2":
        main()
    elif choice == "3":
        run_experiments()
    else:
        print("Invalid choice.")
