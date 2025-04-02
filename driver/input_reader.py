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
    tracemalloc.start() 
    mem_before = memory_usage()[0]
    start_time = time.time()
    
    create_pipeline(
        chunk_size=base_config["chunk_size"],
        overlap=base_config["overlap"],
        text_prep=base_config["text_prep"],
        embedding_model=base_config["embedding_model"],
        database=base_config["database"],
        local_llm=base_config["local_llm"]
    )
    
    total_time = time.time() - start_time
    total_mem = memory_usage()[0] - mem_before
    peak_mem = tracemalloc.get_traced_memory()[1] / 1024 / 1024  # Convert to MB
    tracemalloc.stop()
    
    result_data = [total_time, total_mem, peak_mem]
    log_results("custom_pipeline.csv", ["Time (s)", "Memory (MB)", "Peak Memory (MB)"], result_data)


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