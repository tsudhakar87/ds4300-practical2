import time 

def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        print(f"Calling {func.__name__}...", flush=True)
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} executed in {end_time - start_time} seconds.", flush=True)
        return result
    return wrapper
