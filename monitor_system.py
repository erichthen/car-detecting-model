import psutil
import time
from threading import Thread

#since the training on over 200,000 items takes a long time, I will monitor the system resources
def log_resources(interval=10):
    try:
        while True:
            cpu_usage = psutil.cpu_percent()
            mem_usage = psutil.virtual_memory().percent
            print(f'cpu usage: {cpu_usage}%, memory usage: {mem_usage}%')
            time.sleep(interval)
    except KeyboardInterrupt:
        print('Monitoring stopped by user') 

if __name__ == '__main__':
    monitor_logging_thread = Thread(target=log_resources, daemon=True)
    monitor_logging_thread.start()
    
    try:
        while True:
            time.sleep(1) #keep program running for the thread to work
    except KeyboardInterrupt:
        print('Main program stopped')