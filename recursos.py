import os
import platform 
import psutil 
from multiprocessing import cpu_count

print("Sistema:", platform.system(), platform.release())
print("Procesador:", platform.processor())
print("CPUs lógicos detectados:", cpu_count())
print("CPUs lógicas (os.cpu_count()):", os.cpu_count())
print("CPUs físicas:", psutil.cpu_count(logical=False))
print("CPUs lógicas:", psutil.cpu_count(logical=True))
print("RAM total (GB):", round(psutil.virtual_memory().total / (1024**3), 2))
