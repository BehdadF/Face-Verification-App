import psutil

class SystemUtils:
    @staticmethod
    def get_num_cpus():
        """Get the number of available CPU cores for Ray."""
        return psutil.cpu_count(logical=True)

    @staticmethod
    def get_memory_info():
        """Get total and available system memory in GB."""
        mem = psutil.virtual_memory()
        return {"total": round(mem.total / 1e9, 2), "available": round(mem.available / 1e9, 2)}

    @staticmethod
    def print_system_info():
        """Print system information for debugging."""
        print(f"Available CPUs: {SystemUtils.get_num_cpus()}")
        mem_info = SystemUtils.get_memory_info()
        print(f"Total Memory: {mem_info['total']} GB")
        print(f"Available Memory: {mem_info['available']} GB")
