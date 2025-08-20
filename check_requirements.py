import os
import re
import sys
import importlib
import sysconfig

def find_imports(root_dir):
    """
    Recursively scan .py files for import statements and collect module names.
    """
    import_pattern = re.compile(r'^\s*(?:from|import)\s+([a-zA-Z0-9_\.]+)')
    modules = set()

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".py"):
                filepath = os.path.join(dirpath, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        for line in f:
                            match = import_pattern.match(line)
                            if match:
                                module = match.group(1).split('.')[0]  # only top-level
                                modules.add(module)
                except Exception as e:
                    print(f"⚠️ Could not read {filepath}: {e}")
    return modules


def get_stdlib_modules():
    """
    Return a set of standard library module names.
    Works for Python 3.10+ via sys.stdlib_module_names,
    otherwise falls back to sysconfig.
    """
    if hasattr(sys, "stdlib_module_names"):
        return sys.stdlib_module_names
    else:
        # Fallback: approximate by listing stdlib dir
        stdlib_dir = sysconfig.get_paths()["stdlib"]
        return {name.split(".")[0] for name in os.listdir(stdlib_dir)
                if os.path.isdir(os.path.join(stdlib_dir, name)) or name.endswith(".py")}


def check_modules(modules):
    """
    Try importing modules, separate stdlib and external dependencies.
    """
    stdlib = get_stdlib_modules()
    external = modules - stdlib

    print("\n🔎 Checking external dependencies:\n")
    missing = []
    for module in sorted(external):
        try:
            importlib.import_module(module)
            print(f"[OK] {module}")
        except ImportError:
            print(f"[MISSING] {module}")
            missing.append(module)

    print("\n📦 Missing packages:", missing if missing else "None")


if __name__ == "__main__":
    project_root = "."  # change to your repo root if needed
    found_modules = find_imports(project_root)
    print("Discovered imports:", sorted(found_modules))
    check_modules(found_modules)
