import os 

PROJECT_DIRECTORY = os.path.realpath(os.path.curdir)

REMOVE_PATHS = []

if '{{cookiecutter.use_mypy}}' != "y":
    REMOVE_PATHS.append(".github/workflows/mypy.yaml")
    REMOVE_PATHS.append("mypy.ini")


for path in REMOVE_PATHS:
    path = path.strip()
    path = os.path.join(PROJECT_DIRECTORY, path)
    print(path)
    if path and os.path.exists(path):
        if os.path.isdir(path):
            os.rmdir(path)
        else:
            os.unlink(path)