import json

def dataLoader(DBname):
    with open(f'./src/DB/{DBname}') as f:
        db = json.load(f)
    return db