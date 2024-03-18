import json

# JSON 파일을 읽고 Python 객체로 변환하는 함수
def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)

# Python 객체를 JSON 파일에 저장하는 함수
def save_json(filename, data):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

# JSON 파일에서 특정 항목을 변경하는 함수
def update_json_item(filename, key, new_value):
    data = load_json(filename)
    data[key] = new_value
    save_json(filename, data)

# JSON 파일에서 특정 키의 값을 찾는 함수
def get_json_item(filename, key):
    data = load_json(filename)
    return data.get(key, None)

