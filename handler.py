import runpod

def handler(event):
    print(">>> Получена заявка:", event)
    return {"output": "Готово, брат!"}

print(">>> Стартиране на runpod.handler")
runpod.serverless.start({"handler": handler})
