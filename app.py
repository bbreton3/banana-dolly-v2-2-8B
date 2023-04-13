from potassium import Potassium, Request, Response
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = Potassium("my_app")


# @app.init runs at startup, and initializes the app's context
@app.init
def init():
    device = 0 if torch.cuda.is_available() else -1
    MODEL_NAME = "databricks/dolly-v2-2-8b"
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )

    context = {"model": model, "tokenizer": tokenizer}

    return context


# @app.handler is an http post handler running for every call
@app.handler()
def handler(context: dict, request: Request) -> Response:
    prompt = request.json.get("prompt")
    model = context.get("model")
    tokenizer = context.get("tokenizer")

    # Run the model
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    generated_ids = model.generate(input_ids)
    result = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    return Response(json={"outputs": result}, status=200)


if __name__ == "__main__":
    app.serve()
