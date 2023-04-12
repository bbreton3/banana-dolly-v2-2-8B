from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    global tokenizer

    MODEL_NAME = "databricks/dolly-v2-12b"

    device = 0 if torch.cuda.is_available() else -1
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )


# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs: dict) -> dict:
    global model
    global tokenizer

    # Parse out your arguments
    prompt = model_inputs.get("prompt", None)
    if prompt == None:
        return {"message": "No prompt provided"}

    # Run the model
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    generated_ids = model.generate(input_ids)
    result = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    # Return the results as a dictionary
    return result
