import pandas as pd
import random
from openai import OpenAI
import torch
import argparse
import os
from tqdm import tqdm

random.seed(10)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device = {DEVICE}")

gpt_model_names = {
    "gpt3": "gpt-3.5-turbo-0125",
    "gpt4": "gpt-4"
}


def get_chat_completion(client, model, prompt, stream, temperature=1):
    response_json = client.chat.completions.create(
            model=model,
            messages=[
                {
                "role": "user",
                "content": prompt
                }
            ],
            
            n=1,
            stream=stream,
            max_tokens=30,
            temperature=temperature
        )
    return response_json.choices[0].message.content.split("\n\n")[0]


def get_responses(data, model, generation_type='chat', temperature_list=[0.0, 0.3, 1.0]):
    if "gpt" in model:
        client = OpenAI(
            api_key = "<---insert OpenAI API key--->"
        )
        model = gpt_model_names[model]
        model_id = model
    else:
        # host model using vllm https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html; 
        # vllm applies chat templates automatically for below models
        url_dict = {
        "llama2-7b-chat": "http://localhost:8003/v1",
        "llama2-13b-chat": "http://localhost:8002/v1",
        "llama2-70b-chat": "http://localhost:8001/v1",
        "mistral-chat": "http://localhost:8000/v1",
        "zephyr-chat": "http://localhost:8004/v1",
        }
        client = OpenAI(
            api_key = "EMPTY",
            base_url = url_dict[model]
        )
        models = client.models.list()
        model_id = models.data[0].id
    
    stream = False
    results_df = pd.DataFrame()
    
    task_instr = "Generate a short, concise single sentence response. \n" 
    # generate response for context + utterance turn 1
    for i in tqdm(range(len(data))):
        ground_truth_prompt = data.loc[i, "context_without_dialog_prefix"] + "\n" + data.loc[i, 'dialog_prefix'] + data.loc[i, 'dialog'] 
        
        for temp in temperature_list:
            if generation_type == "chat":
                
                ground_truth_prompt = task_instr + ground_truth_prompt + "\nGenerate a co-operative response without any non-literal language as {} \n".format(data.loc[i,'person2']) + "\n" + data.loc[i,'person2'] + " replies, " 
                turn2_response = get_chat_completion(client, model_id, ground_truth_prompt, stream, temperature=temp)
                
            results_df.loc[i, "turn2_response_temp_{}".format(str(temp))] = turn2_response
            
    results_df["model"] = model_id
    return results_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, default="data/hu_gpt4augmented_turn2_data.csv")
    parser.add_argument("--output_path", type=str,  default="outputs/")
    parser.add_argument("--model", type=str, default="mistral-chat")
    parser.add_argument("--temperature_list", nargs='+',  default="0.3,0.5",
                        type=lambda s: [float(item) for item in s.split(',')])

    args = parser.parse_args()
    print(args)
    args.generation_type = "chat"
    input_data = pd.read_csv(args.input_data, encoding="ISO-8859-1")
    # write to file
    if not os.path.exists(os.path.join(args.output_path, args.generation_type, args.model)):
        os.makedirs(os.path.join(args.output_path, args.generation_type, args.model)) 

    input_data.fillna('', inplace=True)
    
    results_df = get_responses(input_data, args.model, generation_type=args.generation_type, temperature_list=args.temperature_list)
    results_df.fillna('', inplace=True)
    if results_df is not None:
        output_df = pd.concat([input_data, results_df], axis=1)
        output_df.to_csv(os.path.join(args.output_path, args.generation_type, args.model, "results_turn2.csv"))
        print("done!")


if __name__ == "__main__":
    main()