from openai import OpenAI
import pandas as pd
from tqdm import tqdm
import argparse
import os
import random

prompt_template_contextual_similarity = """
Task: You will read a short story. The story will be followed by a question. Your task is to decide which response option is closest to the 'Generated Response'. The answer options are 1 or 2 or 'neither'.

Scenario: {context}
{u_1_n}
Intention: {i_t}

Generated Response:  
{person2} replies, {u_2_n}	 
 		 
Compare the below utterances to the Generated Response. Which of the below utterances is closest to the above Generated Response? 
Options: 
1. {u_2_t}
2. {u_2_l}
Answer (option number 1 or 2): {number}
"""

prompt_template_noncontextual_similarity = """
Task:  Your task is to decide which response option is closest to the 'Generated Response'. The answer options are 1 or 2 or 'neither'.

Generated Response:  
{person2} replies, {u_2_n}	 
 		 
Compare the below utterances to the Generated Response. Which of the below utterances is closest to the above Generated Response? 
Options: 
1. {u_2_t}
2. {u_2_l}
Answer (option number 1 or 2): {number}
"""

def get_chat_completion(client, model, prompt, stream):
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
            #logprobs=1 if "gpt" not in model else None,
            max_tokens=1,
            temperature=0
        )
    return response_json.choices[0].message.content

def get_scores(data, tag=''):
    client = OpenAI(
            api_key = "<---insert OpenAI API key--->"
        )
    model = "gpt-4"
    mapping = {1: "true", 2: "literal"}
    results_df = pd.DataFrame()
    
    for i in tqdm(range(len(data))):
        for temp in [0.3, 0.5]:
            
            number = random.randint(1, 2)
            input_prompt = prompt_template_contextual_similarity.format(
                context=data.loc[i, "context_without_dialog_prefix"],
                u_1_n=data.loc[i, 'dialog_prefix'] + data.loc[i, 'dialog'],
                person2=data.loc[i, "person2"],
                i_t=data.loc[i, 'true_intention'],
                u_2_n=data.loc[i, "{}turn2_response_temp_{}".format(tag, temp)],
                u_2_t=data.loc[i, "turn2_response_from_{}_intention".format(mapping[number])],
                u_2_l=data.loc[i, "turn2_response_from_{}_intention".format(mapping[3 - number])],
                number='number'
                )
            gt_response = get_chat_completion(client, model, input_prompt, False)
            
            results_df.loc[i, "{}gpt_eval_temp_{}_output".format(tag, temp)] = int(gt_response)
            results_df.loc[i, "{}gpt_eval_temp_{}_ground_truth_true".format(tag, temp)] = number
            
    for temp in [0.3, 0.5]:
        results_df["{}gpt_eval_temp_{}_final".format(tag, temp)] = (results_df["{}gpt_eval_temp_{}_output".format(tag, temp)] == results_df["{}gpt_eval_temp_{}_ground_truth_true".format(tag, temp)]).astype(int)
    return results_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default="outputs/")
    parser.add_argument("--generation_type", type=str, default="chat")
    parser.add_argument("--model", type=str, default="mistral-chat")
    parser.add_argument("--version", type=str, default="", help="'version1', 'version2' etc")
    
    args = parser.parse_args()
    print(args)
    
    '''
    args.version = "_version_6"
    if "6" in args.version:
        input_data = pd.read_csv(os.path.join(args.output_path, args.generation_type, args.model, "results_turn2{}.csv".format(args.version)))
        input_data.fillna('', inplace=True)
        input_data = input_data.rename(columns={"exp_1b_turn2_response_from_flout_temp_1.0": "exp_6b_turn2_response_from_flout_temp_1.0",
                                        "exp_1b_turn2_response_from_flout_temp_0.3": "exp_6b_turn2_response_from_flout_temp_0.3",
                                        "exp_1b_turn2_response_from_flout_temp_0.5": "exp_6b_turn2_response_from_flout_temp_0.5",
                                        "exp_1b_turn2_response_from_flout_temp_0.0": "exp_6b_turn2_response_from_flout_temp_0.0"})
        tag = "exp_6a_"
        results_df_a = get_scores(input_data, tag)
        tag = "exp_6b_"
        results_df_b = get_scores(input_data, tag)
        results_df = pd.concat([results_df_a, results_df_b], axis=1)

        if results_df is not None:
            output_df = pd.concat([input_data, results_df], axis=1)
            output_df.to_csv(os.path.join(args.output_path, args.generation_type, args.model, "results_turn2{}.csv".format(args.version)))
    
    args.version = "_version_1"
    if "1" in args.version:
        input_data = pd.read_csv(os.path.join(args.output_path, args.generation_type, args.model, "results_turn2{}.csv".format(args.version)))
        input_data.fillna('', inplace=True)

        tag = "exp_1a_"
        results_df_a = get_scores(input_data, tag)
        tag = "exp_1b_"
        results_df_b = get_scores(input_data, tag)
        results_df = pd.concat([results_df_a, results_df_b], axis=1)

        if results_df is not None:
            output_df = pd.concat([input_data, results_df], axis=1)
            output_df.to_csv(os.path.join(args.output_path, args.generation_type, args.model, "results_turn2{}.csv".format(args.version)))
    
    args.version = "_version_2"
    if "2" in args.version:
        input_data = pd.read_csv(os.path.join(args.output_path, args.generation_type, args.model, "results_turn2{}.csv".format(args.version)))
        input_data.fillna('', inplace=True)

        tag = "exp_2_"
        results_df = get_scores(input_data, tag)
    
        if results_df is not None:
            output_df = pd.concat([input_data, results_df], axis=1)
            output_df.to_csv(os.path.join(args.output_path, args.generation_type, args.model, "results_turn2{}.csv".format(args.version)))
        
    args.version = "_version_3"
    if "3" in args.version:
        input_data = pd.read_csv(os.path.join(args.output_path, args.generation_type, args.model, "results_turn2{}.csv".format(args.version)))
        input_data.fillna('', inplace=True)

        tag = "exp_3a_"
        results_df_a = get_scores(input_data, tag)
        tag = "exp_3b_"
        results_df_b = get_scores(input_data, tag)
        results_df = pd.concat([results_df_a, results_df_b], axis=1)
    
        if results_df is not None:
            output_df = pd.concat([input_data, results_df], axis=1)
            output_df.to_csv(os.path.join(args.output_path, args.generation_type, args.model, "results_turn2{}.csv".format(args.version)))
        
    args.version = "_version_4"
    if "4" in args.version:
        input_data = pd.read_csv(os.path.join(args.output_path, args.generation_type, args.model, "results_turn2{}.csv".format(args.version)))
        input_data.fillna('', inplace=True)

        tag = "exp_4_"
        results_df = get_scores(input_data, tag)

        if results_df is not None:
            output_df = pd.concat([input_data, results_df], axis=1)
            output_df.to_csv(os.path.join(args.output_path, args.generation_type, args.model, "results_turn2{}.csv".format(args.version)))

    args.version = "_version_5"
    if "5" in args.version:
        input_data = pd.read_csv(os.path.join(args.output_path, args.generation_type, args.model, "results_turn2{}.csv".format(args.version)))
        input_data.fillna('', inplace=True)

        tag = "exp_5_"
        results_df = get_scores(input_data, tag)

        if results_df is not None:
            output_df = pd.concat([input_data, results_df], axis=1)
            output_df.to_csv(os.path.join(args.output_path, args.generation_type, args.model, "results_turn2{}.csv".format(args.version)))
    '''
    args.version = ""
    if args.version == "":
        input_data = pd.read_csv(os.path.join(args.output_path, args.generation_type, args.model, "results_turn2{}.csv".format(args.version)))
        input_data.fillna('', inplace=True)
        results_df = get_scores(input_data, '')


        if results_df is not None:
            output_df = pd.concat([input_data, results_df], axis=1)
            output_df.to_csv(os.path.join(args.output_path, args.generation_type, args.model, "results_turn2{}.csv".format(args.version)))
    

if __name__ == "__main__":
    main()