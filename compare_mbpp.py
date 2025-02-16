import json
fp1="/home/u20140041/code/lm-evaluation-harness/evalplus_results/mbpp/fs--archive--share--yulan--data--aa_mini--output--final_stage1_10_rebalanced--checkpoint-13000-rebalanced-llama_vllm_temp_0.3_eval_results.json"
fp2="/home/u20140041/code/lm-evaluation-harness/evalplus_results/mbpp/fs--archive--share--yulan--data--aa_mini--output--miniyulan-2b-stage2--checkpoint-36000-llama_vllm_temp_0.3_eval_results.json"
with open(fp1,"r") as f1,open(fp2,"r") as f2:
    for line in f1:
        obj1=json.loads(line)
        break
    for line in f2:
        obj2=json.loads(line)
        break
    # print(obj1["eval"].keys())
    # print(obj2["eval"].keys())
    pass1_failed2_set=set()
    failed1_pass2_set=set()
    for key in obj1["eval"].keys():
        lis_1 = obj1["eval"][key]
        lis_2 = obj2["eval"][key]
        correct_num_1 = 0
        correct_num_2 = 0
        for dict_now in lis_1:
            if dict_now["base_status"]!="pass":
                correct_num_1+=1
        for dict_now in lis_2:
            if dict_now["base_status"]!="pass":
                correct_num_2+=1
        # print(correct_num_1)
        # print(correct_num_2)
        if correct_num_1>correct_num_2:
            pass1_failed2_set.add(key)
        if correct_num_1<correct_num_2:
            failed1_pass2_set.add(key)
orignal_question_dict={}
question_path="/home/u20140041/.cache/evalplus/MbppPlus-v0.2.0.jsonl"
with open(question_path,"r") as f:
    for line in f:
        obj=json.loads(line)
        orignal_question_dict[obj["task_id"]]=obj["prompt"]
print(len(pass1_failed2_set))
print(len(failed1_pass2_set))
print("第一个答对的数量比第二个多：",pass1_failed2_set)
print("第一个答对的数量比第二个多：",failed1_pass2_set)
pass1_failed2_question=[]
failed1_pass2_question=[]
for key in pass1_failed2_set:
    pass1_failed2_question.append(orignal_question_dict[key])
for key in failed1_pass2_set:
    failed1_pass2_question.append(orignal_question_dict[key])
print("第一个答对的数量比第二个多的问题：")
for i,ques in enumerate(pass1_failed2_question):
    print(f"No{i}:{ques}")

print("第二个答对的数量比第一个多的问题：")
for i,ques in enumerate(failed1_pass2_question):
    print(f"No{i}:{ques}")
