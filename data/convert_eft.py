import statistics as sts
import json

from rex.utils.io import dump_jsonlines, load_jsonlines


def load_tree_data(
    tree_filepath,
    instruction_quality: float = 0.6,
    response_quality: float = 0.6,
    instruction_word_num: int = 5,
    response_word_num: int = 5,
    lang: str = "en",
):
    trees = load_jsonlines(tree_filepath)
    tree_lst = []

    print(f"Loading {len(trees)} trees...")
    for tree in trees:
        ins = tree["prompt"]
        d = {"instruction": ins["text"], "responses": [], "response_quality": []}
        for reply in ins["replies"]:
            if (ins.get("lang") == lang and reply.get("lang") == lang and reply.get("rank") != None):
                inst_qlt = ins["labels"].get("quality", {"value": 0.0})["value"]
                resp_qlt = reply["labels"].get("quality", {"value": 0.0})["value"]
                if inst_qlt > instruction_quality and resp_qlt > response_quality:
                    if (len(ins["text"].split()) > instruction_word_num and len(reply["text"].split()) > response_word_num):
                        d["responses"].append(reply["text"])
                        d["response_quality"].append(resp_qlt)
        if len(d["responses"]) > 1:
            tree_lst.append(d)

    return tree_lst


if __name__ == "__main__":
    instruction_quality = 0
    response_quality = 0
    # 1775 + 531
    dump_num = 2306
    print('not actually truncate by dump_num')
    tree_lst = load_tree_data(
        "2023-04-12_oasst_ready.trees.jsonl", 
        instruction_quality=instruction_quality, 
        response_quality=response_quality, 
    )
    num_responses = sum([len(tree["responses"]) for tree in tree_lst])
    # print average response number
    avg_response_num = num_responses / len(tree_lst)
    print(f"Average response number: {avg_response_num:.2f}")
    print(f"#data: {len(tree_lst)}, #dump: {dump_num}, #responses: {num_responses}")
    # pairs.sort(
    #     key=lambda ins: ins["instruction_quality"] + ins["response_quality"],
    #     reverse=True,
    # )
    # dump_data = pairs[:dump_num]
    instruction_lens = []
    response_lens = []
    with open('tree_lst.json', 'w') as f:
        json.dump(tree_lst, f, indent=4)
    # for ins in dump_data:
    #     instruction_lens.append(len(ins["instruction"]))
    #     response_lens.append(len(ins["response"]))
    # print(
    #     f"Instruction len: {sts.mean(instruction_lens):.0f}±{sts.stdev(instruction_lens):.0f}, "
    #     f"Response len: {sts.mean(response_lens):.0f}±{sts.stdev(response_lens):.0f}"
    # )
    # dump_jsonlines(dump_data, "tree.json")
