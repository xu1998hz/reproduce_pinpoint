import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

def parse_text(text, task, language):
    if task == 'mt':
        # extract src text
        if language == 'zh-en':
            return text[23: -50]
        elif language == 'en-de': 
            return text[23: -49]
        else:
            raise NotImplementedError
    elif task == 'qa':
        return text[10:]
    elif task == 'summ':
        # find 'Question:'
        # idx = text.find('Question:')
        # q = text[idx:]
        # return text[: idx], q
        return text
    else:
        raise NotImplementedError


# LlaMA prompt format:
# <s>[INST] <<SYS>>
# {{ system_prompt }}
# <</SYS>>

# {{ user_msg_1 }} [/INST] {{ model_answer_1 }} </s><s>[INST] {{ user_msg_2 }} [/INST]

# TODO: save prompt to some file and load prompt from that

class TSVDataSet(Dataset):

    def __init__(self, tsv_file, task, language=None, nrows=None):
        if tsv_file:
            if nrows: # for testing
                self.data = pd.read_csv(tsv_file, sep='\t', nrows=nrows)
            else:
               self.data = pd.read_csv(tsv_file, sep='\t')
        self.task = task
        self.language = language
        if self.task == 'mt':
            self.system_prompt = "<s>[INST] <<SYS>>\n" + \
                "Perform the following translation task, providing only the translated text without any additional explanation or comments." + \
                "\n<</SYS>>\n\n"
            if self.language == 'zh-en':
                self.example_prompt = "Translating Chinese into English. Chinese:\n以免再次发生这样的事情. \nEnglish: [/INST] " + \
                    "So that such a thing won't happen again. </s>" + \
                    "<s>[INST] Translating Chinese into English. Chinese:\n众所周知，在事故发生后救护车来的过程中其实是还有一段时间的，而正是这段时间才是最关键的。\nEnglish: [/INST] " + \
                    "As everyone knows, there is a period before the coming of the ambulance after the accident happens, which is most critical. </s>" 
                self.start_prompt = "<s>[INST] Translating Chinese into English. Chinese:\n"
                self.end_prompt = " \nEnglish: [/INST]"
            elif self.language == 'en-de':
                self.example_prompt = "Translating English into German. English:\nI sincerely hope you get to find a resolution \nGerman: [/INST] " + \
                    "Ich hoffe wirklich, dass Sie eine Lösung finden werden </s>" + \
                    "<s>[INST] Translating English into German. English:\nIf you require your order urgently, please choose the express courier postage option (if this is not shown for your country, please contact us for a quote). \nGerman: [/INST] " + \
                    "Wenn Sie Ihre Bestellung dringend brauchen, wählen Sie bitte die Option für Versand mit Expresskurier (wenn sie für Ihr Land nicht angezeigt wird, wenden Sie sich für einen Kostenvoranschlag an uns). </s>"
                self.start_prompt = "<s>[INST] Translating English into German. English:\n"
                self.end_prompt = " \nGerman: [/INST]"
            else:
                raise NotImplementedError
        elif self.task == 'qa':
            # TODO: TBD
            self.system_prompt = "<s>[INST] <<SYS>>\n" + \
                "Write a comprehensive, paragraph-long answer to the following question. Ensure that your answer integrates the given factoid context, is well-structured, and provides a thorough explanation or analysis as required by the question. Your response should be clear, concise, and focused, forming a cohesive paragraph." + \
                "\n<</SYS>>\n\n"
            self.start_prompt = "Question: "
            self.end_prompt = " \nAnswer: [/INST]"
            self.example_prompt = ''
        elif self.task == 'summ':
            self.system_prompt = "<s>[INST] <<SYS>>\n" + \
                "Read the following passage. Then, based on the passage, summarize the specific aspect or section as indicated in the question. Your summary should be concise, accurate, and focused on the relevant part of the passage, capturing the key points and main ideas related to the query" + \
                "\n<</SYS>>\n\n"
            # self.start_prompt = "Passage: "
            self.start_prompt = ''
            self.end_prompt = " \nAnswer: [/INST]"
            self.example_prompt = ''
        else:
            raise NotImplementedError
        self.p_lens = [len(self.system_prompt) + len(self.example_prompt) + len(self.start_prompt), len(self.end_prompt)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.task == 'mt':
            input = self.data.inputs[idx]
            target = self.data.targets[idx]
        elif self.task == 'qa':
            input = self.data.source[idx]
            r1 = self.data.ref1[idx]
            r2 = self.data.ref2[idx]
            target = [[r1, r2]] # not [r1, r2]!
        elif self.task == 'summ':
            input = self.data.source[idx]
            target = self.data.refs[idx]
        main_prompt = parse_text(input, self.task, self.language)
        prompt = self.system_prompt + self.example_prompt + self.start_prompt + main_prompt + self.end_prompt
        
        return prompt, target

if __name__ == '__main__':
    language = 'en-de'
    task = 'summ'
    # path = f'mt_test_data/{language}_wmt_test_wmt22.tsv'
    path = f'summ_test.tsv'
    dataset = TSVDataSet(path, task, language, nrows=20)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    for batch in dataloader:
        input, target = batch
        print(input[0])
        print('\n')
        print(target[0])
        break