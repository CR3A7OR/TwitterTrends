class T5SentenceGeneratorCG():
    def __init__(self, text):
        self.text = text
        self.sentence = ""

    def generate(self, max_length=30):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        input_text = self.text
        tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-common_gen")
        model = AutoModelForSeq2SeqLM.from_pretrained("mrm8488/t5-base-finetuned-common_gen")
        features = tokenizer([input_text], return_tensors='pt')

        output = model.generate(input_ids=features['input_ids'], 
                    attention_mask=features['attention_mask'],
                    max_length=max_length,)

        self.sentence =  tokenizer.decode(output[0], skip_special_tokens=True)
    def get_sentence(self):
        return self.sentence