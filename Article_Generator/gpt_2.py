class GPT2ArticleGenerator():
    def __init__(self, text):
        self.text = text
        self.article = ""
    def generate(self, ngram_size = 2, min_length = 50):
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
        model=GPT2LMHeadModel.from_pretrained("gpt2-large",pad_token_id=tokenizer.eos_token_id)
        input_ids = tokenizer.encode(self.text, return_tensors='pt') 
        output = model.generate(input_ids, max_length = 1024, min_length = min_length, num_beams=5, no_repeat_ngram_size=ngram_size, early_stopping=True)
        self.article = tokenizer.decode(output[0])
    def get_article(self):
       return self.article
