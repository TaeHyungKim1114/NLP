import os
import openai
from data import papers
from process import process

from nltk.tokenize import TreebankWordTokenizer, TreebankWordDetokenizer
tb_tokenizer = TreebankWordTokenizer()
tb_d = TreebankWordDetokenizer()

# Load your API key from an environment variable or secret management service
openai.api_key = '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'
#os.getenv('OPENAI_API_KEY')

engine = "text-davinci-003"
max_tokens = 100
temperature = 0.3

token = tb_tokenizer.tokenize(papers.transformer) #treebank tokenizer 이용해서 토큰화, 토큰은 총 7102개

#Abstract ~ reference 전까지
want_delete = []
#논문 제목을 중요하게 생각하는 방법?
token_cut = token[token.index("Abstract"):token.index("References")] #cut 하면 총 Token 5283 개

chunk = []
tokenperchunk = 800

for i in range(len(token_cut)//tokenperchunk+1):
    chunk.append(token_cut[i*tokenperchunk:(i+1)*tokenperchunk])

summary = []

instruct = "Summarize this paper titled 'Attention is all you need': "

for text in chunk:
    prompt = tb_d.detokenize(text)
    new_str = []
    new_str.append(instruct)
    new_str.append(prompt)
    prompt = ''.join(new_str)
    response = openai.Completion.create(engine=engine, prompt=prompt,temperature=temperature, max_tokens=max_tokens)
    summary.append(response.choices[0].text)

summary.append(instruct)
summary_prompt = ' '.join(summary)

response = openai.Completion.create(engine=engine, prompt=summary_prompt,temperature=temperature, max_tokens= 200)

print(response.choices[0].text)
