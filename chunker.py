from langchain.document_loaders import UnstructuredFileLoader
import tiktoken
import matplotlib.pyplot as plt
import seaborn as sns
from langchain.text_splitter import RecursiveCharacterTextSplitter
import hashlib
from tqdm.auto import tqdm
import json


loader = UnstructuredFileLoader("/home/dan/langchain_projects/chunker/cal.html")
docs = loader.load()
print(len(docs))

tokenizer = tiktoken.get_encoding('cl100k_base')
m = hashlib.md5()

documents = []

# create the length function
def tiktoken_len(text):
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)

#token_counts = [tiktoken_len(doc.page_content) for doc in docs]
#print(token_counts)

# # set style and color palette for the plot
# sns.set_style("whitegrid")
# sns.set_palette("muted")

# # create histogram
# plt.figure(figsize=(12, 6))
# sns.histplot(token_counts, kde=False, bins=50)

# # customize the plot info
# plt.title("Token Counts Histogram")
# plt.xlabel("Token Count")
# plt.ylabel("Frequency")

# plt.show()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=20,  # number of tokens overlap between chunks
    length_function=tiktoken_len,
    separators=['\n\n', '\n', ' ', '']
)

chunks = text_splitter.split_text(docs[0].page_content)
print(len(chunks))

for doc in tqdm(docs):
    url = doc.metadata['source'].replace('rtdocs/', 'https://')
    m.update(url.encode('utf-8'))
    uid = m.hexdigest()[:12]
    chunks = text_splitter.split_text(doc.page_content)
    for i, chunk in enumerate(chunks):
        documents.append({
            'id': f'{uid}-{i}',
            'text': chunk,
            'source': url
        })

print(len(documents))

with open('train.jsonl', 'w') as f:
    for doc in documents:
        f.write(json.dumps(doc) + '\n')

with open('train.jsonl', 'r') as f:
    for line in f:
        documents.append(json.loads(line))

len(documents)

print(documents[10])
