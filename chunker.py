from langchain.document_loaders import UnstructuredFileLoader
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
import hashlib
from tqdm.auto import tqdm
import json
#import matplotlib.pyplot as plt
#import seaborn as sns

# Initialize the loader and load documents
loader = UnstructuredFileLoader("/home/dan/langchain_projects/chunker/cal.html")
docs = loader.load()
print(len(docs))

# Initialize the tokenizer
tokenizer = tiktoken.get_encoding('cl100k_base')

# Initialize the MD5 hash object
m = hashlib.md5()

# Initialize the documents list
documents = []

# Define the length function
def tiktoken_len(text):
    tokens = tokenizer.encode(text, disallowed_special=())
    return len(tokens)

# Initialize the text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=20,  # number of tokens overlap between chunks
    length_function=tiktoken_len,
    separators=['\n\n', '\n', ' ', '']
)

# Split the text of the first document
chunks = text_splitter.split_text(docs[0].page_content)
print(len(chunks))

# Process each document
for doc in tqdm(docs):
    url = doc.metadata['source'].replace('rtdocs/', 'https://')
    m.update(url.encode('utf-8'))
    uid = m.hexdigest()[:12]
    chunks = text_splitter.split_text(doc.page_content)
    
    # Create an entry for each chunk
    for i, chunk in enumerate(chunks):
        documents.append({
            'id': f'{uid}-{i}',
            'text': chunk,
            'source': url
        })

# Print the total number of documents
print(len(documents))

# Save documents to a file
with open('train.jsonl', 'w') as f:
    for doc in documents:
        f.write(json.dumps(doc) + '\n')

# Read the documents from the file
with open('train.jsonl', 'r') as f:
    for line in f:
        documents.append(json.loads(line))

# Print the total number of documents after reading from the file
print(len(documents))

# Print the 11th document
print(documents[10])



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
