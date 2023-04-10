from langchain.document_loaders import UnstructuredFileLoader
import tiktoken
import matplotlib.pyplot as plt
import seaborn as sns


loader = UnstructuredFileLoader("/home/dan/langchain_projects/chunker/cal.html")
docs = loader.load()

print(len(docs))


tokenizer = tiktoken.get_encoding('cl100k_base')

# create the length function
def tiktoken_len(text):
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)

token_counts = [tiktoken_len(doc.page_content) for doc in docs]
print(token_counts)

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
