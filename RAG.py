import chromadb
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer
from langchain_community.chat_models import ChatOllama
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# NEW DATASET
df = pd.read_csv('oscar-dataset/the_oscar_award.csv')

df = df.loc[df['year_ceremony'] == 2023]
df = df.dropna(subset=['film'])
df.loc[:, 'category'] = df['category'].str.lower()
df.loc[:, 'text'] = df['name'] + ' got nominated under the category, ' + df['category'] + ', for the film ' + df[
    'film'] + ' to win the award'
df.loc[df['winner'] == False, 'text'] = df['name'] + ' got nominated under the category, ' + df[
    'category'] + ', for the film ' + df['film'] + ' but did not win'

client = chromadb.Client()
collection = client.get_or_create_collection("oscars-2023")
docs = df["text"].tolist()
ids = [str(x) for x in df.index.tolist()]

collection.add(
    documents=docs,
    ids=ids
)


def text_embedding(text):
    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    return embedding_model.encode(text)


def generate_context(context_query):
    vector = text_embedding(context_query).tolist()

    results = collection.query(
        query_embeddings=vector,
        n_results=5,
        include=["documents"]
    )

    res = "\n".join(str(item) for item in results['documents'][0])
    return res


def chat_completion(sysprompt, usrprompt, length=1000):
    final_prompt = f"""<s>[INST]<<SYS>>
    {sysprompt}
    <</SYS>>

    {usrprompt} [/INST]"""
    return final_prompt


query = "Did Lady Gaga win an award at Oscars 2023?"
context = generate_context(query)

system_prompt = """\
You are a helpful AI assistant that can answer questions on Oscar 2023 awards. Answer based on the context provided. 
If you cannot find the correct answers, say I don't know. Be concise and just include the response.
"""

user_prompt = f"""
Based on the context below:
{context}
Answer the below query:
{query}
"""

print(context)

ollama_llm = "llama2:7b-chat"
model = ChatOllama(model=ollama_llm, temperature=0.1)

prompt_template_name = PromptTemplate(
    input_variables=[],
    template=chat_completion(system_prompt, user_prompt)
)

chain = LLMChain(llm=model, prompt=prompt_template_name)

response = chain({'input': ''})

print("\n", response['text'])
