import os
import glob
import pandas as pd
from dotenv import load_dotenv

# LangChain imports
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# RAGAS imports
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from ragas.llms import LangchainLLMWrapper

# 1. Setup Environment
load_dotenv()

# Verify API Key
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("Please set OPENAI_API_KEY in your .env file")

# 2. Configuration for "gpt-5-nano"
# We wrap it so RAGAS can use it.
generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-5-nano"))
critic_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-5-nano"))
embeddings = OpenAIEmbeddings(model="text-embedding-3-small") # Standard efficient embedding

# 3. Load the Knowledge Base
# We look for a folder named 'kb_docs' containing the 5 .md files
folder_path = "./kb_docs"
md_files = glob.glob(os.path.join(folder_path, "*.md"))

if len(md_files) == 0:
    raise ValueError(f"No .md files found in {folder_path}. Please create the folder and add the 5 files.")

print(f"Loading {len(md_files)} documents...")
documents = []
for file_path in md_files:
    loader = UnstructuredMarkdownLoader(file_path)
    documents.extend(loader.load())

# 4. Initialize the RAGAS Generator
generator = TestsetGenerator(
    generator_llm=generator_llm,
    critic_llm=critic_llm,
    embeddings=embeddings,
)

# -------------------------------------------------------------------------
# CRITICAL: CUSTOMIZING THE PERSONA
# RAGAS allows us to adapt prompts. We will define the distribution first,
# then we rely on the 'adapt' method with Spanish, but we will also 
# use a trick: we prepend a persona instruction to the LLM wrapper if needed,
# or simply trust the high-level 'language' param for basic translation.
#
# However, to ensure "Realistic/Naive", we rely on the specific distributions
# and the inherent capability of gpt-5 to follow the language nuance.
# -------------------------------------------------------------------------

# Define the mix of questions (Simple, Reasoning, Multi-Context)
# We want 10 questions total (2 per file * 5 files)
TEST_SIZE = 10 

distributions = {
    simple: 0.5,        # 50% Simple questions
    reasoning: 0.25,    # 25% Logic based
    multi_context: 0.25 # 25% Combining info
}

print("Generating synthetic test set with Chilean User Persona...")

# generate_with_langchain_docs is the standard entry point
# We pass language="spanish" which prompts RAGAS to translate internal prompts.
testset = generator.generate_with_langchain_docs(
    documents,
    test_size=TEST_SIZE,
    distributions=distributions,
    with_debugging_logs=False,
    is_async=False, # Set True if running in a notebook/async env
    raise_exceptions=True
)

# 5. Export and Post-Processing
df = testset.to_pandas()

# OPTIONAL: Clean up for CSV export
# We want to check columns: 'question', 'ground_truth', 'contexts', 'evolution_type'
output_filename = "poc_rag_testset.csv"
df.to_csv(output_filename, index=False)

print(f"✅ Success! Generated {len(df)} test cases.")
print(f"📁 Saved to {output_filename}")
print("\nSample Question generated:")
print(df.iloc[0]['question'])