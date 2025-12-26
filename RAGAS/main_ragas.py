import os
import asyncio
from dotenv import load_dotenv
import pandas as pd

# OpenAI Client
from openai import OpenAI

# Ragas Imports
from ragas.testset import TestsetGenerator
from ragas.testset.persona import Persona
from ragas.testset.synthesizers import (
    SingleHopSpecificQuerySynthesizer,
    MultiHopSpecificQuerySynthesizer,
    MultiHopAbstractQuerySynthesizer
)
from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings

# Document Loading & Splitting
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Load Environment Variables
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Please set OPENAI_API_KEY in your .env file")

async def main():
    print("--- Starting RAG Testset Generation (Chilean PoC) ---")

    # 2. Setup Models
    openai_client = OpenAI(api_key=api_key)

    # We use gpt-5-nano. 
    # Note: We rely on the model's default max_tokens, but we control input size via chunking.
    ragas_llm = llm_factory(
        model="gpt-5-nano", 
        client=openai_client
    )

    ragas_embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small", 
        client=openai_client
    )

    # 3. Load Documents
    print(f"Loading documents from ./knowledge_base...")
    loader = DirectoryLoader(
        "./knowledge_base", 
        glob="*.md", 
        loader_cls=UnstructuredMarkdownLoader
    )
    raw_documents = loader.load()
    print(f"Loaded {len(raw_documents)} raw documents.")

    # 4. CRITICAL FIX: Split documents manually
    # We split into smaller chunks (1024 chars) to prevent the LLM from 
    # hitting output token limits during summarization.
    print("Splitting documents to prevent token overflow...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=200
    )
    documents = text_splitter.split_documents(raw_documents)
    print(f"Created {len(documents)} chunks from raw documents.")

    # 5. Define Chilean Personas
    personas = [
        Persona(
            name="Joven Profesional",
            role_description="Soy un joven profesional chileno de 26 años. Es mi primera vez lidiando con bancos. "
                             "Hablo de forma informal, uso modismos chilenos suaves (como 'cachái', 'pega', 'plata'). "
                             "Me interesan las cosas rápidas, digitales y no entiendo términos técnicos financieros. "
                             "Hago preguntas abstractas sobre 'cómo lograr cosas' más que pedir datos duros."
        ),
        Persona(
            name="Dueña de Pyme",
            role_description="Tengo una pequeña empresa de repostería en Santiago. "
                             "Necesito capital de trabajo y ordenarme. "
                             "Soy directa, pragmática y busco soluciones para mi negocio. "
                             "Pregunto sobre financiamiento, plazos y requisitos reales."
        ),
        Persona(
            name="Jefe de Hogar",
            role_description="Padre de familia, 45 años. Busco seguridad y ahorrar para la universidad de mis hijos. "
                             "Soy desconfiado con la letra chica. "
                             "Hago preguntas sobre beneficios, tasas, seguridad y créditos de consumo. "
                             "Hablo en español de Chile estándar."
        )
    ]

    # 6. Define Query Distribution
    query_distribution = [
        (MultiHopAbstractQuerySynthesizer(llm=ragas_llm), 0.6),
        (SingleHopSpecificQuerySynthesizer(llm=ragas_llm), 0.2),
        (MultiHopSpecificQuerySynthesizer(llm=ragas_llm), 0.2),
    ]

    # 7. Initialize TestsetGenerator
    generator = TestsetGenerator(
        llm=ragas_llm,
        embedding_model=ragas_embeddings,
        persona_list=personas,
        llm_context="Toda la generación de preguntas, respuestas y contexto debe ser estrictamente en Español de Chile. "
                    "Evita el español neutro o de España. Usa un tono de 'Asistente Virtual' útil."
    )

    # 8. Generate Testset using 'generate_with_chunks'
    print(f"Generating 20 samples using Knowledge Graph approach...")
    
    # We use generate_with_chunks because we already split the docs manually
    testset = generator.generate_with_chunks(
        chunks=documents, 
        testset_size=20,
        query_distribution=query_distribution,
        raise_exceptions=False,
        with_debugging_logs=True 
    )

    # 9. Export Results
    df = testset.to_pandas()
    
    print("\nGeneration Complete! Preview:")
    if not df.empty:
        print(df[['user_input', 'reference', 'synthesizer_name']].head())
        output_filename = "ragas_testset_chilean_poc.csv"
        df.to_csv(output_filename, index=False)
        print(f"\nSaved full testset to {output_filename}")
    else:
        print("DataFrame is empty. No samples were generated.")

if __name__ == "__main__":
    asyncio.run(main())