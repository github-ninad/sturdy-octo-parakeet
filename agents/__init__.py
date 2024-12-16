from phi.embedder.openai import OpenAIEmbedder
from phi.knowledge.pdf import PDFKnowledgeBase
from phi.vectordb.lancedb import LanceDb, SearchType
from phi.reranker.cohere import CohereReranker
import os
os.environ["OPENAI_API_KEY"] = "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"
os.environ["GROQ_API_KEY"] = "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"
os.environ["CO_API_KEY"] = "IZeUXEuPCG4dTFunJbM20oNtDxwqRNdywlALxC0x"
