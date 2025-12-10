# ingest.py
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import glob
import json


def load_and_split(pdf_path, chunk_size=1000, chunk_overlap=200):
    """Load a PDF and split it into chunks"""
    print(f"  Loading {pdf_path}...")
    loader = UnstructuredPDFLoader(pdf_path)
    docs = loader.load()
    
    print(f"    Extracted {len(docs)} document(s)")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    split_docs = splitter.split_documents(docs)
    
    print(f"    Split into {len(split_docs)} chunks")
    return split_docs


def main():
    """Process all PDFs in the data/ directory"""
    
    # Create output directory
    os.makedirs("chunks", exist_ok=True)
    
    # Find all PDF files
    pdf_files = glob.glob("data/*.pdf")
    
    if not pdf_files:
        print("ERROR: No PDF files found in data/ directory!")
        print("Please download IPCC PDFs and place them in the data/ folder.")
        return
    
    print(f"Found {len(pdf_files)} PDF file(s) to process\n")
    
    # Process each PDF
    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path}")
        
        try:
            # Load and split the PDF
            docs = load_and_split(pdf_path)
            
            # Convert to JSON-serializable format
            output = [
                {
                    "page_content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in docs
            ]
            
            # Save to JSON file
            output_filename = os.path.join(
                "chunks",
                os.path.basename(pdf_path) + ".json"
            )
            
            with open(output_filename, "w", encoding="utf8") as f:
                json.dump(output, f, ensure_ascii=False, indent=2)
            
            print(f"  ✓ Saved {len(output)} chunks to {output_filename}\n")
            
        except Exception as e:
            print(f"  ✗ ERROR processing {pdf_path}: {e}\n")
            continue
    
    print("✓ Ingestion complete!")


if __name__ == "__main__":
    main()