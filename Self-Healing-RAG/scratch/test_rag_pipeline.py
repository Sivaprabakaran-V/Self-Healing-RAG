import os
import sys
import shutil
import time
from pathlib import Path

# Add app directory to Python path at high priority
app_path = str(Path(__file__).resolve().parent.parent / "app")
if app_path not in sys.path:
    sys.path.insert(0, app_path)

# pyrefly: ignore [missing-import]
from rag import SelfHealingRAG

def test_pipeline():
    print("==================================================")
    print("STARTING SELF-HEALING RAG PIPELINE VERIFICATION")
    print("==================================================")
    
    # 1. Instantiate the RAG system
    # This will automatically trigger initial scanning and ingestion of the PDF files
    print("\n--- Test 1: Initialization & Auto-Ingestion ---")
    rag = SelfHealingRAG()
    
    # 2. Test in-scope query (information should be present in PDFs)
    print("\n--- Test 2: In-Scope Query ---")
    in_scope_query = "What is the agent harness?"
    print(f"Query: '{in_scope_query}'")
    result = rag.ask(in_scope_query)
    print("Result:")
    print(result)
    
    assert "answer" in result
    assert "sources" in result
    print("In-scope query test completed successfully.")
    
    # 3. Test out-of-scope query (fallback behavior)
    print("\n--- Test 3: Out-of-Scope Query (Fallback) ---")
    out_of_scope_query = "What is the employee leave policy?"
    print(f"Query: '{out_of_scope_query}'")
    result_fallback = rag.ask(out_of_scope_query)
    print("Result:")
    print(result_fallback)
    
    assert result_fallback["answer"] == "I could not find sufficient information in the uploaded documents."
    assert result_fallback["sources"] == []
    print("Out-of-scope query test completed successfully.")
    
    # 4. Test duplicate/unchanged file skip on re-initialization
    print("\n--- Test 4: Re-initialization & Duplicate/Unchanged Detection ---")
    print("Instantiating RAG again. It should detect that PDFs are unchanged and skip indexing them.")
    rag_second = SelfHealingRAG()
    
    # 5. Test incremental ingestion of a new document
    print("\n--- Test 5: Incremental Ingestion ---")
    temp_doc_path = rag.docs_dir / "temp_security_rule.txt"
    secret_text = "The system administrator password for the backup vault is 'Antigravity-Secure-99'."
    
    print(f"Creating a new temporary file: {temp_doc_path.name}")
    with open(temp_doc_path, "w", encoding="utf-8") as f:
        f.write(secret_text)
        
    try:
        # Re-run update to ingest the new file
        print("Re-running build_or_update_vectorstore...")
        rag_second.build_or_update_vectorstore()
        
        # Query the newly added secret
        secret_query = "What is the system administrator password for the backup vault?"
        print(f"Query: '{secret_query}'")
        secret_result = rag_second.ask(secret_query)
        print("Result:")
        print(secret_result)
        
        assert "Antigravity-Secure-99" in secret_result["answer"]
        assert temp_doc_path.name in secret_result["sources"]
        print("Incremental ingestion and retrieval succeeded!")
        
    finally:
        # 6. Test file deletion and re-indexing
        print("\n--- Test 6: File Deletion & Re-indexing ---")
        print(f"Deleting the temporary file: {temp_doc_path.name}")
        if temp_doc_path.exists():
            temp_doc_path.unlink()
            
        print("Re-running build_or_update_vectorstore to remove deleted file from index...")
        rag_second.build_or_update_vectorstore()
        
        # Query the secret again - it should now be out of scope and fallback
        print(f"Re-querying: '{secret_query}'")
        deleted_secret_result = rag_second.ask(secret_query)
        print("Result after deletion:")
        print(deleted_secret_result)
        
        assert deleted_secret_result["answer"] == "I could not find sufficient information in the uploaded documents."
        print("Deletion cleanup test succeeded!")
        
    print("\n==================================================")
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("==================================================")

if __name__ == "__main__":
    test_pipeline()
