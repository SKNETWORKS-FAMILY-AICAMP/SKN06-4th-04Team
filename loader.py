from tabula import read_pdf
from langchain.document_loaders import PyMuPDFLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
import os
    
def replace_pattern(patterns, text): 
    _text = text
    for pattern in patterns:
        _text = re.sub(pattern["target_str"], pattern.get("replace_str", ""), _text)
    
    return _text

def load_process_split_doc(
    filename, 
    directory_path="data/tax_etc",
    page_ranges=[],
    table_pages_range='',
    replace_string=[]
    ):

    docs = []
    pdf_file = f"{directory_path}/{filename}.pdf"
    
    # pdf loader
    try:
        text_loader = PyMuPDFLoader(pdf_file)
        texts = text_loader.load()

        for i, text in enumerate(texts):
            text.metadata["page"] = i + 1      

        texts = [
            text for text in texts
            if any(start <= text.metadata.get("page", 0) <= end for start, end in page_ranges)
        ]
    except Exception as e:
        texts = []
        
    # pdf loader - 테이블
    try:
        tables = read_pdf(pdf_file, pages=table_pages_range, stream=True, multiple_tables=True)
        table_texts = [table.to_string(index=False, header=True) for table in tables]
    except Exception as e:
        table_texts = []
    
    text_docs = [
        Document(page_content=text.page_content, metadata={
            "filename": filename, 
            "source": pdf_file
        })
        for text in texts
    ]
        
    table_docs = [
        Document(page_content=text, metadata={
            "filename": filename, 
            "source": f"{pdf_file} - Table {i + 1}"
        })
        for i, text in enumerate(table_texts)
    ]

    # 문서결합 (텍스트 문서 + 표 문서)
    docs.extend(text_docs + table_docs)

    update_docs = []
    
    for doc in docs:
        if doc.page_content:
            edit_content = replace_pattern(replace_string, doc.page_content)
        else:
            edit_content = ""
        
        updated_docs = Document(page_content=edit_content, metadata=doc.metadata)   
        update_docs.append(updated_docs)
        
    # split
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-4o",
        chunk_size=1000,
        chunk_overlap=100,
    )
    split_docs = splitter.split_documents(update_docs)

    return split_docs

def load_process_split_doc_law(
    filename,
    directory_path="data",    
    ):
    """
    PDF 파일들을 처리하여 임베딩을 Chroma Vector Store에 저장합니다.
    """
    
    file_path = os.path.join(directory_path, f"{filename}.pdf")

    loader = PyMuPDFLoader(file_path)
    load_document = loader.load()

    delete_patterns = [
        rf"법제처\s*\d+\s*국가법령정보센터\n{filename.replace('_', ' ')}\n",
        r'\[[\s\S]*?\]', 
        r'<[\s\S]*?>',
    ]
    full_text=" ".join([document.page_content for document in load_document])

    for pattern in delete_patterns:
        full_text = re.sub(pattern, "", full_text)
        
    full_text = re.sub(r"\s+", " ", full_text).strip()
    
    document_list = []

    doc = Document(page_content=full_text, metadata={"source":f"{filename}.pdf", "filename":f"{filename}"})
    document_list.append(doc)

    split_docs=[]

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-4o",
        chunk_size=1000,
        chunk_overlap=100,
    )
    split_docs = splitter.split_documents(document_list)

    return split_docs
# def load_process_split_doc_law(filename):
    
#     """
#     PDF 파일들을 처리하여 임베딩을 Chroma Vector Store에 저장합니다.
#     """
#     document_list = []

#     file_path = f"data/tax_law/{filename}.pdf"

#     loader = PyMuPDFLoader(file_path)
#     load_document = loader.load()

#     # 전처리 - 반복 텍스트 삭제
#     delete_patterns = [
#         rf"법제처\s*\d+\s*국가법령정보센터\n{filename.replace('_', ' ')}\n",
#         r'\[[\s\S]*?\]', 
#         r'<[\s\S]*?>',
#     ]
    
#     full_text = " ".join([document.page_content for document in load_document])
    
#     for pattern in delete_patterns:
#         full_text = re.sub(pattern, "", full_text)
    
#     docs = {"file_name": filename, "full_text": full_text}

#     splitter=RecursiveCharacterTextSplitter.from_tiktoken_encoder(
#         model_name="gpt-4o",
#         chunk_size=1000,
#         chunk_overlap=100,
#     )
#     doc = splitter.split_text(docs['full_text'])

#     metadata = {"file_name": docs["file_name"]}

#     for doc in docs: 
#         _doc = Document(metadata=metadata, page_content=doc)
#         document_list.append(_doc)
        
#     return document_list
