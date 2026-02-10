import os
import pandas as pd
import torch
from tqdm import tqdm

# Load dữ liệu
from langchain_community.document_loaders import DataFrameLoader
from langchain_core.documents import Document

# Vector Store (Dùng gói mới langchain_chroma)
from langchain_chroma import Chroma

# Embedding (Dùng gói mới langchain_huggingface)
from langchain_huggingface import HuggingFaceEmbeddings

# (Tùy chọn) Nếu sau này cần chia nhỏ văn bản thì dùng cái này:
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- CẤU HÌNH ---
INPUT_FILE = "C:\\Users\\HP\\Downloads\\NLP API\\Chia_chunks_Full_Finalll.xlsx"
# Đổi tên folder DB để tránh nhầm với cái cũ
PERSIST_DIRECTORY = "chroma_db_bge_m3"

def main():
    # 1. KIỂM TRA GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 THIẾT BỊ: {device.upper()}")
    if device == "cpu":
        print("⚠️ CẢNH BÁO: Nên dùng GPU T4 trên Colab để chạy BGE-M3.")

    # 2. ĐỌC FILE
    print(f"\n--- Đang đọc file: {INPUT_FILE} ---")
    if not os.path.exists(INPUT_FILE):
        print("❌ Lỗi: Không tìm thấy file.")
        return

    try:
        # Đọc file (engine openpyxl cho xlsx)
        df = pd.read_excel(INPUT_FILE, engine='openpyxl')
        print(f"-> Đã đọc {len(df)} dòng dữ liệu.")
    except Exception as e:
        print(f"Lỗi đọc file: {e}")
        return

    # 3. CHUẨN BỊ DOCUMENTS (TỐI ƯU CHO BGE-M3)
    documents = []
    print("--- Đang chuẩn bị Documents ---")

    # BGE-M3 không cần thêm "passage: " như E5
    # Việc này giúp giữ nguyên văn bản gốc và tiết kiệm token

    for index, row in df.iterrows():
        clean_content = str(row['page_content']).strip()

        # Metadata giữ nguyên
        metadata = {
            "symbol": str(row['symbol']).upper().strip() if pd.notna(row['symbol']) else "UNKNOWN",
            "year": int(row['year']) if pd.notna(row['year']) else 0,
            "report_type": str(row['report_type']) if pd.notna(row['report_type']) else "UNKNOWN",
            "source": str(row['source']) if pd.notna(row['source']) else "UNKNOWN"
        }
        documents.append(Document(page_content=clean_content, metadata=metadata))

    # 4. LOAD MODEL BGE-M3 (SOTA HIỆN TẠI)
    print("\n--- Đang tải Model BAAI/bge-m3 (Mạnh hơn E5)... ---")

    model_kwargs = {"device": device}
    encode_kwargs = {"normalize_embeddings": True} # BGE khuyến nghị normalize

    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    # 5. VECTOR HÓA VÀ LƯU
    print(f"\n--- Bắt đầu Vector hóa vào '{PERSIST_DIRECTORY}' ---")

    vector_db = Chroma(
        embedding_function=embedding_model,
        collection_name="financial_reports_bge",
        persist_directory=PERSIST_DIRECTORY
    )

    # Giảm batch size xuống 32 hoặc 16 vì BGE-M3 nặng hơn E5-base
    batch_size = 32
    total_docs = len(documents)

    for i in tqdm(range(0, total_docs, batch_size), desc="Đang Vector hóa"):
        batch = documents[i : i + batch_size]
        vector_db.add_documents(batch)

    print("\n==========================================")
    print("🎉 HOÀN TẤT! Đã nâng cấp lên model BGE-M3.")
    print(f"Lưu tại: {PERSIST_DIRECTORY}")
    print("==========================================")

if __name__ == "__main__":
    main()