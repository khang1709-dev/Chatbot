import os
import pandas as pd
import torch
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# --- CẤU HÌNH ---
# Độ dài tối đa cho mỗi chunk (ký tự). 1500 ký tự ~ 300-400 tokens (an toàn cho RAG)
MAX_CHUNK_SIZE = 1500
OUTPUT_FILE = 'Chia_chunks_Full_Finalll.xlsx'

def create_semantic_chunks(file_path, report_type, sheet_name=None):
    """
    Hàm đọc file và chia nhỏ dữ liệu thành các chunk ngữ nghĩa.
    Có xử lý cắt nhỏ nếu dòng quá dài và tự động lặp lại ngữ cảnh.
    """
    print(f"-> Đang xử lý: {report_type}...")

    # Đọc dữ liệu (Hỗ trợ cả Excel và CSV)
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        if sheet_name:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
        else:
            df = pd.read_excel(file_path)

    chunks_data = []

    # Duyệt qua từng dòng (từng công ty theo từng năm)
    for index, row in df.iterrows():
        symbol = str(row.get('Symbol', 'Unknown'))
        year = str(row.get('Năm', 'Unknown'))

        # Metadata cơ bản (dùng để lọc trong Vector DB)
        base_metadata = {
            "symbol": symbol,
            "year": year,
            "report_type": report_type,
            "source": os.path.basename(file_path)
        }

        # TẠO NGỮ CẢNH CƠ SỞ (Context Injection)
        # Câu này sẽ luôn đứng đầu mỗi chunk
        base_intro = f"Dữ liệu {report_type} của công ty {symbol} năm {year}"

        # Bắt đầu chunk mới
        current_text = f"{base_intro}:\n"

        # Duyệt qua từng cột dữ liệu trong dòng
        for col in df.columns:
            # Bỏ qua các cột định danh hoặc cột rỗng
            if col in ['Symbol', 'CP', 'Năm'] or pd.isna(row[col]):
                continue

            val = row[col]
            line = f"- {col}: {val}\n"

            # --- LOGIC KIỂM TRA ĐỘ DÀI ---
            # Nếu thêm dòng này vào mà vượt quá giới hạn -> Cắt chunk
            if len(current_text) + len(line) > MAX_CHUNK_SIZE:
                # 1. Lưu chunk hiện tại
                chunks_data.append({
                    "page_content": current_text,
                    **base_metadata
                })

                # 2. Reset chunk mới VÀ LẶP LẠI NGỮ CẢNH (Quan trọng)
                # Thêm "(tiếp theo)" để phân biệt nhưng vẫn giữ tên công ty/năm
                current_text = f"{base_intro} (tiếp theo):\n{line}"
            else:
                # Nếu chưa đầy, nối tiếp vào chunk đang có
                current_text += line

        # Lưu chunk cuối cùng của dòng (phần dư còn lại)
        if current_text:
            chunks_data.append({
                "page_content": current_text,
                **base_metadata
            })

    return chunks_data

# --- CHƯƠNG TRÌNH CHÍNH ---
if __name__ == "__main__":
    # Danh sách các file đầu vào và loại báo cáo tương ứng
    # Bạn hãy thay đổi đường dẫn file (path) cho đúng với máy tính của bạn
    data_file_path = r"C:\Users\HP\Downloads\bao_cao_tai_chinh_final.xlsx"
    files_map = [
        {
            "path": data_file_path,
            "sheet_name": "BS",
            "type": "Bảng Cân đối kế toán (Balance Sheet)"
        },
        {
            "path": data_file_path,
            "sheet_name": "IS",
            "type": "Báo cáo Kết quả kinh doanh (Income Statement)"
        },
        {
            "path": data_file_path,
            "sheet_name": "CF",
            "type": "Báo cáo Lưu chuyển tiền tệ (Cash Flow)"
        }
    ]

    all_chunks = []

    # Vòng lặp chạy qua tất cả các file
    for item in files_map:
        try:
            file_chunks = create_semantic_chunks(item['path'], item['type'], item.get('sheet_name'))
            all_chunks.extend(file_chunks)
        except Exception as e:
            print(f"Lỗi khi xử lý file {item['path']}: {e}")

    # Chuyển thành DataFrame và lưu ra Excel
    if all_chunks:
        final_df = pd.DataFrame(all_chunks)
        final_df.to_excel(OUTPUT_FILE, index=False)

        print("-" * 30)
        print(f"HOÀN THÀNH! Tổng số chunks tạo ra: {len(final_df)}")
        print(f"File kết quả đã được lưu tại: {OUTPUT_FILE}")

        # In thống kê để kiểm tra
        print("\nSố lượng chunks theo từng loại báo cáo:")
        print(final_df['report_type'].value_counts())
    else:
        print("Không tạo được chunk nào. Vui lòng kiểm tra lại đường dẫn file.")