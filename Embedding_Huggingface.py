import os
import re
import datetime
import torch
import warnings
from transformers import logging as hf_logging
from dotenv import load_dotenv

# --- TẮT CẢNH BÁO ---
warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- IMPORT ---
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_groq import ChatGroq
    from langchain_chroma import Chroma
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
except ImportError:
    import os
    os.system('pip install -q langchain-groq langchain-huggingface huggingface_hub langchain-chroma')
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_groq import ChatGroq
    from langchain_chroma import Chroma
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser

# =============================================================================
# CẤU HÌNH HỆ THỐNG
# =============================================================================
PERSIST_DIRECTORY = "/content/drive/MyDrive/chroma_db_bge_m3"



# 1. Nạp các biến từ file .env vào chương trình
load_dotenv()

# 2. Lấy key an toàn (Lúc này Python sẽ tự tìm trong file .env)
HF_TOKEN = os.getenv("HF_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Kiểm tra xem có lấy được key không (chỉ dùng khi test, xóa trước khi push hoặc cẩn thận khi in)
if not HF_TOKEN or not GROQ_API_KEY:
    raise ValueError("Chưa tìm thấy API Key. Hãy kiểm tra file .env!")

# 3. Gán vào biến môi trường hệ thống (để các thư viện khác tự nhận diện)
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

print("Đã nạp API Key thành công!")

# =============================================================================
# 1. TỪ ĐIỂN DỮ LIỆU (Đảm bảo có OCH)
# =============================================================================
STOCK_SYMBOLS = {
    "AAV", "ADC", "ALT", "AMC", "AME", "AMV", "API", "APS", "ARM", "ATS", "BAB", "BAX", "BBS", "BCC", "BCF",
    "BDB", "BED", "BKC", "BNA", "BPC", "BSC", "BST", "BTS", "BTW", "BVS", "BXH", "C69", "CAG", "CAN", "CAP",
    "CAR", "CCR", "CDN", "CEO", "CET", "CIA", "CJC", "CKV", "CLH", "CLM", "CMC", "CMS", "CPC", "CSC", "CTB",
    "CTC", "CTP", "CTT", "CVN", "CX8", "D11", "DAD", "DAE", "DC2", "DDG", "DHP", "DHT", "DIH", "DL1", "DNC",
    "DNP", "DP3", "DS3", "DST", "DTC", "DTD", "DTG", "DTK", "DVM", "DXP", "EBS", "ECI", "EID", "EVS", "FID",
    "GDW", "GIC", "GKM", "GLT", "GMA", "GMX", "HAD", "HAT", "HBS", "HCC", "HCT", "HDA", "HEV", "HGM", "HHC",
    "HJS", "HKT", "HLC", "HLD", "HMH", "HMR", "HOM", "HTC", "HUT", "HVT", "ICG", "IDC", "IDJ", "IDV", "INC",
    "INN", "IPA", "ITQ", "IVS", "KDM", "KHS", "KKC", "KMT", "KSD", "KSF", "KSQ", "KST", "KSV", "KTS", "L14",
    "L18", "L40", "LAS", "LBE", "LCD", "LDP", "LHC", "LIG", "MAC", "MAS", "MBG", "MBS", "MCC", "MCF", "MCO",
    "MDC", "MED", "MEL", "MKV", "MST", "MVB", "NAG", "NAP", "NBC", "NBP", "NBW", "NDN", "NDX", "NET", "NFC",
    "NHC", "NRC", "NSH", "NST", "NTH", "NTP", "NVB", "OCH", "ONE", "PBP", "PCE", "PCG", "PCH", "PCT", "PDB",
    "PEN", "PGN", "PGS", "PGT", "PHN", "PIA", "PIC", "PJC", "PLC", "PMB", "PMC", "PMP", "PMS", "POT", "PPE",
    "PPP", "PPS", "PPT", "PPY", "PRC", "PRE", "PSC", "PSD", "PSE", "PSI", "PSW", "PTD", "PTI", "PTS", "PTX",
    "PV2", "PVB", "PVC", "PVG", "PVI", "PVS", "QHD", "QST", "QTC", "RCL", "S55", "S99", "SAF", "SCG", "SCI",
    "SD5", "SD9", "SDA", "SDC", "SDG", "SDN", "SDU", "SEB", "SED", "SFN", "SGC", "SGD", "SGH", "SHE", "SHN",
    "SHS", "SJ1", "SJE", "SLS", "SMN", "SMT", "SPC", "SPI", "SRA", "SSM", "STC", "STP", "SVN", "SZB", "TA9",
    "TBX", "TDT", "TET", "TFC", "THB", "THD", "THS", "THT", "TIG", "TJC", "TKU", "TMB", "TMC", "TMX", "TNG",
    "TOT", "TPH", "TPP", "TSB", "TTC", "TTH", "TTL", "TTT", "TV3", "TV4", "TVC", "TVD", "TXM", "UNI", "V12",
    "V21", "VBC", "VC1", "VC2", "VC3", "VC6", "VC7", "VC9", "VCC", "VCM", "VCS", "VDL", "VE1", "VE3", "VE4",
    "VE8", "VFS", "VGP", "VGS", "VHE", "VHL", "VIF", "VIG", "VIT", "VLA", "VMC", "VMS", "VNC", "VNF", "VNR",
    "VNT", "VSA", "VSM", "VTC", "VTH", "VTJ", "VTV", "VTZ", "WCS", "WSS", "X20"
}

# Từ khóa dễ gây nhầm lẫn
STOP_WORDS = {
    "MUA", "BAN", "GIA", "LOI", "LAI", "VON", "NO", "PHI", "THU", "CHI",
    "KHI", "NAO", "DAU", "SAO", "VOI", "CUA", "CHO", "CAC", "TAI", "TU",
    "TOI", "ANH", "CHI", "EM", "NO", "HO", "LA", "CO", "VE", "XEM",
    "NAM", "QUY", "TIN", "HOT", "TOP", "GDP", "CPI", "USD", "VND"
}

FINANCIAL_TERM_MAPPING = {
    "LỢI NHUẬN SAU THUẾ": ["lãi ròng", "lợi nhuận ròng", "lãi sau thuế", "lnst", "lời", "lãi", "lợi nhuận"],
    "LỢI NHUẬN TRƯỚC THUẾ": ["lntt", "lãi trước thuế"],
    "DOANH THU": ["doanh số", "tổng thu", "dt", "doanh thu thuần"],
    "TỔNG TÀI SẢN": ["tài sản", "tts"],
    "VỐN CHỦ SỞ HỮU": ["vốn chủ", "vcsh"],
    "NỢ PHẢI TRẢ": ["nợ", "tổng nợ"],
    "CỔ TỨC": ["chia cổ tức", "trả cổ tức"],
}

# =============================================================================
# 2. XỬ LÝ LOGIC (DÙNG TỪ ĐIỂN - CHÍNH XÁC CAO)
# =============================================================================
def handle_fast_chit_chat(query):
    q = query.lower().strip()
    if any(q == g or q.startswith(g + " ") for g in ['hi', 'hello', 'chào', 'alo']):
        return "Chào bạn! Tôi là trợ lý tài chính (Fix OCH). Mời bạn hỏi số liệu."
    if any(q == g or q.startswith(g + " ") for g in ['bye', 'tạm biệt', 'goodbye', 'stop']):
        return "Tạm biệt!"
    return None

def extract_years_from_query(query):
    current_year = datetime.datetime.now().year
    matches = re.findall(r'\b([12]\d{3})\b', query) # <--- Đã sửa (thẳng hàng với current_year)

    found_years = set([int(m) for m in matches])

    query_lower = query.lower()
    time_mapping = {
        "năm nay": current_year,
        "năm ngoái": current_year - 1, # Đã sửa lại cho tự động thay vì fix cứng 2024
        "năm trước": current_year - 1,
        "năm kia": current_year - 2,
        "covid": [2020, 2021]
    }
    for key, val in time_mapping.items():
        if key in query_lower:
            if isinstance(val, list): found_years.update(val)
            else: found_years.add(val)
    return list(found_years)

def extract_symbols_from_query(query):
    """
    LOGIC MỚI: Tách từ thông minh và so khớp với STOCK_SYMBOLS.
    Đảm bảo 100% bắt được OCH nếu nó có trong danh sách.
    """
    q_upper = query.upper()
    # Thay thế các ký tự đặc biệt bằng khoảng trắng để tách từ chuẩn hơn
    q_clean = re.sub(r'[^\w\s]', ' ', q_upper)
    tokens = q_clean.split()

    found_symbols = []
    seen = set()

    for token in tokens:
        # Kiểm tra trực tiếp trong TỪ ĐIỂN
        if token in STOCK_SYMBOLS and token not in STOP_WORDS:
            if token not in seen:
                found_symbols.append(token)
                seen.add(token)

    return found_symbols

def analyze_query_rule_based(query):
    years = extract_years_from_query(query)
    symbols = extract_symbols_from_query(query)

    if symbols and years:
        targets = [{"symbol": s, "year": y} for s in symbols for y in years]
        return {"intent": "search", "targets": targets}

    elif symbols or years:
         return {
             "intent": "missing_info",
             "missing": "year" if not years else "symbol",
             "found": {"symbols": symbols, "years": years}
         }

    is_financial = any(kw in query.lower() for kw in ["tài sản", "vốn", "lãi", "lỗ", "doanh thu", "nợ"])
    if is_financial:
        return {"intent": "missing_info", "missing": "all", "found": {}}

    return {"intent": "chat", "targets": []}

# =============================================================================
# 3. TÌM KIẾM TRỰC TIẾP (NO RERANK - FAST)
# =============================================================================
def direct_retrieve(query, analyzed_data, vector_db):
    if not analyzed_data or analyzed_data.get('intent') != 'search':
        return []

    targets = analyzed_data.get('targets', [])
    k_per_target = 3
    raw_docs = []

    expanded_query = query
    for term, synonyms in FINANCIAL_TERM_MAPPING.items():
        for syn in synonyms:
             if syn.lower() in query.lower():
                 expanded_query += f" ({term})"
                 break

    print(f"--> [Tìm kiếm] Mục tiêu: {targets} ...")

    for target in targets:
        symbol = target.get('symbol')
        year = target.get('year')

        # FILTER CỨNG
        filters = {}
        if symbol: filters['symbol'] = symbol
        if year: filters['year'] = year
        chroma_filter = {"$and": [{k: v} for k, v in filters.items()]} if len(filters) > 1 else filters

        try:
            docs = vector_db.similarity_search(expanded_query, k=k_per_target, filter=chroma_filter)
            raw_docs.extend(docs)
        except: pass

    return raw_docs

# =============================================================================
# 4. CHƯƠNG TRÌNH CHÍNH
# =============================================================================
def main():
    print("==================================================")
    print("   CHATBOT TÀI CHÍNH (OCH FIX + 70B FAST)         ")
    print("==================================================")

    # 1. Kết nối Database
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"--> Load Embedding BGE-M3 (Device: {device})...")
        embedding_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True}
        )
        vector_db = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embedding_model,
            collection_name="financial_reports_bge"
        )
    except Exception as e:
        print(f"❌ Lỗi DB: {e}"); return

    # 2. Kết nối Groq (Model 70B theo yêu cầu)
    try:
        print("--> Đang kết nối Groq (Llama 3.3 - 70B Versatile)...")
        llm = ChatGroq(
            temperature=0.0,
            model_name="llama-3.3-70b-versatile",
            groq_api_key=GROQ_API_KEY
        )
    except Exception as e:
        print(f"❌ Lỗi Groq: {e}"); return

    # 3. PROMPT NÂNG CAO (CHO PHÉP TÍNH TOÁN)
    prompt_template = PromptTemplate.from_template(
        """Bạn là Chuyên gia Tài chính. Dựa vào Context được cung cấp bên dưới, hãy trả lời câu hỏi của người dùng.

        Context:
        {context}

        Câu hỏi: {question}

        QUY TẮC TRẢ LỜI:
        1. **Tra cứu**: Nếu câu hỏi yêu cầu tìm kiếm chỉ số có sẵn trong database thì chỉ cần trích xuất số liệu theo yêu cầu, không cần giải thích thêm.
        2. **Thực hiện phép tính:** Nếu câu hỏi yêu cầu cộng/trừ hoặc so sánh (Ví dụ: "So sánh năm 2020 và 2021", "Tổng tài sản trừ Nợ"), bạn BẮT BUỘC phải thực hiện phép tính và trình bày rõ ràng (Ví dụ: A - B = Kết quả).
        3. **So sánh:** Nếu so sánh giữa 2 năm, hãy tính ra số chênh lệch tuyệt đối và phần trăm tăng trưởng (nếu có thể).
        4. **Diễn giải:** Trả lời bằng câu hoàn chỉnh, chuyên nghiệp, không cộc lốc,
        5. **Nguồn:** Ghi rõ [Nguồn: Báo cáo <Loại> - <Mã> - <Năm>] ở cuối câu trả lời.

        Trả lời bằng Tiếng Việt:
        """
    )

    # Cấu trúc Chain chuẩn
    final_chain = prompt_template | llm | StrOutputParser()

    print("\n✅ Sẵn sàng! (Gõ 'exit' để thoát)")

    while True:
        try:
            print("-" * 60)
            user_query = input("Bạn: ").strip()
            if not user_query: continue
            if user_query.lower() in ["exit", "quit"]: break

            if handle_fast_chit_chat(user_query):
                print(f"Bot: {handle_fast_chit_chat(user_query)}")
                continue

            # Phân tích
            analysis = analyze_query_rule_based(user_query)
            intent = analysis.get('intent')

            if intent == 'chat':
                print("Bot: Tôi chỉ trả lời về tài chính thôi ^^")
                continue

            if intent == 'missing_info':
                found = analysis.get('found', {})
                print(f"Bot: Vui lòng cung cấp đủ **Mã** và **Năm**. (Đã nhận diện: {found})")
                continue

            # Tìm kiếm
            docs = direct_retrieve(user_query, analysis, vector_db)

            if not docs:
                print(f"Bot: Không tìm thấy dữ liệu trong DB cho {analysis['targets']}.")
                continue

            # Trả lời
            context_text = "\n".join([d.page_content for d in docs])
            response = final_chain.invoke({"context": context_text, "question": user_query})

            if "</think>" in response: response = response.split("</think>")[-1].strip()
            print(f"Bot: {response}")

        except Exception as e:
            print(f"Bot: Lỗi hệ thống ({e})")

if __name__ == "__main__":
    main()