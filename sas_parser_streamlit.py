import uuid
import re
import pandas as pd
import csv
from collections import defaultdict
import streamlit as st

# --- Detect chunk type ---
def infer_chunk_type(code: str) -> str:
    upper_code = code.upper()
    first_line = code.strip().split("\n")[0].upper()
    if first_line.startswith("%MACRO"):
        return "MACRO"
    if first_line.startswith("PROC "):
        return "PROC"
    if first_line.startswith("DATA "):
        return "DATA"

    dq_keywords = [
        "DQMATCH", "DQSTANDARDIZE", "DQPARSE", "DQIDENTIFY", "DQGENDER",
        "DQCLUSTER", "DQLOAD", "DQRULE", "DQREVIEW", "DQPROCESS"
    ]
    for dq_keyword in dq_keywords:
        if dq_keyword in upper_code:
            return "DQSTATEMENT"
    return "UNKNOWN"

# --- Extract DQ function names ---
def extract_dq_functions(code: str) -> list:
    dq_keywords = [
        "dqMatch", "dqStandardize", "dqParse", "dqIdentify", "dqGender",
        "dqCluster", "dqLoad", "dqRule", "dqReview", "dqProcess"
    ]
    found = set()
    lower_code = code.lower()
    for func in dq_keywords:
        pattern = r"\b" + re.escape(func.lower()) + r"\b"
        if re.search(pattern, lower_code):
            found.add(func)
    return sorted(list(found))

# --- Count each DQ function in a chunk ---
def count_dq_functions_in_chunk(code: str) -> dict:
    dq_keywords = [
        "dqMatch", "dqStandardize", "dqParse", "dqIdentify", "dqGender",
        "dqCluster", "dqLoad", "dqRule", "dqReview", "dqProcess"
    ]
    lower_code = code.lower()
    dq_counts = {}
    for func in dq_keywords:
        pattern = r"\b" + re.escape(func.lower()) + r"\b"
        count = len(re.findall(pattern, lower_code))
        if count > 0:
            dq_counts[func] = count
    return dq_counts

# --- Chunk SAS string by max lines (simple chunker) ---
def process_sas_string(sas_code: str, max_chunk_size: int = 100):
    lines = sas_code.splitlines()
    chunks = []
    i = 0
    while i < len(lines):
        chunk_lines = lines[i:i + max_chunk_size]
        chunk_code = "\n".join(chunk_lines)
        chunks.append({
            "id": str(uuid.uuid4()),
            "code": chunk_code
        })
        i += max_chunk_size
    return chunks

# --- Save chunks to CSV ---
def save_chunks_to_csv(chunk_list: list, output_file: str):
    for chunk in chunk_list:
        chunk["dq_count"] = chunk.get("dq_count", 0)
        chunk["dq_functions"] = ", ".join(chunk.get("dq_functions", []))
        summary = chunk.get("dq_function_summary", {})
        chunk["dq_function_summary"] = ", ".join(f"{k}:{v}" for k, v in summary.items())
    df = pd.DataFrame(chunk_list)
    df.to_csv(
        output_file,
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        doublequote=True,
        lineterminator="\n"
    )

# --- Main parser node ---
def parse_node(sas_code: str, max_chunk_size: int = 100, logs=None, graph_trace=None):
    raw_chunks = process_sas_string(sas_code, max_chunk_size)
    ast_blocks = []
    dqstatement_count = 0
    dq_function_counts = defaultdict(int)
    for chunk in raw_chunks:
        chunk_type = infer_chunk_type(chunk["code"])
        dq_func_summary = count_dq_functions_in_chunk(chunk["code"])
        dq_funcs = list(dq_func_summary.keys())
        dq_count = 1 if dq_funcs else 0
        dqstatement_count += dq_count
        for func, count in dq_func_summary.items():
            dq_function_counts[func] += count
        ast_blocks.append({
            "id": chunk["id"],
            "type": chunk_type,
            "code": chunk["code"],
            "dq_count": dq_count,
            "dq_functions": dq_funcs,
            "dq_function_summary": dq_func_summary
        })
    if not ast_blocks:
        ast_blocks.append({
            "id": str(uuid.uuid4()),
            "type": "UNKNOWN",
            "code": sas_code,
            "dq_count": 0,
            "dq_functions": [],
            "dq_function_summary": {}
        })
    return ast_blocks, dqstatement_count, dict(dq_function_counts)

# ---------------- STREAMLIT APP ----------------
st.title("SAS DQ Parser & Chunker")

uploaded_file = st.file_uploader("Upload SAS file (.sas)", type=["sas", "txt"])
max_chunk_size = st.number_input("Max lines per chunk", min_value=1, max_value=500, value=100)

if uploaded_file is not None:
    sas_code = uploaded_file.read().decode("utf-8")
    st.text_area("SAS Code Preview (first 5000 chars)", sas_code[:5000], height=200)
    ast_blocks, dqstatement_count, dq_func_summary = parse_node(sas_code, max_chunk_size)
    st.success(f"Parsed {len(ast_blocks)} blocks. DQ Statements: {dqstatement_count}")
    st.json(dq_func_summary)

    # Show DataFrame preview
    df = pd.DataFrame(ast_blocks)
    st.dataframe(df[["id", "type", "dq_count", "dq_functions", "dq_function_summary"]])

    # Download CSV
    from io import StringIO
    csv_buf = StringIO()
    # Convert lists/dicts for CSV
    for row in ast_blocks:
        row["dq_functions"] = ", ".join(row.get("dq_functions", []))
        summary = row.get("dq_function_summary", {})
        if isinstance(summary, dict):
            row["dq_function_summary"] = ", ".join(f"{k}:{v}" for k, v in summary.items())
    df = pd.DataFrame(ast_blocks)
    df.to_csv(csv_buf, index=False, quoting=csv.QUOTE_ALL, escapechar="\\", doublequote=True, lineterminator="\n")
    st.download_button(
        label="Download Results as CSV",
        data=csv_buf.getvalue(),
        file_name="ast_blocks_latest.csv",
        mime="text/csv"
    )