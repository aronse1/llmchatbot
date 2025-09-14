
from pathlib import Path
from collections import defaultdict, Counter

from llama_index.core import Document
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter

def md_to_sentence_chunks_with_numbers(
    md_path: str,
    chunk_size: int = 512,
    chunk_overlap: int = 80,
):
    
    text = Path(md_path).read_text(encoding="utf-8")
    base_doc = Document(text=text, metadata={"file_name": Path(md_path).name})

    md_parser = MarkdownNodeParser.from_defaults(include_metadata=True, include_prev_next_rel=True)
    md_nodes = md_parser.get_nodes_from_documents([base_doc])

    for n in md_nodes:
        headers = n.metadata.get("header_path", [])
        n.metadata.update({
            "section": headers[0] if len(headers) > 0 else None,
            "subsection": headers[1] if len(headers) > 1 else None,
            "subsubsection": headers[2] if len(headers) > 2 else None,
        })

    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


    docs_for_split = []
    for parent_idx, n in enumerate(md_nodes):
        docs_for_split.append(
            Document(
                text=n.text,
                metadata={
                    **n.metadata,
                    "parent_idx": parent_idx,  
                },
            )
        )

    small_nodes = splitter.get_nodes_from_documents(docs_for_split)

    by_parent = defaultdict(list)
    for sn in small_nodes:
        by_parent[sn.metadata["parent_idx"]].append(sn)

    global_ctr = Counter()  
    global_ctr["i"] = 0

    for parent_idx, group in by_parent.items():
        for local_i, sn in enumerate(group, start=1):
            global_ctr["i"] += 1
            sn.metadata["chunk_idx"] = local_i
            sn.metadata["global_chunk_idx"] = global_ctr["i"]
            sn.metadata["num_chunks_in_parent"] = len(group)
            # Schön formatiert, z.B. doc|section|0001
            sn.metadata["chunk_id"] = (
                f"{sn.metadata['file_name']}"
                f"|{sn.metadata.get('section') or 'ROOT'}"
                f"|{local_i:04d}"
            )

    return small_nodes

def mdtable_to_sentence_chunks_with_numbers(
    md_path: str,
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 80,

):
    base_doc = Document(text=text, metadata={"file_name": Path(md_path).name})

    md_parser = MarkdownNodeParser.from_defaults(include_metadata=True, include_prev_next_rel=True)
    md_nodes = md_parser.get_nodes_from_documents([base_doc])

    for n in md_nodes:
        headers = n.metadata.get("header_path", [])
        n.metadata.update({
            "section": headers[0] if len(headers) > 0 else None,
            "subsection": headers[1] if len(headers) > 1 else None,
            "subsubsection": headers[2] if len(headers) > 2 else None,
        })

    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


    docs_for_split = []
    for parent_idx, n in enumerate(md_nodes):
        docs_for_split.append(
            Document(
                text=n.text,
                metadata={
                    **n.metadata,
                    "parent_idx": parent_idx,  
                },
            )
        )

    small_nodes = splitter.get_nodes_from_documents(docs_for_split)

    by_parent = defaultdict(list)
    for sn in small_nodes:
        by_parent[sn.metadata["parent_idx"]].append(sn)

    global_ctr = Counter()  
    global_ctr["i"] = 0

    for parent_idx, group in by_parent.items():
        for local_i, sn in enumerate(group, start=1):
            global_ctr["i"] += 1
            sn.metadata["chunk_idx"] = local_i
            sn.metadata["global_chunk_idx"] = global_ctr["i"]
            sn.metadata["num_chunks_in_parent"] = len(group)
            # Schön formatiert, z.B. doc|section|0001
            sn.metadata["chunk_id"] = (
                f"{sn.metadata['file_name']}"
                f"|{sn.metadata.get('section') or 'ROOT'}"
                f"|{local_i:04d}"
            )

    return small_nodes

