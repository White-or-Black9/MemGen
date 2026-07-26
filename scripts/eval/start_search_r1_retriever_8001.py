"""Launch the local Search-R1 E5 retriever on the MemGen-compatible port 8001."""

import argparse
import sys


SEARCH_R1_ROOT = "/mnt/18T/baishilong/Search-R1"
DEFAULT_INDEX_PATH = "/mnt/18T/baishilong/retrieval_assets/wiki-18/e5_Flat.index"
DEFAULT_CORPUS_PATH = "/mnt/18T/baishilong/retrieval_assets/wiki-18/wiki-18.jsonl"
DEFAULT_MODEL_PATH = "/mnt/18T/baishilong/retrieval_assets/e5-base-v2"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch local Search-R1 E5 retrieval for MemGen TriviaQA."
    )
    parser.add_argument("--index-path", default=DEFAULT_INDEX_PATH)
    parser.add_argument("--corpus-path", default=DEFAULT_CORPUS_PATH)
    parser.add_argument("--retriever-model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--faiss-gpu", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    sys.path.insert(0, SEARCH_R1_ROOT)
    import uvicorn
    from search_r1.search import retrieval_server

    retrieval_server.config = retrieval_server.Config(
        retrieval_method="e5",
        retrieval_topk=args.topk,
        index_path=args.index_path,
        corpus_path=args.corpus_path,
        faiss_gpu=args.faiss_gpu,
        retrieval_model_path=args.retriever_model,
        retrieval_pooling_method="mean",
        retrieval_query_max_length=256,
        retrieval_use_fp16=True,
        retrieval_batch_size=512,
    )
    retrieval_server.retriever = retrieval_server.get_retriever(
        retrieval_server.config
    )
    uvicorn.run(retrieval_server.app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
