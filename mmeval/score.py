from mmeval.scoring.pipeline import discover_result_files, score_result_file
from mmeval.utils.argparser import parse_args


def run_score(args):
    files = discover_result_files(
        out_dir=args.out_dir,
        pattern=args.score_result_glob,
        recursive=args.score_scan_recursive,
    )
    if not files:
        raise RuntimeError(f"No result files found under {args.out_dir} with pattern {args.score_result_glob}")

    done = []
    failed = []
    for result_file in files:
        try:
            done.append(score_result_file(result_file, args))
        except Exception as exc:
            failed.append({"result_file": result_file, "error": str(exc)})

    done.sort(key=lambda x: x["result_file"])
    failed.sort(key=lambda x: x["result_file"])

    for item in done:
        print(
            f"✅ [score] {item['result_file']} -> {item['score_file']} "
            f"(acc={item['summary']['accuracy']:.4f}, resumed={item.get('resumed_count', 0)})"
        )
    if failed:
        for item in failed:
            print(f"❌ [score] {item['result_file']}: {item['error']}")
        raise SystemExit(1)


if __name__ == "__main__":
    args = parse_args()
    run_score(args)
