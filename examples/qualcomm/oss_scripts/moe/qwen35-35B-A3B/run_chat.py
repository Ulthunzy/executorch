import json

from workflow.artifacts import check_artifacts
from workflow.config import parse_args
from workflow.inference import inference


def main():
    args = parse_args()
    if args.print_config:
        print(json.dumps(args.model_settings.to_dict(), ensure_ascii=False, indent=2))
        return
    check_artifacts(args)
    return inference(args)


if __name__ == "__main__":
    main()
