import sys
from finetune_eval.process_args import process_arguments


def main():
    process_arguments(sys.argv[1:])

if __name__ == "__main__":
    main()
