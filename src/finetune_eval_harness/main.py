import sys, os
sys.path.append(os.getcwd())
from finetune_eval_harness.process_args import process_arguments


def main():

    process_arguments(sys.argv[1:])

if __name__ == "__main__":
    main()
