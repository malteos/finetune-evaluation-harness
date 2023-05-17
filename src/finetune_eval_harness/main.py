import sys
sys.path.append('./')
from finetune_eval_harness.process_args import process_arguments

def main():
    #process_args.process_arguments(sys.argv[1:])
    process_arguments(sys.argv[1:])

if __name__ == "__main__":
    main()
