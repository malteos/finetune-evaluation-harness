import sys
#from process_args import process_arguments
#from src.finetune_eval.process_args import process_arguments
#from .process_args import process_arguments
import process_args

def main():
    process_args.process_arguments(sys.argv[1:])

if __name__ == "__main__":
    main()
