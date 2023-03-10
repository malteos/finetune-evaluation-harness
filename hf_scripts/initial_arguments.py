from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class InitialArguments:

    results_logging_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    task_list: Optional[List[str]] = field(
        default=None, metadata={"help": "List of tasks passed in order."}
    )
