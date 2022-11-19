import argparse
import numpy as np

import pocket

def name_parser(name):
    """
    {INDEX}.png
    """
    seg = name.split(".")

    return "Dataset index: {}".format(seg[0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate HTML table")
    parser.add_argument("--image-dir",
                        required=True,
                        type=str)

    args = parser.parse_args()

    table = pocket.utils.ImageHTMLTable(
        4, args.image_dir,
        parser=name_parser,
        width="75%"
    )

    table()
