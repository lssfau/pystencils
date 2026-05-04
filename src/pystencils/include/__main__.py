import argparse


def main():
    parser = argparse.ArgumentParser(
        "Print the include path for the pystencils runtime headers"
    )
    parser.add_argument(
        "-I", dest="dash_i", action="store_true", help="Emit as `-I` compiler argument"
    )
    parser.add_argument(
        "-s", dest="strip", action="store_true", help="Emit without trailing newline"
    )
    args = parser.parse_args()

    from . import get_pystencils_include_path

    include_path = get_pystencils_include_path()

    if args.dash_i:
        print(f"-I{include_path}", end="")
    else:
        end = "" if args.strip else "\n"
        print(get_pystencils_include_path(), end=end)


main()
