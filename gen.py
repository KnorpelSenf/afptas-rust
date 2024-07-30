from random import uniform
from argparse import ArgumentParser

def main():
    parser = ArgumentParser(
        prog='job generation util',
        description='generate very large random scheduling instances')

    parser.add_argument('-o', '--filename', type=str, default="-")
    parser.add_argument('-n', '--count', type=int, default=100)
    parser.add_argument('--min-processing-time', type=float, default=0.0)
    parser.add_argument('--max-processing-time', type=float, default=30.0)
    parser.add_argument('--min-resource-amount', type=float, default=0.0)
    parser.add_argument('--max-resource-amount', type=float, default=10.0)

    args = parser.parse_args()

    file = args.filename
    n = args.count
    min_p, max_p = args.min_processing_time, args.max_processing_time
    min_r, max_r = args.min_resource_amount, args.max_resource_amount

    header = "processing_time,resource_amount"
    lines = (str(uniform(min_p, max_p)) + "," +
             str(uniform(min_r, max_r))
             for _ in range(n))

    if file == "-":
        print(header)
        for line in lines:
            print(line)
    else:
        with open(file, "w") as f:
            f.write(header + "\n")
            f.writelines(line + "\n" for line in lines)


if __name__ == '__main__':
    main()
