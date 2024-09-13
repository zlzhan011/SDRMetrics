import os, sys
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
parent_parent_dir = os.path.dirname(parent_dir)
sys.path.append(current_dir)
sys.path.append(parent_dir)
sys.path.append(parent_parent_dir)


import argparse
import DDFA.sastvd as svd
import DDFA.sastvd.helpers.datasets as svdd
import DDFA.sastvd.helpers.evaluate as ivde


def bigvul():
    """Run preperation scripts for BigVul dataset."""
    print(svdd.bigvul(sample=args.sample))
    ivde.get_dep_add_lines_bigvul("bigvul", sample=args.sample)
    # svdglove.generate_glove("bigvul", sample=args.sample)
    # svdd2v.generate_d2v("bigvul", sample=args.sample)
    print("success")


def devign():
    raise NotImplementedError
    print(svdd.devign(sample=args.sample))
    ivde.get_dep_add_lines("devign", sample=args.sample)
    svdglove.generate_glove("devign", sample=args.sample)
    svdd2v.generate_d2v("devign", sample=args.sample)
    print("success")


def diversevul():
    # raise NotImplementedError
    print(svdd.diversevul(sample=args.sample))
    ivde.get_dep_add_lines_bigvul("bigvul", sample=args.sample)
    # svdglove.generate_glove("diversevul", sample=args.sample)
    # svdd2v.generate_d2v("diversevul", sample=args.sample)
    print("diversevul success")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare master dataframe")
    parser.add_argument("--sample", action="store_true", help="Extract a sample only")
    parser.add_argument("--global_workers", type=int, help="Number of workers to use")
    parser.add_argument("--dataset", default='bigvul', choices=['bigvul', 'devign', 'diversevul'])
    args = parser.parse_args()

    print("start prepare")
    if args.global_workers is not None:
        svd.DFMP_WORKERS = args.global_workers

    if args.dataset == "bigvul":
        bigvul()
        # diversevul()
    if args.dataset == "devign":
        devign()
    if args.dataset == "diversevul":
        diversevul()
