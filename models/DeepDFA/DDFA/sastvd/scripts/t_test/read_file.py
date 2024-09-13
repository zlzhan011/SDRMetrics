import pickle
import DDFA.sastvd as svd
import os
intermediate_result =  os.path.join(svd.processed_dir(), 'bigvul_intermediate_result')


def read_pkl():
    c_root = os.path.join(intermediate_result, 'graph')
    v_cnt = 0
    for file in os.listdir(c_root):
        if file.endswith('pkl'):
            with open(os.path.join(c_root, file), 'rb') as f:
                graph = pickle.load(f)

                _VULN = graph['g.ndata']['_VULN']

                _VULN_max = max(_VULN)
                if _VULN_max == 1:
                    v_cnt += 1

                if _VULN_max == 1:
                    print("\n\n\n********************************")
                    print(graph)


    print("v_cnt:", v_cnt)





if __name__ == '__main__':
    read_pkl()