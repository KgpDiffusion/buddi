import os
import numpy as np
import pickle
def merge_results(results_dir:str):
    """ Merge results from intercap and behave
        Args:
            results_dir: Directory containing results of different runs.
        
        Returns:
            merged_results: Merged results"""
    

    assert os.path.exists(results_dir), f"Results directory {results_dir} does not exist"

    # Load intercap pickle file
    icap_file = os.path.join(results_dir,'intercap_no_opti_results_flt16.pkl')
    behave_file = os.path.join(results_dir,'behave_no_opti_results_flt16.pkl')

    # Read pickle file
    res={}
    with open(icap_file, 'rb') as file:
        icap_data = pickle.load(file)

    res.update(icap_data)

    with open(behave_file, 'rb') as file:
        behave_data = pickle.load(file)

    res.update(behave_data)

    # store final results in pkl file
    with open(os.path.join('/home/shubhikg/exp/buddi', "results.pkl"), "wb") as f:
        pickle.dump(res, f)


if __name__ == '__main__':

    merge_results('/home/shubhikg/')