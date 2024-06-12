import os
import numpy as np
import pickle
import json


def verify_pkl_file(filepath:str):

    assert os.path.exists(filepath), f"Results directory {filepath} does not exist"

    # Read pickle file
    with open(filepath, 'rb') as file:
        data = pickle.load(file)

    print("Total num keys in pkl file: ", len(data))
    # for fileID, data in data.items():
    #     for key in data.keys():
    #         print(key, data[key])
            # assert isinstance(data[key], float16), f"Data type of {key} is not float16 and is {data[key].dtype}"

    


def jsons_to_pkl(results_dir:str, filename:str):
    """ Merge results from intercap and behave
        Args:
            results_dir: Directory containing results of different runs.
        
        Returns:
            merged_results: Merged results"""
    

    assert os.path.exists(results_dir), f"Results directory {results_dir} does not exist"

    # load all json files
    json_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
    # remove duplicate files
    json_files = list(set(json_files))
    print("Num json files: ", len(json_files))
    results = {}
    for file in json_files:
        with open(os.path.join(results_dir, file), 'r') as f:
            file_data = json.load(f)

            ## Convert all fields value to np.float16
            res = {}
            for fileID, data in file_data.items():
                for key in data.keys():
                    # print(key, data[key])
                    data[key] = np.array(data[key]).astype(np.float16)

                res[fileID] = data
            
            results.update(res)

    # dump thing in pkl file
    print("total keys in pkl file: ", len(results))
    with open(os.path.join(filename), "wb") as f:
        pickle.dump(results, f)


if __name__ == '__main__':

    results_dir = '/home/user/Abhinav/buddi/demo/training/behave_obj_condn_final/vis_res/results/jsons'
    filename = '/home/user/Abhinav/buddi/demo/training/behave_obj_condn_final/vis_res/results/behave_opti_results_flt16.pkl'
    pkl_file_path = '/home/user/Abhinav/buddi/results.pkl'
    pkl_file_path = '/home/user/Abhinav/buddi/demo/training/behave_obj_condn_final/vis_res/results/behave_opti_results_flt16.pkl'

    jsons_to_pkl(results_dir=results_dir, filename=filename)
    verify_pkl_file(pkl_file_path)