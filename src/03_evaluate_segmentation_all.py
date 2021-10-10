"""
Evaluating all checkpoints from one experiment
"""

import os
import json
import numpy as np
import warnings

from lib.arguments import get_directory_argument
import lib.utils as utils

warnings.filterwarnings("ignore")
Evaluator = __import__('03_evaluate_segmentation')


def main():
    """
    Iteratively evaluating each model checkpoint in an experiment
    """

    exp_path, _ = get_directory_argument()
    print(exp_path)
    models_path = os.path.join(exp_path, "models", "PCDNet_models")
    models = sorted(os.listdir(models_path))
    if(len(models) == 0):
        print(f"No models found in {models_path}")
        return
    else:
        print(f"Found {len(models)} model checkpoints")

    results_path = os.path.join(exp_path, "results")
    results_file = os.path.join(results_path, "best_results.json")

    good_models = []
    mses = []
    aris = []
    aris_bkgs = []
    for model in models:
        try:
            print(f"Processing checkpoint: {model}...")
            evaluator = Evaluator.Evaluator(exp_path=exp_path, checkpoint=model)
            evaluator.load_data()
            evaluator.setup_model()
            evaluator.predict()
            mses.append(evaluator.mean_mse)
            aris.append(evaluator.mean_ari)
            aris_bkgs.append(evaluator.mean_ari_bkg)
            good_models.append(model)
        except Exception as e:
            print(e)
            print(f"  There was an error processing model {model}...")

    print(mses)
    best_mse = np.argmin(mses)
    best_ari = np.argmax(aris)
    best_ari_bkg = np.argmax(aris_bkgs)

    print("")
    print(f"Best MSE: {mses[best_mse]}   , obtained by model {good_models[best_mse]}")
    print(f"Best ARI: {aris[best_ari]}   , obtained by model {good_models[best_ari]}")
    print(f"Best ARI-BKG: {aris_bkgs[best_ari_bkg]}   , obtained by model {good_models[best_ari_bkg]}")

    # saving results into json file
    preds = {
        "best_mse": {
            "result": mses[best_mse],
            "model": good_models[best_mse]
        },
        "best_ari": {
            "result": aris[best_ari],
            "model": good_models[best_ari]
        },
        "best_ari_bkg": {
            "result": aris_bkgs[best_ari_bkg],
            "model": good_models[best_ari_bkg]
        }
    }
    with open(results_file, "w") as f:
        json.dump(preds, f)

    return



if __name__ == "__main__":
    utils.clear_cmd()
    main()
