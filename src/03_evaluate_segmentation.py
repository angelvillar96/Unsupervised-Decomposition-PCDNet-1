"""
Evaluating the decomposition model for instance segmentation
"""
import os
import json
from tqdm import tqdm
import numpy as np
import torch
import warnings

from lib.arguments import get_directory_argument
from lib.config import load_exp_config_file
from lib.logger import Logger, log_function, for_all_methods
import lib.metrics as metrics
import lib.setup_model as setup_model
import lib.utils as utils
import data
warnings.filterwarnings("ignore")


@for_all_methods(log_function)
class Evaluator:
    """
    Class for evaluating a decomposition model
    """

    def __init__(self, exp_path, checkpoint):
        """
        Initializing the evaluator object
        """

        utils.set_random_seed()
        self.exp_path = exp_path
        self.exp_params = load_exp_config_file(exp_path)

        checkpoint_name = checkpoint.split(".")[0]
        self.results_path = os.path.join(self.exp_path, "results")
        self.results_file = os.path.join(self.results_path, f"instance_seg_results_{checkpoint_name}.json")
        utils.create_directory(self.results_path)

        self.model_path = os.path.join(exp_path, "models", "PCDNet_models", checkpoint)
        return

    def load_data(self):
        """
        Loading dataset and fitting data-loader for iterating in a batch-like fashion
        """

        # loading dataset and data loaders
        utils.set_random_seed()

        self.test_set = data.load_data(exp_params=self.exp_params, split="test")
        return

    def setup_model(self):
        """
        Instanciating model and loading pretrained parameters
        """
        torch.backends.cudnn.fastest = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = setup_model.setup_model(
                model_params=self.exp_params["model"],
                dataset_name=self.exp_params["dataset"]["dataset_name"]
            )
        model = setup_model.load_checkpoint(self.model_path, model=model, only_model=True)
        self.model = model.to(self.device).eval()

        self.mse = metrics.MSE()
        return

    @torch.no_grad()
    def predict(self):
        """
        Predicting decomposed objects for the test set and saving the results
        on a json file for evaluation
        """

        mse_list = []
        ari_list = []
        ari_bkg_list = []
        preds = {}

        # for batch, (imgs, meta) in enumerate(tqdm(self.test_loader)):
        N = len(self.test_set)
        for i in tqdm(range(N)):
            imgs, meta = self.test_set.__getitem__(i, get_mask=True)
            C, H, W = imgs.shape
            imgs = imgs.to(self.device).view(1, C, H, W).float()
            reconstruction, (objects, object_ids, protos) = self.model(imgs)

            # computing segmentation masks and filling pixel-holes
            sel_masks = self.model.temp_masks[0, object_ids[0, :]-1]
            thr = 0.7
            processed_masks = sel_masks.clone()
            processed_masks[processed_masks < thr] = 0
            processed_masks[processed_masks >= thr] = 1
            processed_masks = metrics.fill_mask(processed_masks, thr=0.8)

            # reconstruction metric
            loss = self.mse(reconstruction, imgs)
            mse_list.append(loss.item())

            # computing segmentation metrics
            pred, gt = torch.zeros(35, 35), torch.zeros(35, 35)
            N_obj = len(meta["mask"])-1

            for i in range(N_obj):
                cur_gt_mask = torch.Tensor(meta["mask"][i+1, :, :, 0])
                cur_pred_mask = processed_masks[-i-1, 0].cpu()
                gt[cur_gt_mask > 0] = (i+1) * cur_gt_mask[cur_gt_mask > 0]
                pred[cur_pred_mask > 0] = (i+1) * cur_pred_mask[cur_pred_mask > 0]

            ari = metrics.segmentation_ari(pred=pred, gt=gt, use_bkg=False)
            ari_list.append(ari)
            ari_bkg = metrics.segmentation_ari(pred=pred, gt=gt, use_bkg=True)
            ari_bkg_list.append(ari_bkg)

        mean_mse = np.mean(mse_list)
        mean_ari = np.mean(ari_list)
        mean_ari_bkg = np.mean(ari_bkg_list)
        print(f"Mean Eval MSE: {round(mean_mse, 5)}")
        print(f"Mean Eval ARI: {round(mean_ari, 5)}")
        print(f"Mean Eval ARI Bkg: {round(mean_ari_bkg, 5)}")
        preds["test_set_mse"] = mean_mse
        preds["test_set_ari"] = mean_ari
        preds["test_set_ari_bkg"] = mean_ari_bkg
        preds["mses"] = mse_list
        preds["aris"] = ari_list
        preds["aris_bkgs"] = ari_bkg_list
        self.mean_mse = mean_mse
        self.mean_ari = mean_ari
        self.mean_ari_bkg = mean_ari_bkg

        # saving results into json file
        with open(self.results_file, "w") as f:
            json.dump(preds, f)

        return




if __name__ == "__main__":
    utils.clear_cmd()
    exp_path, checkpoint = get_directory_argument(get_checkpoint=True)
    logger = Logger(exp_path=exp_path)
    logger.log_info("Starting DecompNet evaluation procedure", message_type="new_exp")

    print("Initializing Evaluator...")
    evaluator = Evaluator(exp_path=exp_path, checkpoint=checkpoint)
    print("Loading dataset...")
    evaluator.load_data()
    print("Setting up model and optimizer")
    evaluator.setup_model()
    print("Decomposing the test set")
    evaluator.predict()


#
