import numpy as np
import torch
import skimage
import matplotlib.pyplot as plt
import celldetection as cd

from torch.cuda.amp import autocast
from utils.ops import convert_to_uint8, get_noise, mask_cpn, compute_kde
from models.cpn_custom_forward_2 import CPN  # This only works when the main package is in the python path
from models.density_fill_forward import DensityInpainting
from PIL import Image
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
from torchvision.transforms.functional import resize


class Evaluate:
    """Evaluation class for Density Inpainting Networks"""
    def __init__(self, loader, cuda, device, order=6, score_thresh=.9):
        self.cuda = cuda
        self.device = device
        self.loader = loader
        self.cpn = CPN(cuda, device, order, score_thresh)
        self.DI = DensityInpainting(cuda, device)
        self.order = order
    
    @autocast()
    def cell_count_error_fourier(self, results=None, plot=False, save_as=None):
        if results is None:
            bs = self.loader.batch_size
            n_examples = len(self.loader)*bs
            n_cells_true = []
            n_cells_model = []
            n_cells_true_all = []

            for it, (crop, mask) in enumerate(self.loader):
                crop_dim = crop.shape[2]
                if self.cuda:
                    crop = crop.cuda(self.device)
                    mask = mask.cuda(self.device)

                # Get the outputs
                cpn_out = self.cpn.inference(crop)
                xy_true = cpn_out["xy"]
                cpn_masked, _, locs_removed = mask_cpn(cpn_out, mask)

                xy_batch, locs_in_mask = self.DI.fill_locations(cpn_masked, mask)
                mask = mask.squeeze(1).cpu().numpy()

                for locs in locs_in_mask:
                    if len(locs) > 0:
                        n_cells_model.append(locs.shape[0])
                    else:
                        n_cells_model.append(0)

                for locs in locs_removed:
                    if len(locs) > 0:
                        n_cells_true.append(locs.shape[0])
                    else:
                        n_cells_true.append(0)

                for locs in xy_true:
                    n_cells_true_all.append(locs.shape[0])

            n_cells_true = np.array(n_cells_true)
            n_cells_model = np.array(n_cells_model)
            n_cells_true_all = np.array(n_cells_true_all)
            
            results = {"n_cells_true": n_cells_true, 
                       "n_cells_model": n_cells_model, 
                       "n_cells_true_all": n_cells_true_all}
        else:
            n_cells_true = results["n_cells_true"]
            n_cells_model = results["n_cells_model"]
            n_cells_true_all = results["n_cells_true_all"]
            plot = True
        
        if plot:
            plt.figure(figsize=(16,4))
            plt.subplots_adjust(wspace=0.3)
            plt.subplot(1,3,1)
            bin_size = 2
            max_size = max(max(n_cells_model), max(n_cells_true))
            max_size = np.round(max_size/bin_size)*bin_size  # Round to the nearest multiple of bin_size
            bins = np.arange(max_size, step=bin_size)
            plt.title("Sgm. Cell Count Distribution", size=15)
            n_true, _, _ = plt.hist(n_cells_true, bins=bins)
            n_BG, _, _ = plt.hist(-n_cells_model, bins=-bins[::-1])
            plt.xlim([-max_size,max_size])
            plt.gca().set_xticklabels([int(np.abs(x)) for x in plt.gca().get_xticks()])
            plt.legend(["Crop", "Painting"])
            plt.xlabel(f"Sgm. Cell Count (bin size {bin_size})", size=12)
            plt.ylabel("No. of Occurences", size=12)
            
            plt.subplot(1,3,2)
            plt.bar(bins[:-1]/bin_size, (n_BG[::-1] - n_true), label="Painting - Crop")
            plt.title("Sgm. Cell Count Difference", size=15)
            plt.xlabel(f"Sgm. Cell Count (bin size {bin_size})", size=12)
            plt.ylabel("Difference in No. of Occurences", size=12)
            plt.xlim([-0.5, max_size/bin_size])
            plt.gca().set_xticklabels([int(x*bin_size) for x in plt.gca().get_xticks()])
            plt.ylim([-110,310])
            plt.legend()
            
            plt.subplot(1,3,3)
            plt.title("Average Sgm. Cell Count", size=15)
            plt.boxplot([n_cells_true, n_cells_model])
            plt.xticks([1, 2], ["Crop", "Painting"], size=12)          
            
            if save_as is not None:
                plt.savefig("saves/"+save_as+"1.png", dpi=200, bbox_inches='tight')
            plt.show()
            
            plt.figure(figsize=(5,4))
            plt.title(f"Difference in Avg. Sgm. Cell Count", size=15)
            plt.hist((n_cells_model - n_cells_true), np.arange(-max_size, max_size, step=bin_size), label="Painting - Real Crop ")
            plt.vlines(0, 0, plt.gca().get_yticks()[-1], color="black", linestyle=":")
            plt.xlabel(f"Difference (bin size {bin_size})", size=12)
            plt.ylabel("No. of Occurences", size=12)
            plt.legend()

            if save_as is not None:
                plt.savefig("saves/"+save_as+"2.png", dpi=200, bbox_inches='tight')
            plt.show()           
                
            CCE = np.abs((n_cells_model - n_cells_true))  # Account for the "-" from above
            out_msg = f"Average Binary Cell Count Error was {CCE.mean():.2f} ({CCE.std():.2f} std) averaged over {len(n_cells_true)} samples with an average of {n_cells_model.mean():.1f} ({n_cells_true.mean():.1f} in crop) cells per mask. The intact crops contained on average {n_cells_true_all.mean():.1f} cells in total."
            print(out_msg)
        
        return results
    
    @autocast()
    def cell_kde_error_fourier(self, results=None, plot=False, save_as=None):
        if results is None:
            bs = self.loader.batch_size
            n_examples = len(self.loader)*bs
            downscale = 4
            kde_sum_true = []
            kde_sum_model = []

            for it, (crop, mask) in enumerate(self.loader):
                crop_dim = mask.shape[2]
                if self.cuda:
                    crop = crop.cuda(self.device)
                    mask = mask.cuda(self.device)

                # Get the outputs
                mask_small = resize(mask, (crop_dim//downscale, crop_dim//downscale))
                cpn_out = self.cpn.inference(crop)
                xy_true = cpn_out["xy"].copy()  # list of tensors of shape [n_cells, 2]
                kde_true = torch.as_tensor(compute_kde(cd.asnumpy(xy_true), crop_dim, downscale=downscale),
                                      dtype=torch.half)  # [bs, 1, crop_dim/downscale, crop_dim/downscale]
                for i in range(kde_true.shape[0]):
                    kde_sum_true.append(kde_true[i][mask_small[i]].sum().item())

                cpn_masked = mask_cpn(cpn_out, mask)[0]
                xy_batch = cpn_masked["xy"].copy()  # list of tensors of shape [n_cells, 2]
                kde = torch.as_tensor(compute_kde(cd.asnumpy(xy_batch), crop_dim, downscale=downscale),
                                      dtype=torch.half)  # [bs, 1, crop_dim/downscale, crop_dim/downscale]
                if self.cuda:
                    kde = kde.cuda(self.device)
                kde = (kde - 0.02635) * 18  # Normalize approximately to [-1,1] and 0 mean
                DG_out = self.DI.DG(kde, condition=mask_small.float(), tanh=True)  # Account for network normalization
                DG_out = DG_out / 18 + 0.02635  # Revert the normalization to restore correct probability mass

                for i in range(kde.shape[0]):
                    kde_sum_model.append(DG_out[i][mask_small[i]].sum().item())

            kde_sum_true = np.array(kde_sum_true)
            kde_sum_model = np.array(kde_sum_model)
            
            results = {"kde_sum_true": kde_sum_true, 
                       "kde_sum_model": kde_sum_model}
        else:
            kde_sum_true = results["kde_sum_true"]
            kde_sum_model = results["kde_sum_model"]
            plot = True
                
        if plot:
            plt.figure(figsize=(16,4))
            plt.subplots_adjust(wspace=0.3)            
            plt.subplot(1,3,1)
            bin_size = 2
            max_size = max(max(kde_sum_model), max(kde_sum_true))
            max_size = np.round(max_size/bin_size)*bin_size  # Round to the nearest multiple of bin_size
            bins = np.arange(max_size, step=bin_size)
            plt.title("Probability Sum Distribution", size=15)
            n_true, _, _ = plt.hist(kde_sum_true, bins=bins)
            n_BG, _, _ = plt.hist(-kde_sum_model, bins=-bins[::-1])
            plt.xlim([-max_size,max_size])
            plt.gca().set_xticklabels([int(np.abs(x)) for x in plt.gca().get_xticks()])
            plt.legend(["Crop", "Painting"])
            plt.xlabel(f"Probability Sum (bin size {bin_size})", size=12)
            plt.ylabel("No. of Occurences", size=12)
           
            plt.subplot(1,3,2)
            plt.bar(bins[:-1]/bin_size, (n_BG[::-1] - n_true), label="Painting - Crop")
            plt.title("Probability Sum Difference", size=15)
            plt.xlabel(f"Probability Sum (bin size {bin_size})", size=12)
            plt.ylabel("Difference in No. of Occurences", size=12)
            plt.xlim([-0.5, max_size/bin_size])
            plt.gca().set_xticklabels([int(x*bin_size) for x in plt.gca().get_xticks()])
            plt.ylim([-110,310])
            plt.legend()
             
            plt.subplot(1,3,3)
            plt.title("Average Probability Sum", size=15)
            plt.boxplot([kde_sum_true, kde_sum_model])
            plt.xticks([1, 2], ["Crop", "Painting"], size=12)
            if save_as is not None:
                plt.savefig("saves/"+save_as+"1.png", dpi=200, bbox_inches='tight')
            plt.show()
            
            plt.figure(figsize=(5,4))
            plt.title(f"Difference in Avg. Probability Sum", size=15)
            plt.hist((kde_sum_model - kde_sum_true), np.arange(-max_size, max_size, step=bin_size), label="Painting - Real Crop ")
            plt.vlines(0, 0, plt.gca().get_yticks()[-1], color="black", linestyle=":")
            plt.xlabel(f"Difference (bin size {bin_size})", size=12)
            plt.ylabel("No. of Occurences", size=12)
            plt.legend()    
            if save_as is not None:
                plt.savefig("saves/"+save_as+"2.png", dpi=200, bbox_inches='tight')
            plt.show()           
                
            CKE = np.abs((kde_sum_model - kde_sum_true))
            out_msg = f"Average Binary KDE Count Error was {CKE.mean():.2f} ({CKE.std():.2f} std) averaged over {len(kde_sum_true)} samples with an average of {kde_sum_model.mean():.1f} ({kde_sum_true.mean():.1f} in crop) probability mass per mask."
            print(out_msg)
        
        return results
