import numpy as np
import skimage
import matplotlib.pyplot as plt
import celldetection as cd

from torch.cuda.amp import autocast
from .ops import mask_cpn
from models.cpn_custom_forward_2 import CPN  # This only works when the main package is in the python path
from sklearn.decomposition import PCA


class Evaluate:
    """Evaluation class for Binary Inpainting Networks"""
    def __init__(self, model, forward, loader, cuda, device, order=6, score_thresh=.9):
        self.forward = forward  # function that executes the forward pass of the BI model
        self.model = model
        self.cuda = cuda
        self.device = device
        self.loader = loader
        self.cpn = CPN(cuda=cuda, device=device, order=order, score_thresh=score_thresh)
        self.order = order
    
    def _crop_to_masked_labels(self, crop, mask):
        """Helper function which takes in a batch of fully intact crops and a masks and returns
        a list of labels for the intact crop and the painting.
        crop/mask: [bs, 1, crop_dim, crop_dim]
        labels_true/labels_model: list of len bs with entries of size [crop_dim, crop_dim, d]"""
        cpn = self.cpn.inference(crop)
        cpn_masked, fourier_removed, locs_removed = mask_cpn(cpn, mask)

        labels_true = []
        for i, (fourier, locs) in enumerate(zip(fourier_removed, locs_removed)):
            contour = cd.data.cpn.fourier2contour(cd.asnumpy(fourier), cd.asnumpy(locs))
            labels_true.append(cd.data.contours2labels(contour, crop[0].shape[-2:]))

        labels_model, pop_idx = self.forward(self.model, cpn_masked, mask, self.cuda, self.device)
        for i in pop_idx:  # pop_idx is sorted backwards, so this works
            labels_true.pop(i)
            
        return labels_true, labels_model
                    
    @autocast()
    def cell_size_error_fourier(self, results=None, plot=False, save_as=None):
        if results is None:
            size_cells_true = []
            size_cells_model = []
            size_avg_true = []
            size_avg_model = []
            n_examples = 0

            for it, (crop, mask) in enumerate(self.loader):
                if self.cuda:
                    crop = crop.cuda(self.device)
                    mask = mask.cuda(self.device)
                # Get the outputs
                labels_true, labels_model = self._crop_to_masked_labels(crop, mask)
                n_examples += len(labels_true)

                for i, labels in enumerate(labels_true):
                    for l in range(1, labels.max()+1):
                        size_cells_true.append(labels[labels==l].sum()/l)
                    size_avg_true.append(np.mean(size_cells_true[-labels.max():]))

                for i, labels in enumerate(labels_model):
                    for l in range(1, labels.max()+1):
                        size_cells_model.append(labels[labels==l].sum()/l) 
                    size_avg_model.append(np.mean(size_cells_model[-labels.max():]))

            size_avg_true = np.array(size_avg_true)
            size_avg_model = np.array(size_avg_model)
            size_cells_true = np.array(size_cells_true)
            size_cells_model = np.array(size_cells_model)
            
            results = {"size_avg_true": size_avg_true, 
                       "size_avg_model": size_avg_model, 
                       "size_cells_true": size_cells_true, 
                       "size_cells_model": size_cells_model,
                       "n_examples": n_examples}
        else:
            size_cells_true = results["size_cells_true"]
            size_cells_model = results["size_cells_model"]
            size_avg_true = results["size_avg_true"]
            size_avg_model = results["size_avg_model"]
            n_examples = results["n_examples"]
            plot = True
                     
        if plot:
            plt.figure(figsize=(16,4))
            plt.subplots_adjust(wspace=0.3)
            plt.subplot(1,3,1)
            bin_size = 5
            max_size = 400
            bins = np.arange(max_size, step=bin_size)
            plt.title("Sgm. Cell Size Distribution", size=15)
            n_true, _, _ = plt.hist(size_cells_true, bins=bins)
            n_BG, _, _ = plt.hist(-size_cells_model, bins=-bins[::-1])
            plt.xlim([-400,400])
            plt.gca().set_xticklabels([int(np.abs(x)) for x in plt.gca().get_xticks()])
            plt.gca().set_yticklabels([np.round(y/n_examples/bin_size, 2) for y in plt.gca().get_yticks()])
            plt.legend(["Crop", "Painting"])
            plt.xlabel(f"Sgm. Cell Size (bin size {bin_size})", size=12)
            plt.ylabel("Average Occurence per Crop/Painting", size=12)
            
            plt.subplot(1,3,2)
            plt.bar(bins[:-1]/bin_size, (n_BG[::-1] - n_true)/n_examples/bin_size, label="Painting - Crop")
            plt.title("Sgm. Cell Size Difference", size=15)
            plt.xlabel(f"Sgm. Cell Size (bin size {bin_size})", size=12)
            plt.ylabel("Difference in Average Occurence", size=12)
            plt.xlim([0, 80])
            plt.gca().set_xticklabels([int(x*bin_size) for x in plt.gca().get_xticks()])
            plt.legend()
              
            plt.subplot(1,3,3)
            plt.title("Average Sgm. Cell Size", size=15)
            plt.boxplot([size_avg_true, size_avg_model])
            plt.xticks([1, 2], ["Crop", "Painting"], size=12)
            plt.ylim([0,300])
            if save_as is not None:
                plt.savefig("saves/"+save_as+"1.png", dpi=200, bbox_inches='tight')
            plt.show() 
            
            plt.figure(figsize=(5,4))
            plt.title(f"Difference in Avg. Sgm. Cell Size", size=15)
            tmp = np.abs((size_avg_model - size_avg_true)).max()
            tmp = np.round(tmp/bin_size)*bin_size  # Round to the nearest multiple of bin_size
            plt.hist((size_avg_model - size_avg_true), np.arange(-tmp, tmp, step=5), label="Painting - Real Crop ")
            plt.vlines(0, 0, plt.gca().get_yticks()[-1], color="black", linestyle=":")
            plt.xlabel("Difference (bin size 5)", size=12)
            plt.ylabel("No. of Occurences", size=12)
            plt.legend()                                   
            if save_as is not None:
                plt.savefig("saves/"+save_as+"2.png", dpi=200, bbox_inches='tight')                           
            plt.show()
            
            CSE = np.abs(size_avg_model - size_avg_true)
            out_msg = f"Binary Cell Size Error was {np.mean(CSE):.2f} ({np.std(CSE):.2f} std) with an average cell size of {np.mean(size_cells_model):.1f} ({np.mean(size_cells_true):.1f} in real) averaged over {n_examples} samples."
            print(out_msg)
           
        return results
    
    @autocast()
    def cell_eccentricity_error_fourier(self, results=None, plot=False, save_as=None):
        if results is None:
            # Lists with eccentricites for all cells
            ecc_cells_true = []
            ecc_cells_model = []
            ecc_avg_true = []
            ecc_avg_model = []
            n_examples = 0

            for it, (crop, mask) in enumerate(self.loader):
                if self.cuda:
                    crop = crop.cuda(self.device)
                    mask = mask.cuda(self.device)
                # Get the outputs
                labels_true, labels_model = self._crop_to_masked_labels(crop, mask)
                n_examples += len(labels_true)

                for i, labels in enumerate(labels_true):
                    length = len(ecc_cells_true)
                    d = labels.shape[-1]  # labels has shape [crop_dim, crop_dim, d] to account for cell overlap
                    for j in range(d):
                        props = skimage.measure.regionprops(labels[:,:,j])
                        for cell in props:
                            ecc_cells_true.append(cell.eccentricity)
                    if length == len(ecc_cells_true):
                        ecc_avg_true.append(0.75)  # Workaround for empty crops
                    else:
                        ecc_avg_true.append(np.mean(ecc_cells_true[-len(ecc_cells_true)+length:]))

                for i, labels in enumerate(labels_model):
                    length = len(ecc_cells_model)
                    d = labels.shape[-1]  # labels has shape [crop_dim, crop_dim, d] to account for cell overlap
                    for j in range(d):
                        props = skimage.measure.regionprops(labels[:,:,j])
                        for cell in props:
                            ecc_cells_model.append(cell.eccentricity)
                    if length == len(ecc_cells_model):
                        ecc_avg_model.append(0.75)  # Workaround for empty crops
                    else:
                        ecc_avg_model.append(np.mean(ecc_cells_model[-len(ecc_cells_model)+length:]))

            ecc_avg_true = np.array(ecc_avg_true)
            ecc_avg_model = np.array(ecc_avg_model)        
            ecc_cells_true = np.array(ecc_cells_true)
            ecc_cells_model = np.array(ecc_cells_model)
            
            results = {"ecc_avg_true": ecc_avg_true, 
                       "ecc_avg_model": ecc_avg_model, 
                       "ecc_cells_true": ecc_cells_true, 
                       "ecc_cells_model": ecc_cells_model,
                       "n_examples": n_examples}
        else:
            ecc_avg_true = results["ecc_avg_true"]
            ecc_avg_model = results["ecc_avg_model"]
            ecc_cells_true = results["ecc_cells_true"]
            ecc_cells_model = results["ecc_cells_model"]
            n_examples = results["n_examples"]
            plot = True
            
        if plot:
            plt.figure(figsize=(16,4))
            plt.subplots_adjust(wspace=0.3)
            plt.subplot(1,3,1)
            n_bins = 50
            bin_size = 1/50
            bins = np.linspace(0, 1, num=n_bins)
            plt.title("Sgm. Cell Eccentricity Distribution", size=15)
            n_true, _, _ = plt.hist(ecc_cells_true, bins=bins)
            n_BG, _, _ = plt.hist(-ecc_cells_model, bins=-bins[::-1])
            plt.gca().set_yticklabels([np.round(y/n_examples, 1) for y in plt.gca().get_yticks()])
            plt.gca().set_xticklabels([np.round(np.abs(x), 2) for x in plt.gca().get_xticks()])
            plt.legend(["Crop", "Painting"])
            plt.xlabel(f"Sgm. Cell Eccentricity (bin size {bin_size:.2f})", size=12)
            plt.ylabel("Average Occurence per Crop/Painting", size=12)
            
            plt.subplot(1,3,2)
            plt.bar(bins[:-1]/bin_size, (n_BG[::-1] - n_true)/n_examples, label="Painting - Crop")
            plt.title("Sgm. Cell Eccentricity Difference", size=15)
            plt.xlabel(f"Sgm. Cell Eccentricity (bin size {bin_size:.2f})", size=12)
            plt.ylabel("Difference in Average Eccentricity", size=12)
            plt.gca().set_xticklabels([x*bin_size for x in plt.gca().get_xticks()])
            plt.legend()
            
            plt.subplot(1,3,3)
            plt.title("Average Sgm. Cell Eccentricity", size=15)
            plt.boxplot([ecc_avg_true, ecc_avg_model])
            plt.xticks([1, 2], ["Crop", "Painting"], size=12)
            if save_as is not None:
                plt.savefig("saves/"+save_as+"1.png", dpi=200, bbox_inches='tight')
            plt.show()
            
            plt.figure(figsize=(5,4))
            plt.title(f"Difference in Avg. Sgm. Cell Eccentricity", size=15)
            tmp = np.abs((ecc_avg_model - ecc_avg_true)).max()
            plt.hist((ecc_avg_model - ecc_avg_true), bins=np.arange(-tmp, tmp+bin_size, step=bin_size), label="Painting - Real Crop ")
            plt.vlines(0, 0, plt.gca().get_yticks()[-1], color="black", linestyle=":")
            plt.xlabel(f"Difference (bin size ({bin_size:.2f})", size=12)
            plt.ylabel("No. of Occurences", size=12)
            plt.legend()
            if save_as is not None:
                plt.savefig("saves/"+save_as+"2.png", dpi=200, bbox_inches='tight')                                    
            plt.show()
            CEE = np.abs(ecc_avg_model - ecc_avg_true)
            out_msg = f"Binary Cell Eccentricity Error was {np.mean(CEE):.3f} ({np.std(CEE):.3f} std) with an average cell eccentricity of {np.mean(ecc_cells_model):.2f} ({np.mean(ecc_cells_true):.2f} in real) averaged over {n_examples} samples."
            print(out_msg)
            
        return results
         
    @autocast()
    def pca_fourier(self, n_components=2, results=None, plot=False, save_as=None):
        if results is None:
            n_examples = 0  # Counter for the number of crops
            n_cells_true = 0  # Counter of total number of cells
            n_cells_model = 0
            f_dim = 4*self.order
            matrix_true = np.zeros((0,f_dim))
            matrix_model = np.zeros((0,f_dim))

            for it, (crop, mask) in enumerate(self.loader):
                if self.cuda:
                    crop = crop.cuda(self.device)
                    mask = mask.cuda(self.device)
                # Get the outputs
                cpn = self.cpn.inference(crop)
                cpn_masked, fourier_true, _ = mask_cpn(cpn, mask)
                fourier_prd, pop_idx = self.forward(self.model, cpn_masked, mask, cuda=self.cuda, device=self.device, return_fourier=True)
                for i in pop_idx:  # pop_idx is sorted backwards, so this works
                    fourier_true.pop(i)

                n_examples += len(fourier_true)

                for i, fourier in enumerate(fourier_true):
                    n_cells_tmp = fourier.shape[0]
                    matrix_true = np.append(matrix_true, fourier.reshape(-1,f_dim).cpu().numpy(), axis=0)
                    n_cells_true += n_cells_tmp

                for i, fourier in enumerate(fourier_prd):
                    n_cells_tmp = fourier.shape[0]
                    matrix_model = np.append(matrix_model, fourier.reshape(-1,f_dim).cpu().numpy(), axis=0)
                    n_cells_model += n_cells_tmp

            pca_true = PCA(n_components)
            result_true = pca_true.fit_transform(matrix_true)

            result_model = pca_true.transform(matrix_model)
            results = {"result_true": result_true, "result_model": result_model}

        else:
            result_true = results["result_true"]
            result_model = results["result_model"]
            plot = True
        
        if plot:
            plt.figure(figsize=(8,5))
            plt.title("PCA Plot", size=15)
            plt.scatter(result_true[:,0], result_true[:,1], label="Crop")
            plt.scatter(result_model[:,0], result_model[:,1], label="Painting")
            plt.legend()
            plt.xlabel("First Principal Direction")
            plt.ylabel("Second Principal Direction")
            plt.ylim([-20, 20])
            plt.xlim([-20, 20])
            if save_as is not None:
                plt.savefig("saves/"+save_as+".png", dpi=200, bbox_inches='tight')
            plt.show()
        
        return results

    @autocast()
    def cell_count_error_fourier(self, full_out=False, plot=False, save_as=None):
        n_cells_true = []
        n_cells_model = []

        for it, (crop, mask) in enumerate(self.loader):
            if self.cuda:
                crop = crop.cuda(self.device)
                mask = mask.cuda(self.device)
            # Get the outputs
            labels_true, labels_model = self._crop_to_masked_labels(crop, mask)

            for i in range(len(labels_true)):
                n_cells_true.append(labels_true[i].max())
            for i in range(len(labels_model)):
                n_cells_model.append(labels_model[i].max())
                      
        n_cells_true = np.array(n_cells_true)
        n_cells_model = np.array(n_cells_model)
        
        CCE = np.abs((n_cells_model - n_cells_true))
        out_msg = f"Binary Cell Count Error was {CCE.mean():.2f} ({CCE.std():.2f} std) averaged over {len(n_cells_true)} samples with an average of {n_cells_model.mean():.1f} ({n_cells_true.mean():.1f}) cells per mask."
                
        if plot:
            plt.figure(figsize=(16,5))
            plt.suptitle(out_msg, size=14, multialignment="left")

            plt.subplot(1,2,1)
            plt.title(f"Cell Count Difference per Mask")
            tmp = np.abs((n_cells_model - n_cells_true)).max()
            plt.hist((n_cells_model - n_cells_true), np.arange(-tmp, tmp+1))
            plt.vlines(0, 0, plt.gca().get_yticks()[-2], color="black", linestyle=":")
            plt.xlabel("Difference (bin size 1)")
            plt.ylabel("No. of Occurences")
            plt.legend(["Painting - Real Crop "])

            plt.subplot(1,2,2)
            plt.title("Cell Count per Mask")
            plt.boxplot([n_cells_true, n_cells_model])
            plt.xticks([1, 2], ["Crop", "Painting"])

            if save_as is not None:
                plt.savefig(save_as, dpi=200)
            plt.show()
        
        if full_out:
            return CCE, n_cells_true, n_cells_model
