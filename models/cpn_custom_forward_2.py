import celldetection as cd
from celldetection.models.cpn import *


class CPN():
    def __init__(self, cuda, device, order=6, score_thresh=.9):
        cpn = cd.fetch_model('cyto_cpn_u22_i4_b6_o6_s2_boca_model', map_location=device)
#         if cuda:
#             cpn = cpn.cuda(device)
        cpn.eval()
        cpn.forward = lambda x: cpn_forward(cpn, x)
        cpn.score_thresh = score_thresh
        cpn.order = order
        self.cpn = cpn
        self.cuda = cuda
        self.device = device
        
        print(f"Loaded CPN Model with {self.count_parameters():,} parameters")
        cpn.requires_grad_(False)

    def count_parameters(self):
        return sum(p.numel() for p in self.cpn.parameters() if p.requires_grad)  
        
    def inference(self, img_batch):
        """img_batch is expected to be of shape [bs, channels, width, height] and normalized to [-1,1]."""
        # Asserts
        if self.cuda:
            img_batch = img_batch.cuda(self.device)
        return self.cpn(img_batch)


def cpn_forward(
        self,
        inputs,
        targets: Dict[str, Tensor] = None,
        nms=True
):
    # Presets
    original_size = inputs.shape[-2:]

    # Core
    scores, locations, refinement, fourier = self.core(inputs)

    # Scores
    raw_scores = scores
    if self.score_channels == 1:
        classes = torch.squeeze((scores > self.score_thresh).long(), 1)
    elif self.score_channels == 2:
        scores = F.softmax(scores, dim=1)[:, 1:2]
        classes = torch.squeeze((scores > self.score_thresh).long(), 1)
    elif self.score_channels > 2:
        scores = F.softmax(scores, dim=1)
        classes = torch.argmax(scores, dim=1).long()
    else:
        raise ValueError

    actual_size = fourier.shape[-2:]
    n, c, h, w = fourier.shape
    if self.functional:
        fourier = fourier.view((n, c // 2, 2, h, w))
    else:
        fourier = fourier.view((n, c // 4, 4, h, w))

    # Maybe apply changed order
    if self.order < self.core.order:
        fourier = fourier[:, :self.order]

    # Fetch sampling and labels
    if self.training:
        if targets is None:
            raise ValueError("In training mode, targets should be passed")
        sampling = targets.get('sampling')
        labels = targets['labels']
    else:
        sampling = None
        labels = classes.detach()
    labels = downsample_labels(labels[:, None], actual_size)[:, 0]

    locations = rel_location2abs_location(locations, cache=self._rel_location2abs_location_cache)

    # Extract proposals
    fg_mask = labels > 0
    b, y, x = torch.where(fg_mask)
    selected_fourier = fourier[b, :, :, y, x]  # Tensor[-1, order, 4]
    selected_locations = locations[b, :, y, x]  # Tensor[-1, 2]
    selected_classes = classes[b, y, x]

    if self.score_channels in (1, 2):
        selected_scores = scores[b, 0, y, x]  # Tensor[-1]
    elif self.score_channels > 2:
        selected_scores = scores[b, selected_classes, y, x]  # Tensor[-1]
    else:
        raise ValueError

    if sampling is not None:
        sampling = sampling[b]

    # Convert to pixel space
    selected_contour_proposals, sampling = fouriers2contours(selected_fourier, selected_locations,
                                                           samples=self.samples, sampling=sampling,
                                                           cache=self._fourier2contour_cache)

    # Rescale in case of multi-scale
    selected_contour_proposals = scale_contours(actual_size=actual_size, original_size=original_size,
                                                contours=selected_contour_proposals)
    selected_fourier, selected_locations = scale_fourier(actual_size=actual_size, original_size=original_size,
                                                         fourier=selected_fourier, location=selected_locations)

    if self.refinement and self.refinement_iterations > 0:
        det_indices = selected_contour_proposals  # Tensor[num_contours, samples, 2]
        num_loops = self.refinement_iterations
        if self.training and num_loops > 1:
            num_loops = torch.randint(low=1, high=num_loops + 1, size=())

        for _ in torch.arange(0, num_loops):
            det_indices = torch.round(det_indices.detach())
            det_indices[..., 0].clamp_(0, original_size[1] - 1)
            det_indices[..., 1].clamp_(0, original_size[0] - 1)
            indices = det_indices.detach().long()  # Tensor[-1, samples, 2]
            if self.core.refinement_buckets == 1:
                responses = refinement[b[:, None], :, indices[:, :, 1], indices[:, :, 0]]  # Tensor[-1, samples, 2]
            else:
                buckets = resolve_refinement_buckets(sampling, self.core.refinement_buckets)
                responses = None
                for bucket_indices, bucket_weights in buckets:
                    bckt_idx = torch.stack((bucket_indices * 2, bucket_indices * 2 + 1), -1)
                    cur_ref = refinement[b[:, None, None], bckt_idx, indices[:, :, 1, None], indices[:, :, 0, None]]
                    cur_ref = cur_ref * bucket_weights[..., None]
                    if responses is None:
                        responses = cur_ref
                    else:
                        responses = responses + cur_ref
            det_indices = det_indices + responses
        selected_contours = det_indices
    else:
        selected_contours = selected_contour_proposals

    # Bounding boxes
    if selected_contours.numel() > 0:
        selected_boxes = torch.cat((selected_contours.min(1).values,
                                    selected_contours.max(1).values), 1)  # 43.3 µs ± 290 ns for Tensor[2203, 32, 2]
    else:
        selected_boxes = torch.empty((0, 4), device=selected_contours.device)

    # Loss
    if self.training:
        loss, losses = self.compute_loss(
            fourier=selected_fourier,
            locations=selected_locations,
            contours=selected_contour_proposals,
            refined_contours=selected_contours,
            boxes=selected_boxes,
            raw_scores=raw_scores,
            targets=targets,
            labels=labels,
            fg_masks=fg_mask,
            b=b
        )
    else:
        loss, losses = None, None

    if self.training and not self.full_detail:
        return OrderedDict({
            'loss': loss,
            'losses': losses,
        })

    final_contours = []
    final_boxes = []
    final_scores = []
    final_classes = []
    final_locations = []
    final_fourier = []
    final_contour_proposals = []
    for batch_index in range(inputs.shape[0]):
        sel = b == batch_index
        final_contours.append(selected_contours[sel])
        final_boxes.append(selected_boxes[sel])
        final_scores.append(selected_scores[sel])
        final_classes.append(selected_classes[sel])
        final_locations.append(selected_locations[sel])
        final_fourier.append(selected_fourier[sel])
        final_contour_proposals.append(selected_contour_proposals[sel])

    if not self.training and nms:
        nms_r = batched_box_nms(
            final_boxes, final_scores, final_contours, final_locations, final_fourier, final_contour_proposals,
            final_classes,
            iou_threshold=self.nms_thresh
        )
        final_boxes, final_scores, final_contours, final_locations, final_fourier, final_contour_proposals, final_classes = nms_r

    # The dict below can be altered to return additional items of interest
    outputs = OrderedDict({
        'contours': final_contours,
        'boxes': final_boxes,
        'scores': final_scores,
        'classes': final_classes,
        'loss': loss,
        'losses': losses,
        'fourier': fourier,
        'final_fourier': final_fourier,
        'locations': locations,
        'sores_tensor': scores,
        'xy': final_locations
    })

    return outputs
