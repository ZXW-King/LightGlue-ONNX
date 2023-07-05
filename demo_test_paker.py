from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import load_image, match_pair
from lightglue import viz2d
from pathlib import Path
import torch
import matplotlib.pyplot as plt

device=torch.device('cuda')

extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)  # load the extractor

match_conf = {
    'width_confidence': 0.99,  # for point pruning
    'depth_confidence': 0.95,  # for early stopping,
}
matcher = LightGlue(pretrained='superpoint', **match_conf).eval().to(device)

images = Path('assets')
image0, scales0 = load_image('/media/xin/work1/github_pro/LightGlue-ONNX/indemind/L/1614044733261054_L.png', resize=1024, grayscale=False)

image1, scales1 = load_image('/media/xin/work1/github_pro/LightGlue-ONNX/indemind/R/1614044733261054_R.png', resize=1024, grayscale=False)

pred = match_pair(extractor, matcher, image0.to(device), image1.to(device))

kpts0, kpts1, matches = pred['keypoints0'], pred['keypoints1'], pred['matches']
m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

axes = viz2d.plot_images([image0.permute(1, 2, 0), image1.permute(1, 2, 0)])
viz2d.plot_matches(m_kpts0.cpu(), m_kpts1.cpu(), color='lime', lw=0.2)
viz2d.add_text(0, f'Stop after {pred["stop"]} layers', fs=20)

kpc0, kpc1 = viz2d.cm_prune(pred['prune0'].cpu()), viz2d.cm_prune(pred['prune1'].cpu())
viz2d.plot_images([image0.permute(1, 2, 0).cpu(), image1.permute(1, 2, 0).cpu()])
viz2d.plot_keypoints([kpts0.cpu(), kpts1.cpu()], colors=[kpc0, kpc1], ps=10)
plt.show()