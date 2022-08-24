"""
 Copyright (c) 2019 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import os
import torch
import math

from nncf.torch.dynamic_graph.patch_pytorch import register_operator
from nncf.torch.functions import STThreshold

@register_operator()
def binary_mask_by_threshold(importance, threshold=0.5, sigmoid=int(os.environ.get('YUJIE_SIGMOID_THRESHOLD', '1')), max_percentile=0.98):
    if sigmoid:
        with torch.no_grad():
            max_threshold = torch.quantile(torch.sigmoid(importance), q=max_percentile).item()
        return STThreshold.apply(torch.sigmoid(importance), min(threshold, max_threshold))
    else:
        with torch.no_grad():
            max_threshold = torch.quantile(importance, q=max_percentile).item()
        return STThreshold.apply(importance, min(threshold, max_threshold))

# @register_operator()
# def binary_mask_by_threshold(importance, threshold=0.5, sigmoid=(os.environ.get('NNCF_THRESHOLD_SIGMOID', 'true').lower() == 'true'), max_percentile=0.98):
#     if sigmoid:
#         with torch.no_grad():
#             max_threshold = torch.quantile(torch.sigmoid(importance), q=max_percentile).item()
#         result_sig = STThreshold.apply(torch.sigmoid(importance), min(threshold, max_threshold))
#         with torch.no_grad():
#             max_threshold_nosig = torch.quantile(importance, q=max_percentile).item()
#         threshold_nosig = -2.197224577336219
#         result_nosig = STThreshold.apply(importance, min(threshold_nosig, max_threshold_nosig))
#         err = (result_sig - result_nosig).square().sum()
#         if err > 0:
#             print('!!!', err, result_nosig, result_sig, flush=True)
#         return result_nosig
#     else:
#         with torch.no_grad():
#             max_threshold = torch.quantile(importance, q=max_percentile).item()
#         return STThreshold.apply(importance, min(threshold, max_threshold))

# @register_operator()
# def binary_mask_by_threshold(importance, threshold=0.5, sigmoid=(os.environ.get('NNCF_THRESHOLD_SIGMOID', 'true').lower() == 'true'), max_percentile=0.98):
#     if int(os.environ.get('YUJIE_SIGMOID_THRESHOLD', '1')):
#         with torch.no_grad():
#             max_threshold = torch.quantile(torch.sigmoid(importance), q=max_percentile).item()
#         result_sig = STThreshold.apply(torch.sigmoid(importance), min(threshold, max_threshold))
#         return result_sig
#     else:
#         with torch.no_grad():
#             max_threshold_nosig = torch.quantile(importance, q=max_percentile).item()
#         threshold_nosig = -math.inf
#         if threshold > 0:
#             threshold_nosig =  math.log(threshold/ (1-threshold))
#         return STThreshold.apply(importance, min(threshold_nosig, max_threshold_nosig))