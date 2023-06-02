import torch
from torch.nn import functional as F
import numpy as np
from collections import OrderedDict
from . import UnitCAM
from ..utils.gradient_extraction import upsample
from ..utils.training import train_model
from ..utils.datasets import DatasetLoader
from ..utils.model_helpers import SwapLastDims, Squeeze

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class GradCAM(UnitCAM):
    """The implementation of Grad-CAM for multivariate time series classification
    CNN-based deep learning models

    Based on the paper:

        Selvaraju, R. R., Cogswell, M.,
        Das, A., Vedantam, R., Parikh,
        D., & Batra, D. (2017). Grad-cam: Visual explanations from deep networks
        via gradient-based localization. In Proceedings of the
        IEEE international conference on computer vision (pp. 618-626).

    Implementation adapted from:

        https://github.com/jacobgil/pytorch-grad-cam/blob/bf27469f5b3accf9535e04e52106e3f77f5e9cf5/gradcam.py#L31


    This implementation is modified to only support Multivariate Time Series
    Classification data and the corresponding CNN-based models

    """

    def __init__(self, model, feature_module, target_layer_names, use_cuda):
        super().__init__(model, feature_module, target_layer_names, use_cuda)
        self.grads_val = None
        self.target = None

    def calculate_gradients(self, input_features, print_out, index):
        """Implemented method when CAM is called on a given input and its targeted
        index

        Attributes:
        -------
            input_features: A multivariate data input to the model
            print_out: Whether to print the class with maximum likelihood when index is None
            index: Targeted output class

        """
        features, output, index = self.extract_features(
            input_features, print_out, index
        )
        self.feature_module.zero_grad()
        self.model.zero_grad()

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        one_hot.backward(retain_graph=True)

        self.grads_val = self.extractor.get_gradients()[-1].cpu().data

        self.target = features[-1]
        self.target = self.target.cpu().data.numpy()[0, :]

        return output

    def map_gradients(self):
        """Caculate weights based on the gradients corresponding to the extracting layer
        via global average pooling

        Returns:
        -------
            cam: The placeholder for resulting weighted feature maps
            weights: The weights corresponding to the extracting feature maps
        """
        if len(self.grads_val.shape) == 4:
            weights = np.mean(self.grads_val.numpy(), axis=(2, 3))[0, :]
        elif len(self.grads_val.shape) == 3:
            weights = np.mean(self.grads_val.numpy(), axis=2).reshape(
                -1, self.grads_val.size(0)
            )

        cam = np.zeros(self.target.shape[1:], dtype=np.float32)

        return cam, weights

    def __call__(self, input_features, print_out, index=None):
        """Implemented method when CAM is called on a given input and its targeted
        index

        Attributes:
        -------
            input_features: A multivariate data input to the model
            print_out: Whether to print the class with maximum likelihood when index is None
            index: Targeted output class

        Returns:
        -------
            cam: The resulting weighted feature maps
        """
        if index is not None and print_out == True:
            print_out = False

        output = self.calculate_gradients(input_features, print_out, index)

        cam, weights = self.map_gradients()
        assert (
            weights.shape[0] == self.target.shape[0]
        ), "Weights and targets layer shapes are not compatible."
        cam = self.cam_weighted_sum(cam, weights, self.target)

        return cam, output


class GradCAMPlusPlus(GradCAM):
    """The implementation of Grad-CAM++ for multivariate time series classification
    CNN-based deep learning models

    Based on the paper:

        Chattopadhay, A., Sarkar, A., Howlader, P., & Balasubramanian, V. N.
        (2018, March). Grad-cam++: Generalized gradient-based visual explanations
        for deep convolutional networks.
        In 2018 IEEE Winter Conference on Applications of Computer Vision (WACV)
        (pp. 839-847). IEEE.

    Implementation adapted from:

        https://github.com/adityac94/Grad_CAM_plus_plus/blob/4a9faf6ac61ef0c56e19b88d8560b81cd62c5017/misc/utils.py#L51


    This implementation is modified to only support Multivariate Time Series
    Classification data and the corresponding CNN-based models

    """

    def __init__(self, model, feature_module, target_layer_names, use_cuda):
        super().__init__(model, feature_module, target_layer_names, use_cuda)
        self.alphas = None
        self.one_hot = None

    @staticmethod
    def compute_second_derivative(one_hot, target):
        """Second Derivative

        Attributes:
        -------
            one_hot: Targeted index output
            target: Targeted module output

        Returns:
        -------
            second_derivative: The second derivative of the output

        """
        second_derivative = torch.exp(one_hot.detach().cpu()) * target

        return second_derivative

    @staticmethod
    def compute_third_derivative(one_hot, target):
        """Third Derivative

        Attributes:
        -------
            one_hot: Targeted index output
            target: Targeted module output

        Returns:
        -------
            third_derivative: The third derivative of the output

        """
        third_derivative = torch.exp(one_hot.detach().cpu()) * target * target

        return third_derivative

    @staticmethod
    def compute_global_sum(one_hot):
        """Global Sum

        Attributes:
        -------
            one_hot: Targeted index output

        Returns:
        -------
            global_sum: Collapsed sum from the input

        """

        global_sum = np.sum(one_hot.detach().cpu().numpy(), axis=0)

        return global_sum

    def extract_higher_level_gradient(
        self, global_sum, second_derivative, third_derivative
    ):
        """Alpha calculation

        Calculate alpha based on high derivatives and global sum

        Attributes:
        -------
            global_sum: Collapsed sum from the input
            second_derivative: The second derivative of the output
            third_derivative: The third derivative of the output

        """
        alpha_num = second_derivative.numpy()
        alpha_denom = (
            second_derivative.numpy() * 2.0 + third_derivative.numpy() * global_sum
        )
        alpha_denom = np.where(
            alpha_denom != 0.0, alpha_denom, np.ones(alpha_denom.shape)
        )
        self.alphas = alpha_num / alpha_denom

    def calculate_gradients(self, input_features, print_out, index):
        """Implemented method when CAM is called on a given input and its targeted
        index

        Attributes:
        -------
            input_features: A multivariate data input to the model
            print_out: Whether to print the class with maximum likelihood when index is None
            index: Targeted output class

        """
        features, output, index = self.extract_features(
            input_features, print_out, index
        )
        self.feature_module.zero_grad()
        self.model.zero_grad()

        self.one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        self.one_hot[0][index] = 1
        self.one_hot = torch.from_numpy(self.one_hot).requires_grad_(True)
        if self.cuda:
            self.one_hot = torch.sum(self.one_hot.cuda() * output)
        else:
            self.one_hot = torch.sum(self.one_hot * output)

        self.one_hot.backward(retain_graph=True)

        self.grads_val = self.extractor.get_gradients()[-1].cpu().data

        self.target = features[-1]
        self.target = self.target.cpu().data.numpy()[0, :]

        return output

    def map_gradients(self):
        """Caculate weights based on the gradients corresponding to the extracting layer
        via global average pooling

        Returns:
        -------
            cam: The placeholder for resulting weighted feature maps
            weights: The weights corresponding to the extracting feature maps
        """
        if len(self.grads_val.shape) == 4:
            weights = np.sum(F.relu(self.grads_val).numpy() * self.alphas, axis=(2, 3))[
                0, :
            ]
        elif len(self.grads_val.shape) == 3:
            weights = np.sum(
                F.relu(self.grads_val).numpy() * self.alphas, axis=2
            ).reshape(-1, self.grads_val.size(0))
        cam = np.zeros(self.target.shape[1:], dtype=np.float32)

        return cam, weights

    def __call__(self, input_features, print_out, index=None):
        """Implemented method when CAM is called on a given input and its targeted
        index

        Attributes:
        -------
            input_features: A multivariate data input to the model
            print_out: Whether to print the class with maximum likelihood when index is None
            index: Targeted output class

        Returns:
        -------
            cam: The resulting weighted feature maps
        """
        if index is not None and print_out == True:
            print_out = False

        output = self.calculate_gradients(input_features, print_out, index)
        second_derivative = self.compute_second_derivative(self.one_hot, self.target)
        third_derivative = self.compute_third_derivative(self.one_hot, self.target)
        global_sum = self.compute_global_sum(self.one_hot)
        self.extract_higher_level_gradient(
            global_sum, second_derivative, third_derivative
        )
        cam, weights = self.map_gradients()
        assert (
            weights.shape[0] == self.target.shape[0]
        ), "Weights and targets layer shapes are not compatible."
        cam = self.cam_weighted_sum(cam, weights, self.target)

        return cam, output
    

class XGradCAM(GradCAM):
    """The implementation of XGrad-CAM for multivariate time series classification
    CNN-based deep learning models

    Based on the paper:

        Fu, R., Hu, Q., Dong, X., Guo, Y., Gao, Y., & Li, B. (2020). Axiom-based
        grad-cam: Towards accurate visualization and explanation of cnns.
        arXiv preprint arXiv:2008.02312.

    Implementation adapted from:

        https://github.com/Fu0511/XGrad-CAM/blob/main/XGrad-CAM.py


    This implementation is modified to only support Multivariate Time Series
    Classification data and the corresponding CNN-based models

    """

    def map_gradients(self):
        """Caculate weights based on the gradients corresponding to the extracting layer
        via global average pooling

        Returns:
        -------
            cam: The placeholder for resulting weighted feature maps
            weights: The weights corresponding to the extracting feature maps
        """
        if len(self.grads_val.shape) == 4:
            weights = np.sum(self.grads_val.numpy()[0, :] * self.target, axis=(1, 2))
            weights = weights / (np.sum(self.target, axis=(1, 2)) + 1e-6)
        elif len(self.grads_val.shape) == 3:
            weights = np.sum(self.grads_val.numpy()[0, :] * self.target, axis=1)
            weights = weights / (np.sum(self.target, axis=(0, 1)) + 1e-6)
        cam = np.zeros(self.target.shape[1:], dtype=np.float32)

        return cam, weights

    def __call__(self, input_features, print_out, index=None):
        """Implemented method when CAM is called on a given input and its targeted
        index

        Attributes:
        -------
            input_features: A multivariate data input to the model
            index: Targeted output class

        Returns:
        -------
            cam: The resulting weighted feature maps
        """
        output = self.calculate_gradients(input_features, print_out, index)

        cam, weights = self.map_gradients()
        assert (
            weights.shape[0] == self.target.shape[0]
        ), "Weights and targets layer shapes are not compatible."
        cam = self.cam_weighted_sum(cam, weights, self.target)

        return cam, output


class SmoothGradCAMPlusPlus(GradCAMPlusPlus):
    """The implementation of Smooth Grad-CAM++ for multivariate time series classification
    CNN-based deep learning models

    Based on the paper:

        Omeiza, D., Speakman, S., Cintas, C., & Weldermariam, K. (2019).
        Smooth grad-cam++: An enhanced inference level visualization technique for
        deep convolutional neural network models. arXiv preprint arXiv:1908.01224.

    Implementation adapted from:

        https://github.com/frgfm/torch-cam/blob/master/torchcam/cams/gradcam.py#L164

    This implementation is modified to only support Multivariate Time Series
    Classification data and the corresponding CNN-based models

    """

    def __init__(self, model, feature_module, target_layer_names, use_cuda, **kwargs):
        super().__init__(model, feature_module, target_layer_names, use_cuda)
        self.smooth_factor = kwargs["smooth_factor"]
        self.std = kwargs["std"]
        self._distrib = torch.distributions.normal.Normal(0, self.std)
        self.device = device

    def __call__(self, input_features, print_out, index=None):
        """Implemented method when CAM is called on a given input and its targeted
        index

        Attributes:
        -------
            input_features: A multivariate data input to the model
            print_out: Whether to print the class with maximum likelihood when index is None
            index: Targeted output class

        Returns:
        -------
            cam: The resulting weighted feature maps
        """

        if index is not None and print_out == True:
            print_out = False

        grads_vals = None
        second_derivatives = None
        third_derivatives = None
        for _ in range(self.smooth_factor):
            output = self.calculate_gradients(
                input_features
                + self._distrib.sample(input_features.size()).to(self.device),
                print_out,
                index,
            )
            second_derivative = self.compute_second_derivative(
                self.one_hot, self.target
            )
            third_derivative = self.compute_third_derivative(self.one_hot, self.target)
            if (
                grads_vals is None
                or second_derivatives is None
                or third_derivatives is None
            ):
                grads_vals = self.grads_val
                second_derivatives = second_derivative
                third_derivatives = third_derivative
            else:
                grads_vals += self.grads_val
                second_derivatives += second_derivative
                third_derivatives += third_derivative

            second_derivatives = F.relu(second_derivatives)
            second_derivatives_min, second_derivatives_max = (
                second_derivatives.min(),
                second_derivatives.max(),
            )
            if second_derivatives_min == second_derivatives_max:
                return None
            second_derivatives = (
                (second_derivatives - second_derivatives_min)
                .div(second_derivatives_min - second_derivatives_max)
                .data
            )

            third_derivatives = F.relu(third_derivatives)
            third_derivatives_min, third_derivatives_max = (
                third_derivatives.min(),
                third_derivatives.max(),
            )
            if third_derivatives_min == third_derivatives_max:
                return None
            third_derivatives = (
                (third_derivatives - third_derivatives_min)
                .div(third_derivatives_min - third_derivatives_max)
                .data
            )

        output = self.calculate_gradients(input_features, print_out, index)
        global_sum = self.compute_global_sum(self.one_hot)

        self.extract_higher_level_gradient(
            global_sum,
            second_derivatives.div_(self.smooth_factor),
            third_derivatives.div_(self.smooth_factor),
        )
        self.grads_val = grads_vals.div(self.smooth_factor)

        cam, weights = self.map_gradients()
        assert (
            weights.shape[0] == self.target.shape[0]
        ), "Weights and targets layer shapes are not compatible."
        cam = self.cam_weighted_sum(cam, weights, self.target)

        return cam, output

class ScoreCAM(UnitCAM):
    """The implementation of Score-CAM for multivariate time series classification
    CNN-based deep learning models

    Based on the paper:

        Wang, H., Wang, Z., Du, M., Yang, F., Zhang, Z., Ding, S., ... & Hu, X. (2020).
        Score-CAM: Score-weighted visual explanations for convolutional neural networks.
        In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
        Recognition Workshops (pp. 24-25).

    Implementation adapted from:

        https://github.com/haofanwang/Score-CAM/blob/master/cam/scorecam.py

    This implementation is modified to only support Multivariate Time Series
    Classification data and the corresponding CNN-based models

    """

    def __init__(self, model, feature_module, target_layer_names, use_cuda):
        super().__init__(model, feature_module, target_layer_names, use_cuda)
        self.target = None

    def forward_saliency_map(self, input_features, print_out, index):
        """Do forward pass through the network

        Attributes:
        -------
            input_features: A multivariate data input to the model
            print_out: Whether to print the class with maximum likelihood when index is None
            index: Targeted output class

        Returns:
        -------
            activation: The feature maps
            score_saliency_map: The placeholder for the resulting saliency map
            k: The number of channels in the feature maps
            index: The targeted index
            output: The network forward pass output
        """
        _, _, h, w = input_features.size()

        features, output, index = self.extract_features(
            input_features, print_out, index
        )

        self.feature_module.zero_grad()
        self.model.zero_grad()

        activations = features[-1]
        if len(activations.size()) == 4:
            _, k, _, _ = activations.size()
            score_saliency_map = torch.zeros((1, 1, h, w))
        elif len(activations.size()) == 3:
            _, k, _ = activations.size()
            score_saliency_map = torch.zeros((1, 1, h, 1))

        if torch.cuda.is_available():
            activations = activations.cuda()
            score_saliency_map = score_saliency_map.cuda()

        return activations, score_saliency_map, k, index, output

    def compute_score_saliency_map(self, input_features, print_out, index):
        """Compute the score saliency map

        Attributes:
        -------
            input_features: A multivariate data input to the model
            print_out: Whether to print the class with maximum likelihood when index is None
            index: Targeted output class

        Returns:
        -------
            cam: The resulting weighted feature maps
            scores: Corresponding scores to the feature maps
        """
        activations, score_saliency_map, k, index, output = self.forward_saliency_map(
            input_features, print_out, index
        )
        self.target = activations[-1]
        scores = 0

        with torch.no_grad():
            score_saliency_maps = []
            for i in range(k):
                # upsampling
                if len(self.target.size()) == 3:
                    saliency_map = torch.unsqueeze(self.target[i : i + 1, :, :], 0)
                elif len(self.target.size()) == 2:
                    saliency_map = torch.unsqueeze(
                        torch.unsqueeze(self.target[i : i + 1, :], 2), 0
                    )
                if saliency_map.max() != saliency_map.min():
                    # normalize to 0-1
                    norm_saliency_map = (saliency_map - saliency_map.min()) / (
                        saliency_map.max() - saliency_map.min()
                    )
                else:
                    norm_saliency_map = saliency_map
                if input_features.shape[:-1] == norm_saliency_map.shape[:-1]:
                    score_saliency_maps.append(input_features * norm_saliency_map)
                else:
                    norm_saliency_map = (
                        torch.from_numpy(
                            upsample(
                                norm_saliency_map.squeeze().cpu().numpy(),
                                input_features.squeeze().cpu().numpy().T,
                            )
                        )
                        .unsqueeze(0)
                        .unsqueeze(0)
                    ).to(device)
                    assert input_features.shape[:-1] == norm_saliency_map.shape[:-1]
                    score_saliency_maps.append(input_features * norm_saliency_map)

            # how much increase if keeping the highlighted region
            # predication on masked input
            masked_input_features = torch.squeeze(
                torch.stack(score_saliency_maps, dim=1), 0
            )
            output_ = self.model(masked_input_features)

            scores = output_[:, index] - output[0, index]
            cam = np.zeros(self.target.shape[1:], dtype=np.float32)

        return cam, scores, output

    def __call__(self, input_features, print_out, index=None):
        """Implemented method when CAM is called on a given input and its targeted
        index

        Attributes:
        -------
            input_features: A multivariate data input to the model
            print_out: Whether to print the class with maximum likelihood when index is None
            index: Targeted output class

        Returns:
        -------
            cam: The resulting weighted feature maps
        """
        if index is not None and print_out == True:
            print_out = False

        cam, scores, output = self.compute_score_saliency_map(
            input_features, print_out, index
        )

        assert (
            scores.shape[0] == self.target.shape[0]
        ), "Weights and targets layer shapes are not compatible."
        cam = self.cam_weighted_sum(
            cam, scores.detach().cpu().numpy(), self.target.detach().cpu().numpy()
        )

        return cam, output


class IntegratedScoreCAM(ScoreCAM):
    """The implementation of Integrated Score-CAM for multivariate time series classification
    CNN-based deep learning models

    Based on the paper:

        Naidu, R., Ghosh, A., Maurya, Y., & Kundu, S. S. (2020).
        IS-CAM: Integrated Score-CAM for axiomatic-based explanations.
        arXiv preprint arXiv:2010.03023.

    Implementation adapted from:

        https://github.com/frgfm/torch-cam/blob/master/torchcam/cams/cam.py#L291

    This implementation is modified to only support Multivariate Time Series
    Classification data and the corresponding CNN-based models

    """

    def __init__(self, model, feature_module, target_layer_names, use_cuda, **kwargs):
        super().__init__(model, feature_module, target_layer_names, use_cuda)
        self.smooth_factor = kwargs["smooth_factor"]

    def compute_score_saliency_map(self, input_features, print_out, index):
        """Compute the score saliency map

        Attributes:
        -------
            input_features: A multivariate data input to the model
            print_out: Whether to print the class with maximum likelihood when index is None
            index: Targeted output class

        Returns:
        -------
            cam: The resulting weighted feature maps
            scores: Corresponding scores to the feature maps
        """
        activations, score_saliency_map, k, index, output = self.forward_saliency_map(
            input_features, print_out, index
        )
        self.target = activations[-1]

        with torch.no_grad():
            scores = 0
            for idx in range(self.smooth_factor):
                score_saliency_maps = []
                for i in range(k):
                    # upsampling
                    if len(self.target.size()) == 3:
                        saliency_map = torch.unsqueeze(self.target[i : i + 1, :, :], 0)
                    elif len(self.target.size()) == 2:
                        saliency_map = torch.unsqueeze(
                            torch.unsqueeze(self.target[i : i + 1, :], 2), 0
                        )

                    if saliency_map.max() != saliency_map.min():
                        # normalize to 0-1
                        norm_saliency_map = (saliency_map - saliency_map.min()) / (
                            saliency_map.max() - saliency_map.min()
                        )
                    else:
                        norm_saliency_map = saliency_map
                    if input_features.shape[:-1] == norm_saliency_map.shape[:-1]:
                        score_saliency_maps.append(
                            ((idx + 1) / self.smooth_factor)
                            * input_features
                            * norm_saliency_map
                        )
                    else:
                        norm_saliency_map = (
                            torch.from_numpy(
                                upsample(
                                    norm_saliency_map.squeeze().cpu().numpy(),
                                    input_features.squeeze().cpu().numpy().T,
                                )
                            )
                            .unsqueeze(0)
                            .unsqueeze(0)
                        ).to(device)
                        assert input_features.shape[:-1] == norm_saliency_map.shape[:-1]
                        score_saliency_maps.append(
                            ((idx + 1) / self.smooth_factor)
                            * input_features
                            * norm_saliency_map
                        )
                # how much increase if keeping the highlighted region
                # predication on masked input
                masked_input_features = torch.squeeze(
                    torch.stack(score_saliency_maps, dim=1), 0
                )
                output_ = self.model(masked_input_features)

                scores = output_[:, index] - output[0, index]

            scores.div_(self.smooth_factor)
            cam = np.zeros(self.target.shape[1:], dtype=np.float32)

        return cam, scores, output


class InputSmoothScoreCAM(ScoreCAM):
    """The implementation of Input Smooth Score-CAM for multivariate time series classification
    CNN-based deep learning models

    Based on the paper:

        Naidu, R., & Michael, J. (2020). SS-CAM: Smoothed Score-CAM for
        sharper visual feature localization. arXiv preprint arXiv:2006.14255.

    Implementation adapted from:

        https://github.com/frgfm/torch-cam/blob/master/torchcam/cams/cam.py#L179

    This implementation is modified to only support Multivariate Time Series
    Classification data and the corresponding CNN-based models

    """

    def __init__(self, model, feature_module, target_layer_names, use_cuda, **kwargs):
        super().__init__(model, feature_module, target_layer_names, use_cuda)
        self.smooth_factor = kwargs["smooth_factor"]
        self.std = kwargs["std"]
        self._distrib = torch.distributions.normal.Normal(0, self.std)

    def compute_score_saliency_map(self, input_features, print_out, index):
        """Compute the score saliency map

        Attributes:
        -------
            input_features: A multivariate data input to the model
            print_out: Whether to print the class with maximum likelihood when index is None
            index: Targeted output class

        Returns:
        -------
            cam: The resulting weighted feature maps
            scores: Corresponding scores to the feature maps
        """
        activations, score_saliency_map, k, index, output = self.forward_saliency_map(
            input_features, print_out, index
        )
        self.target = activations[-1]

        with torch.no_grad():
            scores = 0
            for _ in range(self.smooth_factor):
                score_saliency_maps = []
                for i in range(k):
                    # upsampling
                    if len(self.target.size()) == 3:
                        saliency_map = torch.unsqueeze(self.target[i : i + 1, :, :], 0)
                    elif len(self.target.size()) == 2:
                        saliency_map = torch.unsqueeze(
                            torch.unsqueeze(self.target[i : i + 1, :], 2), 0
                        )

                    if saliency_map.max() != saliency_map.min():
                        # normalize to 0-1
                        norm_saliency_map = (saliency_map - saliency_map.min()) / (
                            saliency_map.max() - saliency_map.min()
                        )
                    else:
                        norm_saliency_map = saliency_map
                    if input_features.shape[:-1] == norm_saliency_map.shape[:-1]:
                        score_saliency_maps.append(
                            (
                                input_features
                                + self._distrib.sample(input_features.size()).to(device)
                            )
                            * norm_saliency_map
                        )
                    else:
                        norm_saliency_map = (
                            torch.from_numpy(
                                upsample(
                                    norm_saliency_map.squeeze().cpu().numpy(),
                                    input_features.squeeze().cpu().numpy().T,
                                )
                            )
                            .unsqueeze(0)
                            .unsqueeze(0)
                        ).to(device)
                        assert input_features.shape[:-1] == norm_saliency_map.shape[:-1]
                        score_saliency_maps.append(
                            (
                                input_features
                                + self._distrib.sample(input_features.size()).to(device)
                            )
                            * norm_saliency_map
                        )

                # how much increase if keeping the highlighted region
                # predication on masked input
                masked_input_features = torch.squeeze(
                    torch.stack(score_saliency_maps, dim=1), 0
                )
                output_ = self.model(masked_input_features)

                scores = output_[:, index] - output[0, index]

            scores.div_(self.smooth_factor)
            cam = np.zeros(self.target.shape[1:], dtype=np.float32)

        return cam, scores, output


class CAM(UnitCAM):
    """The implementation of CAM for multivariate time series classification
    CNN-based deep learning models

    Attributes:
    -------
        model: The wanna-be explained deep learning model for
            multivariate time series classification
        feature_module: The wanna-be explained module group (e.g. linear_layers)
        target_layer_names: The wanna-be explained module
        use_cuda: Whether to use cuda
        has_gap: True if the model has GAP layer right after
            the being explained CNN layer

    :NOTE:
    -------
    CAM can only applied with models that have Global Average Pooling
    layer. If no Global Average Pooling layer exists, one has to be added
    and the model has to be retrained over. Please state whether your model
    has a Global Average Pooling layer right after the being explained CNN
    layer by setting "has_gap = True" at class initiation.

    Based on the paper:
    -------

        Zhou, B., Khosla, A., Lapedriza,
        A., Oliva, A., & Torralba, A. (2016).
        Learning deep features for discriminative localization.
        In Proceedings of the IEEE conference on computer vision
        and pattern recognition (pp. 2921-2929).

    Implementation adapted from:
    -------

        https://github.com/zhoubolei/CAM/blob/master/pytorch_CAM.py


    This implementation is modified to only support Multivariate Time Series
    Classification data and the corresponding CNN-based models.

    """

    def __init__(self, model, feature_module, target_layer_names, use_cuda, **kwargs):
        super().__init__(model, feature_module, target_layer_names, use_cuda)
        self.has_gap = kwargs["has_gap"]

    def __call__(self, input_features, print_out, index=None, dataset_path=None):
        """Implemented method when CAM is called on a given input and its targeted
        index

        Attributes:
        -------
            input_features: A multivariate data input to the model
            print_out: Whether to print the class with maximum likelihood when index is None
            index: Targeted output class
            dataset_path: Path of the dataset (the same one that has been used to train)
                to retrain the new model (if it does not have GAP right after the explaining conv)

        Returns:
        -------
            cam: The resulting weighted feature maps
        """

        if index is not None and print_out == True:
            print_out = False

        if not self.has_gap:
            if dataset_path is None:
                raise AttributeError(
                    "Dataset path is not defined for retraining the new model"
                )

            for param in self.model.parameters():
                param.requires_grad = False

            if (
                "fc"
                not in list(
                    dict(self.model._modules["linear_layers"].named_children()).keys()
                )[-1]
            ):
                n_classes = self.model._modules["linear_layers"][-2].out_features
            else:
                n_classes = self.model._modules["linear_layers"][-1].out_features

            new_cnn_layer_list = []
            for idx, layer in enumerate(self.feature_module):
                new_cnn_layer_list.append(
                    (
                        list(dict(self.feature_module.named_children()).keys())[idx],
                        layer,
                    )
                )
                if (
                    list(dict(self.feature_module.named_children()).keys())[idx]
                    == self.target_layer_names[0]
                ):
                    out_channels = layer.out_channels
                    break

            new_cnn_layers = OrderedDict(new_cnn_layer_list)

            class TargetedModel(torch.nn.Module):
                def __init__(self, n_classes, out_channels):
                    super().__init__()
                    self.cnn_layers = torch.nn.Sequential(new_cnn_layers)

                    self.linear_layers_1d = torch.nn.Sequential(
                        OrderedDict(
                            [
                                ("avg_pool", torch.nn.AdaptiveAvgPool1d(1)),
                                ("view", SwapLastDims()),
                                ("fc1", torch.nn.Linear(out_channels, n_classes)),
                                ("softmax", torch.nn.Softmax(dim=1)),
                            ]
                        )
                    )

                    self.linear_layers_2d = torch.nn.Sequential(
                        OrderedDict(
                            [
                                ("avg_pool", torch.nn.AdaptiveAvgPool2d(1)),
                                ("squeeze", Squeeze()),
                                ("fc1", torch.nn.Linear(out_channels, n_classes)),
                                ("softmax", torch.nn.Softmax(dim=1)),
                            ]
                        )
                    )

                def forward(self, x):
                    x = self.cnn_layers(x)
                    if len(x.size()) == 4:
                        x = self.linear_layers_2d(x)
                    else:
                        x = self.linear_layers_1d(x)
                    x = torch.squeeze(x)

                    return x

            new_model = TargetedModel(n_classes, out_channels).to(device)

            for param in new_model._modules["linear_layers_1d"].parameters():
                param.requires_grad = True

            for param in new_model._modules["linear_layers_2d"].parameters():
                param.requires_grad = True

            dataset = DatasetLoader(dataset_path)
            dataloaders, datasets_size = dataset.get_torch_dataset_loader_auto(4, 4)

            criterion = torch.nn.CrossEntropyLoss()
            optimizer_ft = torch.optim.Adam(new_model.parameters(), lr=1.5e-4)
            exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer_ft, step_size=10, gamma=0.1
            )

            train_model(
                new_model,
                criterion,
                optimizer_ft,
                exp_lr_scheduler,
                dataloaders,
                datasets_size,
                10,
            )

            features, output, index = self.extract_features(
                input_features, print_out, index
            )

            target = features[-1]
            target = target.cpu().data.numpy()[0, :]

            try:
                print(
                    new_model._modules["linear_layers_1d"][-1]
                    .weight.detach()
                    .cpu()
                    .numpy()
                    .shape
                )
                weights = (
                    new_model._modules["linear_layers_1d"][-1]
                    .weight.detach()
                    .cpu()
                    .numpy()[index, :]
                )
            except AttributeError:
                print(
                    new_model._modules["linear_layers_1d"][-2]
                    .weight.detach()
                    .cpu()
                    .numpy()
                    .shape
                )
                weights = (
                    new_model._modules["linear_layers_1d"][-2]
                    .weight.detach()
                    .cpu()
                    .numpy()[index, :]
                )
            except KeyError:
                try:
                    print(
                        new_model._modules["linear_layers_2d"][-1]
                        .weight.detach()
                        .cpu()
                        .numpy()
                        .shape
                    )
                    weights = (
                        new_model._modules["linear_layers_2d"][-1]
                        .weight.detach()
                        .cpu()
                        .numpy()[index, :]
                    )
                except AttributeError:
                    print(
                        new_model._modules["linear_layers_2d"][-2]
                        .weight.detach()
                        .cpu()
                        .numpy()
                        .shape
                    )
                    weights = (
                        new_model._modules["linear_layers_2d"][-2]
                        .weight.detach()
                        .numpy()[index, :]
                    )

            cam = np.zeros(target.shape[1:], dtype=np.float32)
            target = np.squeeze(target)
            weights = np.squeeze(weights).T

            # assert (
            #     weights.shape[0] == target.shape[0]
            # ), "Weights and targets layer shapes are not compatible."
            cam = self.cam_weighted_sum(cam, weights, target, ReLU=False)

            return cam, output

        features, output, index = self.extract_features(
            input_features, print_out, index
        )

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        try:
            weights = (
                new_model._modules["linear_layers"][-1]
                .weight.detach()
                .cpu()
                .numpy()[:, index]
            )
        except AttributeError:
            weights = (
                new_model._modules["linear_layers"][-2]
                .weight.detach()
                .cpu()
                .numpy()[:, index]
            )

        cam = np.zeros(target.shape[1:], dtype=np.float32)
        target = np.squeeze(target)
        weights = np.squeeze(weights).T

        assert (
            weights.shape[0] == target.shape[0]
        ), "Weights and targets layer shapes are not compatible."
        cam = self.cam_weighted_sum(cam, weights, target, ReLU=False)

        return cam, output


class ActivationSmoothScoreCAM(ScoreCAM):
    """The implementation of Activation Smooth Score-CAM for multivariate time series classification
    CNN-based deep learning models

    Based on the paper:

        Naidu, R., & Michael, J. (2020). SS-CAM: Smoothed Score-CAM for
        sharper visual feature localization. arXiv preprint arXiv:2006.14255.

    Implementation adapted from:

        https://github.com/frgfm/torch-cam/blob/master/torchcam/cams/cam.py#L179

    This implementation is modified to only support Multivariate Time Series
    Classification data and the corresponding CNN-based models

    """

    def __init__(self, model, feature_module, target_layer_names, use_cuda, **kwargs):
        super().__init__(model, feature_module, target_layer_names, use_cuda)
        self.smooth_factor = kwargs["smooth_factor"]
        self.std = kwargs["std"]
        self._distrib = torch.distributions.normal.Normal(0, self.std)

    def compute_score_saliency_map(self, input_features, print_out, index):
        """Compute the score saliency map

        Attributes:
        -------
            input_features: A multivariate data input to the model
            print_out: Whether to print the class with maximum likelihood when index is None
            index: Targeted output class

        Returns:
        -------
            cam: The resulting weighted feature maps
            scores: Corresponding scores to the feature maps
        """
        activations, score_saliency_map, k, index, output = self.forward_saliency_map(
            input_features, print_out, index
        )
        self.target = activations[-1]

        with torch.no_grad():
            scores = 0
            for _ in range(self.smooth_factor):
                score_saliency_maps = []
                for i in range(k):
                    # upsampling
                    if len(self.target.size()) == 3:
                        saliency_map = torch.unsqueeze(self.target[i : i + 1, :, :], 0)
                    elif len(self.target.size()) == 2:
                        saliency_map = torch.unsqueeze(
                            torch.unsqueeze(self.target[i : i + 1, :], 2), 0
                        )

                    if saliency_map.max() != saliency_map.min():
                        # normalize to 0-1
                        norm_saliency_map = (saliency_map - saliency_map.min()) / (
                            saliency_map.max() - saliency_map.min()
                        )
                    else:
                        norm_saliency_map = saliency_map
                    if input_features.shape[:-1] == norm_saliency_map.shape[:-1]:
                        score_saliency_maps.append(
                            input_features
                            * (
                                norm_saliency_map
                                + self._distrib.sample(input_features.size()).to(device)
                            )
                        )
                    else:
                        norm_saliency_map = (
                            torch.from_numpy(
                                upsample(
                                    norm_saliency_map.squeeze().cpu().numpy(),
                                    input_features.squeeze().cpu().numpy().T,
                                )
                            )
                            .unsqueeze(0)
                            .unsqueeze(0)
                        ).to(device)
                        assert input_features.shape[:-1] == norm_saliency_map.shape[:-1]
                        score_saliency_maps.append(
                            input_features
                            * (
                                norm_saliency_map
                                + self._distrib.sample(input_features.size()).to(device)
                            )
                        )

                # how much increase if keeping the highlighted region
                # predication on masked input
                masked_input_features = torch.squeeze(
                    torch.stack(score_saliency_maps, dim=1), 0
                )
                output_ = self.model(masked_input_features)

                scores = output_[:, index] - output[0, index]

            scores.div_(self.smooth_factor)
            cam = np.zeros(self.target.shape[1:], dtype=np.float32)

        return cam, scores, output


class AblationCAM(UnitCAM):
    """The implementation of Ablation-CAM for multivariate time series classification
    CNN-based deep learning models

    Based on the paper:

        Ramaswamy, H. G. (2020). Ablation-cam: Visual explanations for deep
        convolutional network via gradient-free localization.
        In Proceedings of the IEEE/CVF Winter Conference on
        Applications of Computer Vision (pp. 983-991).


    This implementation is modified to only support Multivariate Time Series
    Classification data and the corresponding CNN-based models

    """

    def __init__(self, model, feature_module, target_layer_names, use_cuda):
        super().__init__(model, feature_module, target_layer_names, use_cuda)
        self.target_layer_names = target_layer_names
        self.slope = []
        self.target = None

    def calculate_slope(self, input_features, print_out, index):
        """Implemented method when CAM is called on a given input and its targeted
        index to calculate the slope between the feature maps and their ablation counter part

        Attributes:
        -------
            input_features: A multivariate data input to the model
            print_out: Whether to print the class with maximum likelihood when index is None
            index: Targeted output class

        """
        if index is None:
            features, output, index = self.extract_features(
                input_features, index, print_out
            )
        else:
            features, output, _ = self.extract_features(
                input_features, index, print_out
            )

        self.feature_module.zero_grad()
        self.model.zero_grad()

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output[0])
        else:
            one_hot = torch.sum(one_hot * output[0])

        self.target = features[-1]
        if len(self.target.size()) == 4:
            _, k, _, _ = self.target.size()
        elif len(self.target.size()) == 3:
            _, k, _ = self.target.size()
        self.target = self.target.cpu().data.numpy()[0, :]

        for i in range(k):
            _, output_k, _ = self.extract_features(
                input_features, int(index), print_out, zero_out=i
            )
            one_hot_k = np.zeros((1, output_k.size()[-1]), dtype=np.float32)
            one_hot_k[0][int(index)] = 1
            one_hot_k = torch.from_numpy(one_hot_k).requires_grad_(True)
            if self.cuda:
                one_hot_k = torch.sum(one_hot_k.cuda() * output_k[0])
            else:
                one_hot_k = torch.sum(one_hot_k * output_k[0])

            self.slope.append((one_hot - one_hot_k) / (one_hot + 1e-9))

        return output

    def map_slopes(self):
        """Caculate weights based on the gradients corresponding to the extracting layer
        via global average pooling

        Returns:
        -------
            cam: The placeholder for resulting weighted feature maps
            weights: The weights corresponding to the extracting feature maps
        """
        weights = torch.stack(self.slope)
        self.slope = []

        cam = np.zeros(self.target.shape[1:], dtype=np.float32)

        return cam, weights.detach().cpu().numpy()

    def __call__(self, input_features, print_out, index=None):
        """Implemented method when CAM is called on a given input and its targeted
        index

        Attributes:
        -------
            input_features: A multivariate data input to the model
            print_out: Whether to print the class with maximum likelihood when index is None
            index: Targeted output class

        Returns:
        -------
            cam: The resulting weighted feature maps
        """
        if index is not None and print_out == True:
            print_out = False

        output = self.calculate_slope(input_features, print_out, index)

        cam, weights = self.map_slopes()
        assert (
            weights.shape[0] == self.target.shape[0]
        ), "Weights and targets layer shapes are not compatible."
        cam = self.cam_weighted_sum(cam, weights, self.target)

        return cam, output
