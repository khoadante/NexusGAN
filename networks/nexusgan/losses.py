import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor

from typing import List

__all__ = ["ContentLoss", "GANLoss"]


class ContentLoss(nn.Module):
    """Constructs a content loss function based on the VGG19 network.
    Using high-level feature mapping layers from the latter layers will focus more on the texture content of the image.

    Paper reference list:
        -`Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network <https://arxiv.org/pdf/1609.04802.pdf>` paper.
        -`ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks                    <https://arxiv.org/pdf/1809.00219.pdf>` paper.
        -`Perceptual Extreme Super Resolution Network with Receptive Field Block               <https://arxiv.org/pdf/2005.12597.pdf>` paper.

    """

    def __init__(
        self,
        feature_model_extractor_nodes: list,
        feature_model_normalize_mean: list,
        feature_model_normalize_std: list,
    ) -> None:
        super(ContentLoss, self).__init__()
        # Get the name of the specified feature extraction node
        self.feature_model_extractor_nodes = feature_model_extractor_nodes
        # Load the VGG19 model trained on the ImageNet dataset.
        model = models.vgg19(True)
        # Extract the thirty-sixth layer output in the VGG19 model as the content loss.
        self.feature_extractor = create_feature_extractor(
            model, feature_model_extractor_nodes
        )

        # set to validation mode
        self.feature_extractor.eval()

        # The preprocessing method of the input data.
        # This is the VGG model preprocessing method of the ImageNet dataset.
        self.normalize = transforms.Normalize(
            feature_model_normalize_mean, feature_model_normalize_std
        )

        # Freeze model parameters.
        for model_parameters in self.feature_extractor.parameters():
            model_parameters.requires_grad = False

    def forward(
        self, sr_tensor: torch.Tensor, hr_tensor: torch.Tensor
    ) -> List[torch.Tensor]:
        # Standardized operations
        sr_tensor = self.normalize(sr_tensor)
        hr_tensor = self.normalize(hr_tensor)

        sr_features = self.feature_extractor(sr_tensor)
        hr_features = self.feature_extractor(hr_tensor)

        # Find the feature map difference between the two images
        content_loss1 = F.l1_loss(
            sr_features[self.feature_model_extractor_nodes[0]],
            hr_features[self.feature_model_extractor_nodes[0]],
        )
        content_loss2 = F.l1_loss(
            sr_features[self.feature_model_extractor_nodes[1]],
            hr_features[self.feature_model_extractor_nodes[1]],
        )
        content_loss3 = F.l1_loss(
            sr_features[self.feature_model_extractor_nodes[2]],
            hr_features[self.feature_model_extractor_nodes[2]],
        )
        content_loss4 = F.l1_loss(
            sr_features[self.feature_model_extractor_nodes[3]],
            hr_features[self.feature_model_extractor_nodes[3]],
        )
        content_loss5 = F.l1_loss(
            sr_features[self.feature_model_extractor_nodes[4]],
            hr_features[self.feature_model_extractor_nodes[4]],
        )

        return content_loss1, content_loss2, content_loss3, content_loss4, content_loss5


class GANLoss(nn.Module):
    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer("real_label", torch.tensor(target_real_label))
        self.register_buffer("fake_label", torch.tensor(target_fake_label))

        self.loss = nn.BCEWithLogitsLoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)
