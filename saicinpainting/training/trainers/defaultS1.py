import logging

import torch
from omegaconf import OmegaConf

from saicinpainting.training.losses.feature_matching import feature_matching_loss, masked_l1_loss
from saicinpainting.training.trainers.baseS1 import BaseInpaintingTrainingModule
from saicinpainting.utils import add_prefix_to_keys, get_ramp

LOGGER = logging.getLogger(__name__)


class DefaultInpaintingTrainingModule(BaseInpaintingTrainingModule):
    def __init__(self, *args, concat_mask=True, image_to_discriminator='predicted_image',
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.concat_mask = concat_mask
        self.image_to_discriminator = image_to_discriminator

    def forward(self, batch):

        img = batch['image']
        mask = batch['mask']
        ref_img = batch['ref_image']
        masked_img = img * (1 - mask)

        # print(mask.shape, masked_img.shape)
        if self.concat_mask:
            masked_img = torch.cat([masked_img, mask], dim=1)

        batch['predicted_image'] = self.generator(masked_img, ref_img)
        batch['inpainted'] = mask * batch['predicted_image'] + (1 - mask) * batch['image']
        batch['mask_for_losses'] = mask

        return batch

    def generator_loss(self, batch):
        img = batch['image']
        predicted_img = batch[self.image_to_discriminator]
        original_mask = batch['mask']
        supervised_mask = batch['mask_for_losses']

        # L1
        l1_value = masked_l1_loss(predicted_img, img, supervised_mask,
                                  self.config.losses.l1.weight_known,
                                  self.config.losses.l1.weight_missing)

        total_loss = l1_value
        metrics = dict(gen_l1=l1_value)

        # discriminator
        # adversarial_loss calls backward by itself
        mask_for_discr = original_mask
        self.adversarial_loss.pre_generator_step(real_batch=img, fake_batch=predicted_img,
                                                 generator=self.generator, discriminator=self.discriminator)
        discr_real_pred, discr_real_features = self.discriminator(img)
        discr_fake_pred, discr_fake_features = self.discriminator(predicted_img)
        adv_gen_loss, adv_metrics = self.adversarial_loss.generator_loss(real_batch=img,
                                                                         fake_batch=predicted_img,
                                                                         discr_real_pred=discr_real_pred,
                                                                         discr_fake_pred=discr_fake_pred,
                                                                         mask=mask_for_discr)
        total_loss = total_loss + adv_gen_loss
        metrics['gen_adv'] = adv_gen_loss
        metrics.update(add_prefix_to_keys(adv_metrics, 'adv_'))

        # feature matching
        # 计算判别器对于真值以及预测数据提取特征的MSE损失
        if self.config.losses.feature_matching.weight > 0:
            need_mask_in_fm = OmegaConf.to_container(self.config.losses.feature_matching).get('pass_mask', False)
            mask_for_fm = supervised_mask if need_mask_in_fm else None
            fm_value = feature_matching_loss(discr_fake_features, discr_real_features,
                                             mask=mask_for_fm) * self.config.losses.feature_matching.weight
            total_loss = total_loss + fm_value
            metrics['gen_fm'] = fm_value
        
        if self.loss_pl is not None:
            loss_pl_value = self.loss_pl(predicted_img, img)
            total_loss = total_loss + loss_pl_value
            metrics['gen_loss_pl'] = loss_pl_value

        log_message = f"Epoch: {self.current_epoch} | " + " | ".join(
    [f"{key}: {value.item():.4f}" if isinstance(value, torch.Tensor) and value.numel() == 1 else f"{key}: {value.mean().item():.4f}" if isinstance(value, torch.Tensor) else f"{key}: {value:.4f}" for key, value in metrics.items()]
)

        LOGGER.info(log_message)
        return total_loss, metrics

    def discriminator_loss(self, batch):
        total_loss = 0
        metrics = {}

        predicted_img = batch[self.image_to_discriminator].detach()
        self.adversarial_loss.pre_discriminator_step(real_batch=batch['image'], fake_batch=predicted_img,
                                                     generator=self.generator, discriminator=self.discriminator)
        discr_real_pred, discr_real_features = self.discriminator(batch['image'])
        discr_fake_pred, discr_fake_features = self.discriminator(predicted_img)
        adv_discr_loss, adv_metrics = self.adversarial_loss.discriminator_loss(real_batch=batch['image'],
                                                                               fake_batch=predicted_img,
                                                                               discr_real_pred=discr_real_pred,
                                                                               discr_fake_pred=discr_fake_pred,
                                                                               mask=batch['mask'])
        total_loss = total_loss + adv_discr_loss
        metrics['discr_adv'] = adv_discr_loss
        metrics.update(add_prefix_to_keys(adv_metrics, 'adv_'))

        return total_loss, metrics
