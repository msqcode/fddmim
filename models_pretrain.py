from functools import partial

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed, Block

from loss_func import uniformity_loss
from util.pos_embed import get_2d_sincos_pos_embed
from transformers import CLIPVisionModel, ViTModel
import pdb


def resize_pos_embed(x):
    # [256, C] -> [196, C]
    C = x.shape[-1]
    x = x.reshape(1, 16, 16, C).permute(0, 3, 1, 2)
    x = F.interpolate(x, (14, 14), mode='bicubic', align_corners=False)
    x = x.permute(0, 2, 3, 1).reshape(196, C)
    return x


class LayerNorm(nn.Module):
    """
    A LayerNorm variant, popularized by Transformers, that performs point-wise mean and
    variance normalization over the channel dimension for inputs that have shape
    (batch_size, channels, height, width).
    https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa B950
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class MAE_Decoder(nn.Module):
    def __init__(self, inp_dim, embed_dim=256, out_dim=27, scale=1., num_patches=196, depth=1, num_heads=8,
                 mlp_ratio=4., qkv_bias=False, norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_patches = num_patches
        self.embed = nn.Linear(inp_dim, embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_patches, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=qkv_bias, norm_layer=norm_layer) for _ in range(depth)])
        self.norm = norm_layer(embed_dim)

        # pred head
        hidden = embed_dim
        if scale == 4.0:
            layers = [nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=2, stride=2),
                      LayerNorm(embed_dim // 2),
                      nn.GELU(),
                      nn.ConvTranspose2d(embed_dim // 2, embed_dim // 4, kernel_size=2, stride=2)]
            hidden = embed_dim // 4
        elif scale == 2.0:
            layers = [nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=2, stride=2)]
            hidden = embed_dim // 2
        elif scale == 1.0:
            layers = []
        elif scale == 0.5:
            layers = [nn.AvgPool2d(kernel_size=2, stride=2)]
        else:
            raise NotImplementedError(f"scale_factor={scale} is not supported yet.")
        layers.append(nn.Conv2d(hidden, out_dim, kernel_size=1))
        self.pred = nn.Sequential(*layers)

        self.initialize_weights()

    def initialize_weights(self):
        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize position embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches ** .5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, ids_restore):
        # embed tokens
        x = self.embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.pos_embed

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)  # [B, L, d]

        # predictor projection
        H = W = int(self.num_patches ** 0.5)
        x = x[:, 1:].transpose(1, 2).reshape(x.size(0), -1, H, W)
        x = self.pred(x)
        x = x.flatten(2, 3).transpose(1, 2)

        return x


class HOGLayer(nn.Module):
    def __init__(self, nbins, pool, bias=False, max_angle=math.pi, stride=1, padding=1, dilation=1):
        super(HOGLayer, self).__init__()
        self.nbins = nbins

        self.conv = nn.Conv2d(1, 2, 3, stride=stride, padding=padding, dilation=dilation, padding_mode='reflect',
                              bias=bias)
        mat = torch.FloatTensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        mat = torch.cat((mat[None], mat.t()[None]), dim=0)
        self.conv.weight.data = mat[:, None, :, :]

        self.max_angle = max_angle
        self.pooler = nn.AvgPool2d(pool, stride=pool, padding=0, ceil_mode=False, count_include_pad=True)

    @torch.no_grad()
    def forward(self, x):  # [B, 1, 224, 224]
        gxy = self.conv(x)

        # 2. Mag/ Phase
        mag = gxy.norm(dim=1)
        norm = mag[:, None, :, :]
        phase = torch.atan2(gxy[:, 0, :, :], gxy[:, 1, :, :])

        # 3. Binning Mag with linear interpolation
        phase_int = phase / self.max_angle * self.nbins
        phase_int = phase_int[:, None, :, :]

        n, c, h, w = gxy.shape
        out = torch.zeros((n, self.nbins, h, w), dtype=torch.float, device=gxy.device)
        out.scatter_(1, phase_int.floor().long() % self.nbins, norm)

        hog = self.pooler(out)
        hog = nn.functional.normalize(hog, p=2, dim=1)
        return hog


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16, drop_path_rate=0.,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 loss_weights="mean", mask_type="random", fusion_type="simple", target_norm="none", loss_type="l2",
                 head_type="linear", teacher_model="openai/clip-vit-base-patch16",
                 hog_nbins=9, hog_bias=False, decoder_embed_dim=512, decoder_depth=1, decoder_num_heads=16, mean=None,
                 std=None, weight=0.):
        super().__init__()

        if mean is None:
            mean = []
        assert loss_weights in ["mean", "out", "linear_decay"] or "top" in loss_weights or "mid" in loss_weights
        self.loss_weights = loss_weights
        assert mask_type in ["random", "attention"]
        self.mask_type = mask_type
        assert fusion_type in ["simple", "linear", "sum"]
        self.fusion_type = fusion_type
        assert target_norm in ["none", "l2", "whiten", "bn"]
        self.target_norm = target_norm
        assert loss_type in ["l2", "l1", "smoothl1"]
        self.loss_type = loss_type
        assert head_type in ["linear", "norm_linear", "mlp", "mlp2"]
        self.head_type = head_type
        # assert "clip" in teacher_model or "dino" in teacher_model
        self.teacher_model_name = teacher_model
        self.img_size = img_size
        self.mean = mean
        self.std = std
        self.weight = weight
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer,
                  drop_path=dpr[i])
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.ID = [1, 3, depth - 3, depth - 1]
        self.scale = [4.0, 2.0, 1.0, 0.5]
        self.norm1 = nn.ModuleList([norm_layer(embed_dim) for _ in range(len(self.ID))])
        # MIM decoder specifics
        self.decoder = nn.ModuleList([
            MAE_Decoder(embed_dim, decoder_embed_dim, in_chans * hog_nbins, s, num_patches, decoder_depth,
                        decoder_num_heads, mlp_ratio, True, norm_layer)
            for s in self.scale])
        # target
        self.hog_enc = nn.ModuleList([HOGLayer(nbins=hog_nbins, pool=k, bias=hog_bias) for k in [4, 8, 16, 32]])
        for hog_enc in self.hog_enc:
            for param in hog_enc.parameters():
                param.requires_grad = False

        if "clip-vit-base-patch16" in self.teacher_model_name:
            target_dim = 768
            teacher_depth = 12
        else:
            target_dim = 1024
            teacher_depth = 24

        if self.head_type == "linear":
            self.distill_heads = nn.ModuleList([nn.Linear(embed_dim, target_dim) for i in range(teacher_depth)])
        elif self.head_type == "norm_linear":
            self.distill_heads = nn.ModuleList([nn.Sequential(
                norm_layer(embed_dim),
                nn.Linear(embed_dim, target_dim)
            )
                for i in range(teacher_depth)])
        elif self.head_type == "mlp":
            self.distill_heads = nn.ModuleList([nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, target_dim)
            )
                for i in range(teacher_depth)])
        elif self.head_type == "mlp2":
            self.distill_heads = nn.ModuleList([nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                norm_layer(embed_dim),
                nn.Linear(embed_dim, target_dim)
            )
                for i in range(teacher_depth)])

        if self.fusion_type == "linear":
            # only len(student) == len(teacher)
            self.distill_weights = nn.Parameter(torch.eye(len(self.blocks)) + 0.01, requires_grad=True)
        elif self.fusion_type == "sum":
            self.distill_weights = nn.Parameter(torch.ones(teacher_depth, len(self.blocks)) / len(self.blocks),
                                                requires_grad=True)

        self.initialize_weights()

        if "clip" in self.teacher_model_name:
            self.clip_model = CLIPVisionModel.from_pretrained(self.teacher_model_name)
            for name, param in self.clip_model.named_parameters():
                param.requires_grad = False
                if "clip-vit-large-patch14" in self.teacher_model_name and "position_embedding" in name:
                    param.data = torch.cat([param.data[:1], resize_pos_embed(param.data[1:])], dim=0)
            if "clip-vit-large-patch14" in self.teacher_model_name:
                self.clip_model.vision_model.embeddings.position_ids = torch.arange(197).expand((1, -1))


    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        # torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def HOG(self, imgs, k):  # [B, 3, 224, 224]
        """
        imgs: (N, 3, H, W)
        x: (N, L, d)
        """
        hog_R = self.hog_enc[k](imgs[:, :1, :, :])  # [B, nb, h, w]
        hog_G = self.hog_enc[k](imgs[:, 1:2, :, :])  # [B, nb, h, w]
        hog_B = self.hog_enc[k](imgs[:, 2:, :, :])  # [B, nb, h, w]
        hog_feat = torch.cat([hog_R, hog_G, hog_B], 1)  # [B, 3*nb, h, w]
        hog_feat = hog_feat.flatten(2, 3).transpose(1, 2)
        return hog_feat

    def denormalize(self, images, type="cifar10"):
        # sr_images [B, 3, H, W]
        mean = torch.tensor(self.mean, device=images.device).view(1, 3, 1, 1).type_as(images)
        std = torch.tensor(self.std, device=images.device).view(1, 3, 1, 1).type_as(images)
        return std * images + mean

    def normalize(self, images, type="clip"):
        # images [B, 3, h, w]
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=images.device).view(1, 3, 1, 1).type_as(images)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=images.device).view(1, 3, 1, 1).type_as(images)
        return (images - mean) / std

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, ids_keep, mask, ids_restore

    def forward_encoder(self, x, mask_ratio, attentions):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        # if self.mask_type == "attention":
        #     importance = attentions[-1][:, :, 0, 1:].mean(1)
        #     x, ids_keep = self.attention_masking(x, mask_ratio, importance)
        # else:
        #     x, ids_keep, mask, ids_restore = self.random_masking(x, mask_ratio)
        x, ids_keep, mask, ids_restore = self.random_masking(x, mask_ratio)
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        hidden_states = []
        latent = []
        # apply Transformer blocks
        # for blk in self.blocks:
        #     x = blk(x)
        #     hidden_states.append(x)

        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            hidden_states.append(x)
            if i in self.ID:
                latent.append(self.norm1[self.ID.index(i)](x))
        x = self.norm(x)
        return hidden_states, ids_keep, latent, mask, ids_restore

    @torch.no_grad()
    def forward_clip(self, x):
        if "clip-vit-large-patch14" in self.teacher_model_name:
            x = F.interpolate(x, (196, 196), mode='bicubic', align_corners=False)
        if self.img_size == 64:
            x = F.interpolate(x, (224, 224), mode='bicubic', align_corners=False)
        x = self.normalize(self.denormalize(x))
        input = {
            "pixel_values": x,
            "output_hidden_states": True,
            "output_attentions": True
        }
        outputs = self.clip_model(**input)

        last_hidden_state, pooler_output, hidden_states, attentions = outputs[0], outputs[1], outputs[2], outputs[3]
        return last_hidden_state, pooler_output, hidden_states, attentions

    @torch.no_grad()
    def forward_dino(self, x):
        input = {
            "pixel_values": x,
            "output_hidden_states": True,
            "output_attentions": True
        }
        outputs = self.dino_model(**input)

        last_hidden_state, pooler_output, hidden_states, attentions = outputs[0], outputs[1], outputs[2], outputs[3]
        return last_hidden_state, pooler_output, hidden_states, attentions

    def get_student(self, hidden_states):
        student = hidden_states
        if self.fusion_type != "simple":
            student = [x.unsqueeze(0) for x in student]
            student = torch.cat(student, dim=0)
            student = torch.einsum('ab,bcde->acde', self.distill_weights, student)
            student = torch.chunk(student, student.shape[0], dim=0)
            student = [x.squeeze(0) for x in student]
        student = [self.distill_heads[i](x) for i, x in enumerate(student)]
        return student

    def get_teacher(self, hidden_states, ids_keep):
        teacher = []
        for i in range(1, len(hidden_states)):
            y = hidden_states[i]
            if self.target_norm == "l2":
                y = F.normalize(y, dim=-1)
            elif self.target_norm == "whiten":
                y = F.layer_norm(y, (y.shape[-1],))
            elif self.target_norm == "bn":
                y = (y - y.mean()) / (y.var() + 1.e-6) ** .5
            cls = y[:, :1, :]
            y = y[:, 1:, :]
            y = torch.gather(y, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, y.shape[-1]))
            teacher.append(torch.cat([cls, y], dim=1))
        return teacher

    def recal_mask(self, mask, k):
        B, L, s = mask.size(0), mask.size(1), self.scale[k]
        H = W = int(L ** .5)
        if s >= 1.:
            s = int(s)
            mask = mask.reshape(B, H, W).unsqueeze(3).unsqueeze(2).repeat(1, 1, s, 1, s).reshape(B, -1)
        else:
            s = int(1 / s)
            mask = mask.reshape(B, H // s, s, H // s, s).transpose(2, 3).mean((-2, -1)).reshape(B, -1)

        return mask

    def forward_loss(self, student, teacher, imgs, pred, mask, latent):
        """
        student: ([B*4, L//4, C]...)
        teacher: ([B, 1+L, C]...)
        ids_shuffle: [B, L]
        """
        loss1 = torch.tensor(0., device=student[0].device)
        loss2 = torch.tensor(0., device=student[0].device)

        if self.loss_weights == "mean":
            weight_list = [1 / len(student)] * len(student)
        elif self.loss_weights == "out":
            weight_list = [0.] * (len(student) - 1) + [1.]
        elif self.loss_weights == "linear_decay":
            weight_list_ = list(range(len(student)))
            weight_list = [i / sum(weight_list_) for i in weight_list_]
        elif "top" in self.loss_weights:  # topk
            topk = int(self.loss_weights[3:])
            weight_list = [0.] * (len(student) - topk) + [1 / topk] * topk
        elif "mid" in self.loss_weights:
            mid = int(self.loss_weights[3:])
            weight_list = [0.] * mid + [1.] + [0.] * (len(student) - mid - 1)

        for i, x in enumerate(student):
            y = teacher[i]
            if weight_list[i] > 0:
                if self.loss_type == "l2":
                    loss1 = loss1 + weight_list[i] * ((y - x) ** 2).mean()
                elif self.loss_type == "smoothl1":
                    loss1 = loss1 + weight_list[i] * 2 * F.smooth_l1_loss(y, x)
                elif self.loss_type == "l1":
                    loss1 = loss1 + weight_list[i] * F.l1_loss(y, x)

        target = [self.HOG(imgs, k) for k in range(len(self.hog_enc))]
        for k in range(len(pred)):
            M = self.recal_mask(mask, k)
            loss2 += (((pred[k] - target[k]) ** 2).mean(dim=-1) * M).sum() / M.sum()
        return self.weight * loss1 + (1 - self.weight) * loss2

    def forward(self, imgs, mask_ratio=0.75):
        if "clip" in self.teacher_model_name:
            _, _, hidden_states_teacher, attentions = self.forward_clip(imgs)
        hidden_states, ids_keep, latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio, attentions)
        student = self.get_student(hidden_states)
        teacher = self.get_teacher(hidden_states_teacher, ids_keep)
        pred = [self.decoder[i](latent[i], ids_restore) for i in range(len(latent))]
        loss = self.forward_loss(student, teacher, imgs, pred, mask, latent[3])
        return loss


def mae_vit_tiny_patch16(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=192, depth=12, num_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_small_patch16(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=384, depth=12, num_heads=12,
        mlp_ratio=4,
        decoder_embed_dim=256,
        decoder_depth=1,
        decoder_num_heads=8, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_base_patch16(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_tiny_patch16 = mae_vit_tiny_patch16
mae_vit_small_patch16 = mae_vit_small_patch16
mae_vit_base_patch16 = mae_vit_base_patch16
mae_vit_large_patch16 = mae_vit_large_patch16

