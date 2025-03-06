## SepMark

### 实验目的

本次实验需验证三点：

- 双重水印攻击是有效的，即对于同一张图片，第二次编码将使得第一次编码的信息消失；
- 提出 Loss 约束是有效的，即能够解决双重水印攻击；
- 部分微调是可行的，或者说仅针对解码器微调、而不进行全量微调是可行的；

### 实验结果

- 无微调/微调对比

|    |    g_loss |   error_rate_C |   error_rate_R |   error_rate_F |    psnr |     ssim |   g_loss_on_discriminator |   g_loss_on_encoder_MSE |   g_loss_on_encoder_LPIPS |   g_loss_on_decoder_C |   g_loss_on_decoder_R |   g_loss_on_decoder_F |   d_loss |   g_loss_on_double_watermark |   double_error_rate_C |   double_error_rate_R |   double_error_rate_F |
|---:|----------:|---------------:|---------------:|---------------:|--------:|---------:|--------------------------:|------------------------:|--------------------------:|----------------------:|----------------------:|----------------------:|---------:|-----------------------------:|----------------------:|----------------------:|----------------------:|
|  未微调（Epoch 100） | 1.1644    |     0.00691318 |    0.000124139 |       0.478008 | 38.7788 | 0.938527 |                   2.03329 |             0.000531517 |                0.00808962 |           0.000523669 |           0.000302436 |           1.08015e-06 |  2.00961 |                   0.114804   |            0.498312   |              0.500698 |              0.487456 |
|  微调（Epoch 100） | 0.0316224 |     0.00481385 |    6.89663e-05 |       0.485183 | 38.3354 | 0.929688 |                   2.12371 |             0.000589209 |                0.00874837 |           0.000439849 |           0.000251798 |           5.70558e-07 |  1.95333 |                   0.00159519 |            0.00129381 |              0        |              0.493148 |

### 实验过程

在 `network/Dual_Mark.py` 的第 200 行左右，添加如下损失行：

```Python
# ======================= Double watermarking ====================== #
double_message = torch.Tensor(np.random.choice([-1.0, 1.0], (images.shape[0], 128))).to('cuda')
double_encoded_images, double_noised_images, double_decoded_messages_C, double_decoded_messages_R, double_decoded_messages_F = self.encoder_decoder(encoded_images, double_message, masks)
g_loss_on_double_watermark = (
	self.criterion_MSE(double_encoded_images, encoded_images) + 
	self.criterion_MSE(double_decoded_messages_C, decoded_messages_C)*5 + 
	self.criterion_MSE(double_decoded_messages_R, decoded_messages_R) + 
	self.criterion_MSE(double_decoded_messages_F, decoded_messages_F)
```

添加 Loss 后，从 Epoch 91 （包含 Epoch 91）训练至 Epoch 100。**在测试集上 error_rate_C 依然在 50% 附近**，这说明该 Loss 似乎没有作用（后发现实际上是因为测试代码有问题，且未找出测试代码出问题的原因。实际上，不管是不是由于测试代码有问题，这里的 Loss 这么写都是不合适的，应当先按照下面的方式重新写 Loss）。

猜测失败原因：约束第一次解码信息和第二次解码信息相同，但第一次解码信息本身又不一定准确。

解决方案：约束第一次 **编码** 信息和第二次解码信息相同，同时赋予权重。把添加的 Loss 修改为：

```Python
double_message = torch.Tensor(np.random.choice([-1.0, 1.0], (images.shape[0], 128))).to('cuda')
double_encoded_images, double_noised_images, double_decoded_messages_C, double_decoded_messages_R, double_decoded_messages_F = self.encoder_decoder(encoded_images, double_message, masks)
g_loss_on_double_watermark = (
    self.criterion_MSE(double_encoded_images, encoded_images) + 
    self.criterion_MSE(double_decoded_messages_C, messages)*5 + 
    self.criterion_MSE(double_decoded_messages_R, messages) + 
    self.criterion_MSE(double_decoded_messages_F, torch.zeros_like(messages))
)
```

训练结束后，修改 `test_Dual_Mark.py` 代码，使其编码两次水印，运行得到的 error_rate_C 依然很高（50% 左右浮动）。但是如果借助模型的 `validation()` 函数在测试集上运行，则结果正常：

```Python
with tqdm(total=len(val_dataloader)) as pbar:
    for step, (image, mask) in enumerate(val_dataloader, 1):
        image = image.to(device)
        message = torch.Tensor(np.random.choice([-message_range, message_range], (image.shape[0], message_length))).to(device)

        result_origin, (images, encoded_images, noised_images) = network_origin.validation(image, message, mask)
        result_double_watermark, (images, encoded_images, noised_images) = network_double_watermark.validation(image, message, mask)

        for key in result_origin:
            test_result_origin[key] += float(result_origin[key])
            test_result_double_watermark[key] += float(result_double_watermark[key])
        
for key in result_origin:
    test_result_origin[key] /= len(val_dataloader)
    test_result_double_watermark[key] /= len(val_dataloader)
pd.DataFrame([test_result_origin, test_result_double_watermark], index=[0]).to_markdown("test_result.md")
```

猜测是 `test_Dual_Mark.py` 代码的问题，暂未找到原因。不过好在终于可以在测试集上检验微调效果，暂时