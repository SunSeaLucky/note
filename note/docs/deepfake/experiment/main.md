## SepMark

### 实验目的

本次实验需验证三点：

- 双重水印攻击是有效的，即对于同一张图片，第二次编码将使得第一次编码的信息消失；
- 提出 Loss 约束是有效的，即能够解决双重水印攻击；
- 部分微调是可行的，或者说仅针对解码器微调、而不进行全量微调是可行的；

### 实验记录

---

#### 第一次实验

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

添加 Loss 后，从 Epoch 91 （包含 Epoch 91）训练至 Epoch 100。**在测试集上 error_rate_C 依然在 50% 附近**，这说明该 Loss 似乎没有作用。

猜测失败原因：约束第一次解码信息和第二次解码信息相同，但第一次解码信息本身又不一定准确。

解决方案：约束第一次 **编码** 信息和第二次解码信息相同，同时赋予权重。

#### 第二次实验

把添加的 Loss 修改为：

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

训练结束后，修改 `test_Dual_Mark.py` 代码，使其编码两次水印，运行得到的 error_rate_C 依然很高（50% 左右浮动）。但是如果借助模型的 `validation` 函数在测试集上运行，则结果正常：

```Python
import yaml
from easydict import EasyDict
import os
import time
from shutil import copyfile
import random
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from network.Dual_Mark import *
from utils import *
from tqdm import tqdm


def seed_torch(seed=42):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True


def main():
    # ========================== Init network ========================== #
    network = Network(message_length, noise_layers_R, noise_layers_F, device, batch_size, lr, beta1, attention_encoder, attention_decoder, weight)
    if args.resume_config.resume:
        network.load_model(
            args.resume_config.encoder_decoder_model_path, 
            args.resume_config.discriminator_model_path, 
        )
        
    # ========================== Init dataset ========================== #
    train_dataset = attrsImgDataset(train_dataset_path, image_size, "celebahq")
    #train_dataset = maskImgDataset(os.path.join(dataset_path, "train_" + str(image_size)), image_size)
    assert len(train_dataset) > 0, "train dataset is empty"
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    
    val_dataset = attrsImgDataset(val_dataset_path, image_size, "celebahq")
    assert len(val_dataset) > 0, "val dataset is empty"
    #val_dataset = maskImgDataset(os.path.join(dataset_path, "val_" + str(image_size)), image_size)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # =========================== Train & Val ========================== #
    print("\nStart training : \n\n")


    # =========================== Validation =========================== #
    val_result = {
        "g_loss": 0.0,
        "error_rate_C": 0.0,
        "error_rate_R": 0.0,
        "error_rate_F": 0.0,
        "psnr": 0.0,
        "ssim": 0.0,
        "g_loss_on_discriminator": 0.0,
        "g_loss_on_encoder_MSE": 0.0,
        "g_loss_on_encoder_LPIPS": 0.0,
        "g_loss_on_decoder_C": 0.0,
        "g_loss_on_decoder_R": 0.0,
        "g_loss_on_decoder_F": 0.0,
        "d_loss": 0.0,
        # ==================== Added double watermarking =================== #
        "g_loss_on_double_watermark": 0.0, 
        "double_error_rate_C": 0.0,
        "double_error_rate_R": 0.0,
        "double_error_rate_F": 0.0
    }
    start_time = time.time()

    saved_iterations = np.random.choice(np.arange(1, len(val_dataloader)+1), size=save_images_number, replace=False)
    saved_all = None

    for step, (image, mask) in enumerate(val_dataloader, 1):
        image = image.to(device)
        message = torch.Tensor(np.random.choice([-message_range, message_range], (image.shape[0], message_length))).to(device)

        result, (images, encoded_images, noised_images) = network.validation(image, message, mask)
        print(f"error rate C: {result['error_rate_C']:.4f}, error rate R: {result['error_rate_R']:.4f}, error rate F: {result['error_rate_F']:.4f}, psnr: {result['psnr']:.4f}, ssim: {result['ssim']:.4f}")
        print(f"double error rate C: {result['double_error_rate_C']:.4f}, double error rate R: {result['double_error_rate_R']:.4f}, double error rate F: {result['double_error_rate_F']:.4f}")

        for key in result:
            val_result[key] += float(result[key])

if __name__ == '__main__':
    seed_torch(42) # it does not work if the mode of F.interpolate is "bilinear"
    # ======================= Init configuration ======================= #
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    with open('cfg/train_DualMark.yaml', 'r') as f:
        args = EasyDict(yaml.load(f, Loader=yaml.SafeLoader))
    project_name = args.project_name
    epoch_number = args.epoch_number
    batch_size = args.batch_size
    lr = args.lr
    beta1 = args.beta1
    image_size = args.image_size
    message_length = args.message_length
    message_range = args.message_range
    attention_encoder = args.attention_encoder
    attention_decoder = args.attention_decoder
    weight = args.weight
    dataset_path = args.dataset_path
    save_images_number = args.save_images_number
    noise_layers_R = args.noise_layers.pool_R
    noise_layers_F = args.noise_layers.pool_F    
    train_dataset_path = os.path.join(dataset_path, "train_" + str(image_size))
    # 这里的val_dataset_path是测试集的路径，不是验证集的路径
    val_dataset_path = os.path.join(dataset_path, "test_" + str(image_size))
        
    if noise_layers_R is None:
        noise_layers_R = []
    if noise_layers_F is None:
        noise_layers_F = []
    assert os.path.exists(train_dataset_path), "train dataset is not exist"
    assert os.path.exists(val_dataset_path), "val dataset is not exist"

    project_name += "_" + str(image_size) + "_" + str(message_length) + "_" + str(message_range) + "_" + str(lr) + "_" + str(beta1) + "_" + attention_encoder + "_" + attention_decoder
    for i in weight:
        project_name += "_" +  str(i)
    result_folder = "results/" + time.strftime(project_name + "_%Y_%m_%d_%H_%M_%S", time.localtime()) + "/"
    if not os.path.exists(result_folder): os.mkdir(result_folder)
    if not os.path.exists(result_folder + "images/"): os.mkdir(result_folder + "images/")
    if not os.path.exists(result_folder + "models/"): os.mkdir(result_folder + "models/")
    copyfile("cfg/train_DualMark.yaml", result_folder + "train_DualMark.yaml")
    writer = SummaryWriter('runs/'+ project_name + time.strftime("%_Y_%m_%d__%H_%M_%S", time.localtime()))
    main()
    writer.close()
```

猜测是 `test_Dual_Mark.py` 代码的问题。暂未找到原因。

### 实验结果

运行 `./board.sh`，其中 

```
Dual_watermark_256_128_0.1_5e-05_0.5_se_se_1_10_10_10_0.1_2025_03_03_03_22_22
```

为修改代码后（第二次实验）的运行第 91-100 轮的过程。