# Deepfake Detection

做实验时发现可以使用 writer.add_text() 配合 pandas 生成参数表：

```python
writer.add_text("profile", pd.DataFrame({
	"project_name": project_name, 
	"time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
	"device": device,
	"batch_size": batch_size,
	"lr": lr,
	"epoch_number": epoch_number,
	"message_length": message_length,
	"noise_layers": noise_layers,
	"with_diffusion": with_diffusion,
	"only_decoder": only_decoder,
	"train_continue": train_continue,
	"train_continue_path": train_continue_path,
	"train_continue_epoch": train_continue_epoch,
	"save_images_number": save_images_number,
	"dataset_path": dataset_path
}).T.reset_index().rename(columns={"index": "key", 0: "value"}).to_markdown(index=False))
```

这样在 tensorboard 中就可以看到参数表了。