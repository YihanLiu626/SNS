import h5py
import shutil

# 选择要修改的模型文件
file_path = "best_model_AAPL.h5"
new_file_path = "fixed_best_model_AAPL.h5"

# 先复制原始模型，防止修改失败
shutil.copy(file_path, new_file_path)

# 处理 H5 文件
with h5py.File(new_file_path, "r+") as f:
    if "model_config" in f.attrs:
        model_config = f.attrs["model_config"]
        if isinstance(model_config, bytes):  # 需要解码为字符串
            model_config = model_config.decode("utf-8")

        # 移除 "time_major": false
        model_config = model_config.replace('"time_major": false,', "")
        model_config = model_config.replace('"time_major": false', "")

        # 重新编码并存储
        f.attrs["model_config"] = model_config.encode("utf-8")

print(f"✅ 已移除 time_major 参数，修改后的模型保存为 {new_file_path}")