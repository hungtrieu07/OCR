import paddle

paddle.utils.run_check()
print("Number of GPUs: ", paddle.device.cuda.device_count())
print(paddle.device.cuda.get_device_properties())
