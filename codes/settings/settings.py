#  全局配置类
class Config(object):
    # 调试模式
    DEBUG = True
    # 配置日志
    # LOG_LEVEL = "DEBUG"
    LOG_LEVEL = "INFO"

    # 自定义的配置功能(配置上传文件地址、模型文件地址)
    UPLOAD_FOLDER = '../static/upload_files'
    MODEL_PATH = "../static/fine_tuning_model.model"
    BATCH_LIMIT = 10
