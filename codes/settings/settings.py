#  全局配置类
class Config(object):
    # 调试模式
    DEBUG = True
    # 配置日志
    # LOG_LEVEL = "DEBUG"
    LOG_LEVEL = "INFO"

    # 自定义的配置功能(配置上传文件地址、模型文件地址)
    UPLOAD_FOLDER = 'static/upload_files'
    MODEL_PATH = "static/fine_tuning_model_not_judgement.model"
    BATCH_LIMIT = 10
    LABELS = {0: "标题案号", 1: "当事人、辩护人、被害人情况", 2: "案件始末", 3: "指控", 4: "证实文件", 5: "辩护意见", 6: "事实",
              7: "证据列举", 8: "判决结果", 9: "尾部", 10: "法律条文等附录", 11: '非判决书段落'}
