from flask import Flask, current_app
from flask_cors import CORS
from settings.settings import Config
from routes.clfRoutes import *
from bert2vec_model import load_model



# 启动web容器
if __name__ == '__main__':
    # 初始化app
    app = Flask(__name__)
    app.config.from_object(Config)
    # 允许跨域
    CORS(app, supports_credentials=True)

    #挂载相关路由
    app.register_blueprint(clfApp, url_prefix='/clf')

    app.app_context().push()
    # 初始化模型信息，作为全局变量挂载
    current_app.net, current_app.tokenizer = load_model(app.config['MODEL_PATH'])
    app.run()
