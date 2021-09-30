# 1.基础镜像信息
FROM lightandshadow/pytorch_ml_base

# 2.维护者相关信息
LABEL maintainer="neu_fhf"

# 3. 安装相关环境
# 将所有文件复制到虚拟环境目录
COPY . /JudementProjectWeb
WORKDIR /JudementProjectWeb
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 4. 镜像生成后执行的命令
CMD ["gunicorn", "MainWebApp:app", "-c", "gunicorn.conf.py"]