from celery import Celery

def make_celery(app_name=__name__):
    """
    创建并返回一个Celery实例。

    :param app_name: 应用的名称，默认为当前模块的名称。
    :return: Celery实例。
    """
    # 创建Celery实例，指定应用名称和Broker URL
    return Celery(app_name, broker='redis://localhost:6379/0')

# 调用make_celery函数，创建Celery实例
celery = make_celery()