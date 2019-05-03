from flask import Flask
from flask_login import LoginManager
from conf.config import config
import logging
from logging.config import fileConfig
import os
from flask_uploads import UploadSet, configure_uploads, DEFAULTS
from config import BASE_DIR

login_manager = LoginManager()
login_manager.session_protection = 'strong'
login_manager.login_view = 'auth.login'
fileConfig('conf/log-app.conf')

def get_logger(name):
    return logging.getLogger(name)


def get_basedir():
    return os.path.abspath(os.path.dirname(__file__))


def get_config():
    return config[os.getenv('FLASK_CONFIG') or 'default']


def create_app(config_name):
    app = Flask(__name__)
    app.config.from_object(config[config_name])
    config[config_name].init_app(app)

    login_manager.init_app(app)

    # 配置上传文件属性
    if not os.path.exists(os.path.join(BASE_DIR, 'data/uploads')):
        os.mkdir(os.path.join(BASE_DIR, 'data'))
        os.mkdir(os.path.join(BASE_DIR, 'data', 'uploads'))
    app.config['UPLOADS_DEFAULT_DEST'] = os.path.join(BASE_DIR, 'data')
    files = UploadSet('files', DEFAULTS)
    configure_uploads(app, files)

    from app.main import main as main_blueprint
    from app.auth import auth as auth_blueprint
    app.register_blueprint(main_blueprint)
    app.register_blueprint(auth_blueprint)

    return app,files
