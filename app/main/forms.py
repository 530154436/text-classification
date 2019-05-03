import os
from config import BASE_DIR
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, BooleanField, FileField, SelectField, TextAreaField, HiddenField
from wtforms.validators import DataRequired, Length, Email, Regexp, EqualTo
from flask_wtf.file import FileField, FileAllowed, FileRequired
from run_app_dev import files

class CfgNotifyForm(FlaskForm):
    title = StringField('标题', validators=[DataRequired(message='不能为空'), Length(0, 255, message='长度不正确')])
    user_id = StringField('用户id', default='')
    subject = StringField('学科', default='')
    content = TextAreaField('内容', validators=[DataRequired(message='不能为空')])
    url = StringField('原始链接', default=None)
    status = BooleanField('生效标识', default=True)
    submit = SubmitField('提交')

class UploadForm(FlaskForm):
    files = FileField('files', validators=[
        FileAllowed(files, '只允许上传文本文件!'),
        FileRequired("文件为空!"),
    ])
    submit = SubmitField('提交')