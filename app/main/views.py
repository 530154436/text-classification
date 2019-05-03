import math
import uuid
import os
from config import BASE_DIR

from flask import render_template, redirect, url_for, flash, request,jsonify, app
from flask_login import login_required, current_user
from docx import Document

from app import get_logger, get_config
from app import utils
from app.main.forms import CfgNotifyForm,UploadForm
from models import CfgNotify
from . import main
from run_app_dev import segmenter,text_rnn

logger = get_logger(__name__)
cfg = get_config()

# 通用列表查询
def common_list(DynamicModel, view, subject):
    # 接收参数
    action = request.args.get('action')
    id = request.args.get('id')
    page = int(request.args.get('page')) if request.args.get('page') else 1
    length = int(request.args.get('length')) if request.args.get('length') else cfg.ITEMS_PER_PAGE

    print(subject, action, id, page, length)

    # 删除操作
    if action == 'del' and id:
        try:
            DynamicModel.get(DynamicModel.id == id).delete_instance()
            flash('删除成功')
        except:
            flash('删除失败')

    # 查询列表
    query = DynamicModel.select()
    if subject:
        query = DynamicModel.select().where(DynamicModel.subject==subject)
        print(query)
    total_count = query.count()

    # 处理分页
    if page: query = query.paginate(page, length)

    data_list = utils.query_to_list(query)
    for data in data_list:
        for k,v in data.items():
            if k in [ 'title', 'content']:
                data[k] = data[k][:20]
    dict = {'content': data_list, 'total_count': total_count,
            'total_page': math.ceil(total_count / length), 'page': page, 'length': length}
    return render_template(view, form=dict, current_user=current_user)


# 通用单模型查询&新增&修改
def common_edit(DynamicModel, form, view):
    id = request.args.get('id', )
    if id:
        # 查询
        model = DynamicModel.get(DynamicModel.id == id)
        if request.method == 'GET':
            utils.model_to_form(model, form)
        # 修改
        if request.method == 'POST':
            if form.validate_on_submit():
                utils.form_to_model(form, model)
                model.save()
                flash('保存成功')
            else:
                utils.flash_errors(form)
    else:
        # 新增
        if form.validate_on_submit():
            model = DynamicModel()
            utils.form_to_model(form, model)
            model.id = str(uuid.uuid1())

            # 用户id
            model.user_id = current_user.__dict__['__data__']['id']
            print(current_user.__dict__['__data__'])

            # 标题处理
            model.title = model.title.split('_')[0]

            # 分词、分类
            words = segmenter.seg_sentence(model.content)
            subject = text_rnn.predict_documents([' '.join(words[:600])])[0]
            words = ' '.join(words)
            print(subject, words[:100])
            model.words = words
            model.subject = subject

            model.save(force_insert=True)
            flash('上传成功')
        else:
            utils.flash_errors(form)
    return render_template(view, form=form, current_user=current_user)

def uploads(view):
    '''
    上传
    '''
    form = UploadForm()
    file_names = []
    if form.validate_on_submit() and 'files' in request.files:
        for f in request.files.getlist('files'):
            file_names.append(f.filename)
            file_path = os.path.join(BASE_DIR, 'data/uploads', f.filename)
            f.save(file_path)

            # 处理文件
            if 'doc' in f.filename:
                word_doc = Document(file_path)
                content = []
                for paragraph in word_doc.paragraphs:
                    content.append(paragraph.text)
                model = CfgNotify()
                model.id = str(uuid.uuid1())
                # 用户id
                model.user_id = current_user.__dict__['__data__']['id']
                model.title = f.filename.split('.')[0]
                model.content = '\n'.join(content)

                # 分词、分类
                words = segmenter.seg_sentence(model.content)
                subject = text_rnn.predict_documents([' '.join(words[:600])])[0]
                words = ' '.join(words)
                print(subject, words[:100])
                model.words = words
                model.subject = subject

                model.save(force_insert=True)

        print("{} 上传成功".format(' '.join(file_names)))
        flash("{} 上传成功".format(' '.join(file_names)))
    return render_template(view, form=form, current_user=current_user)

# 查询仪表盘
SUBJECTS={"语文":"yuwen_count", "数学":"shuxue_count", "英语":"yingyu_count",
          "物理":"wuli_count",  "化学":"huaxue_count", "生物":"shengwu_count",
          "历史":"lishi_count", "地理":"dili_count",   "政治":"zhengzhi_count"}

@main.route('/api/stats/summary', methods=['GET'])
@login_required
def count():
    user_id = current_user.__dict__['__data__']['id']

    json = {}
    for subject,count_name in SUBJECTS.items():
        count = CfgNotify.select()\
                         .where((CfgNotify.user_id==user_id) & (CfgNotify.subject==subject))\
                         .count()
        json[count_name] = count
    return jsonify(json)

# 根目录跳转
@main.route('/', methods=['GET'])
@login_required
def root():
    return redirect(url_for('main.index'))

# 文件上传
@main.route('/notifyupload', methods=['GET', 'POST'])
def upload():
    return uploads('notifyupload.html')

# 首页
@main.route('/index', methods=['GET'])
@login_required
def index():
    return render_template('index.html', current_user=current_user)

# 批量上传
@main.route('/notifyupload', methods=['GET', 'POST'])
@login_required
def notifyupload():
    return render_template('notifyupload.html')

# 查询
@main.route('/notifylist', methods=['GET', 'POST'])
@login_required
def notifylist():
    subject = request.args.get('subject', None)
    return common_list(CfgNotify, 'notifylist.html', subject)

# 编辑
@main.route('/notifyedit', methods=['GET', 'POST'])
@login_required
def notifyedit():
    return common_edit(CfgNotify, CfgNotifyForm(), 'notifyedit.html')
