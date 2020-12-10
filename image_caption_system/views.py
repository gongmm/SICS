from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect, JsonResponse
from django.template import loader

from .infer import get_style_image_caption
from .models import StyleImageCaption
from django.urls import reverse
import json
from django.views import generic
import pytz
from datetime import datetime
import os


def index(request):
    """
    index page
    """
    return render(request, 'index.html')


def process(request):
    """
    detection page
    """
    return render(request, 'process.html')


def upload(request):
    """
    上传视频文件，进行处理，并返回处理之后的地址
    :param request:
    :return:
    """
    # 获取上传的文件，如果没有文件，则默认为None
    upload_image = request.FILES.get("image", None)
    # method_id为处理方法的编号
    # 1，表示使用xception；2，表示用模糊检测；
    method_id = request.POST.get('method')
    if not upload_image:
        print("no files!!!")
        result = ['fail']
        return JsonResponse(json.dumps(result), content_type='application/json', safe=False)
    base_path = os.path.dirname(os.path.abspath(__file__))

    # 先将上传的图片保存到本地
    origin_write_path = os.path.join(base_path, "static/images", upload_image.name)
    image_path = os.path.join("static/images", upload_image.name)

    if not os.path.exists(origin_write_path):
        # 存数据库的只是相对路径
        # 打开特定的文件进行二进制的写操作
        destination = open(origin_write_path, 'wb+')
        # 分块写入文件
        for chunk in upload_image.chunks():
            destination.write(chunk)
        destination.close()

    vocab_path = os.path.join(base_path, "models/xception_focal.pth.tar")
    encoder_model_path = os.path.join(base_path, "models/xception_focal.pth.tar")
    decoder_model_path = os.path.join(base_path, "models/xception_focal.pth.tar")

    style_image_caption = get_style_image_caption(image_path, vocab_path, encoder_model_path, decoder_model_path)
    if style_image_caption is None:
        print("处理失败")
        result = ['processfail']
        return JsonResponse(json.dumps(result), content_type='application/json', safe=False)

    # 将路径保存到数据库中。
    tz = pytz.timezone('Asia/Shanghai')
    t = datetime.now(tz)
    timestamp = t.strftime('%Y-%m-%d %H:%M:%S')
    image_id = t.strftime('%Y%m%d%H%M%S')

    style_image_caption = StyleImageCaption(video_id=image_id)
    style_image_caption.image_path = image_path
    style_image_caption.time_stamp = timestamp
    style_image_caption.caption = style_image_caption

    style_image_caption.save()

    result = [image_id]
    # print(result)
    return JsonResponse(json.dumps(result), content_type='application/json', safe=False)


def detail(request, video_id):
    """
    显示图片详情
    :param request:
    :param video_id:
    :return:
    """
    image = StyleImageCaption.objects.get(video_id=video_id)
    return render(request, 'detail.html', {'image': image})


def history(request):
    """
    按照时间顺序显示系统中记录在数据库中的所有图片
    :param request:
    :return:
    """
    images = StyleImageCaption.objects.order_by('-time_stamp')

    return render(request, 'history.html', {'images': images})


def delete(request, image_id):
    """
    根据video_id删除对应的图片
    :param request:
    :param image_id:
    :return:
    """
    videos = StyleImageCaption.objects.get(image_id=image_id)
    videos.delete()
    # 转发
    return HttpResponseRedirect(reverse('history'))
