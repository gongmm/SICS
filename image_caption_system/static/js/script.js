jQuery(document).ready(function($) {
"use strict";
//  TESTIMONIALS CAROUSEL HOOK
$('#customers-testimonials').owlCarousel({
    loop: true,
    items: 3,
    margin: 0,
    autoplay: true,
    dots:true,
    nav:true,
    autoplayTimeout: 8500,
    smartSpeed: 450,
  navText: ['<i class="fa fa-angle-left fa-5x"></i>','<i class="fa fa-angle-right fa-5x"></i>'],
    responsive: {
      0: {
        items: 1
      },
      768: {
        items: 1
      },
      1170: {
        items: 1
      }
    }
  });
// 下面几行，在任何js中只要需要发送ajax请求，都要包含。html中还要包含{% csrf_token %}
	var csrftoken = jQuery("[name=csrfmiddlewaretoken]").val();
	function csrfSafeMethod(method) {
		// these HTTP methods do not require CSRF protection
		return (/^(GET|HEAD|OPTIONS|TRACE)$/.test(method));
	}
	$.ajaxSetup({
		beforeSend: function(xhr, settings) {
			if (!csrfSafeMethod(settings.type) && !this.crossDomain) {
				xhr.setRequestHeader("X-CSRFToken", csrftoken);
			}
		}
	});
});


function UploadFile(method){
    // alert(method)
    // var fileobj = $("#videofile")[0].files[0];
    console.log($("#file"))
    // console.log($("#file")[0])
    var fileobj = $("#file")[0].files[0];
    var form = new FormData();
    form.append('method',method);
    form.append('image',fileobj);
    $("#shadowid").show();
    $.ajax({
        type:'POST',
        url:'/image_caption_system/upload',
        dataType:"json",
        data:form,
        //data: {method:"upvideo",fileobj:fileobj},
        processData:false,  // 告诉jquery不转换数据
        contentType:false,  // 告诉jquery不设置内容格式
        success:function (result) {
            // alert("评测成功！");
            $("#shadowid").hide();
            console.log(result)
            var list = eval(result);

            var video_id = list[0];
            if(video_id==='fail'){
                alert("请勿提交空文件");
                // console.log("请勿提交空文件");
                location.reload();
                return
            }
            if(video_id==='processfail'){
                alert("处理失败");
                location.reload();
                return
            }
            var showcap=document.getElementById("caption-res");
            console.log(showcap);
            document.getElementById("caption-res").value=result.caption;
            console.log(result.caption);

            // alert(video_id);
            // newurl = '/image_caption_system/'+video_id;
            // alert(newurl)
            // 拿到新的id，去新的界面渲染。
            // window.location.href = newurl
        }

    }
    );
}
