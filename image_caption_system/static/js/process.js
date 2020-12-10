$(document).ready(function(){
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
    var fileobj = $("#videofile")[0].files[0];
    var form = new FormData();
    form.append('method',method);
    form.append('video',fileobj);
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
        	var list = eval(result);

        	var video_id = list[0];
        	if(video_id==='fail'){
        		alert("请勿提交空文件");
				location.reload();
        		return
			}
        	if(video_id==='processfail'){
        		alert("处理失败");
				location.reload();
        		return
			}
        	// alert(video_id);
        	newurl = '/image_caption_system/'+video_id;
			// alert(newurl)
			// 拿到新的id，去新的界面渲染。
        	window.location.href = newurl


        }

    });
}

var myVideo1=document.getElementById("videoid1"); 
var myVideo2=document.getElementById("videoid2"); 
function makePause(){
//	$('#videoid1').pause();
//	$('#videoid2').pause();
	myVideo1.pause(); 
	myVideo2.pause();
}
function makePlay(){ 
	myVideo1.play(); 
	myVideo2.play(); 
}
function makeRePlay(){ 
	myVideo1.load(); 
	myVideo2.load(); 
}
function makeBig(){ 
	myVideo1.width=560; 
}
function makeSmall(){ 
	myVideo1.width=320; 
}
function makeNormal(){ 
	myVideo1.width=420; 
}

function onInputFileChange() {
	// alert("hello");
	var file = document.getElementById('videofile').files[0];
	var url = URL.createObjectURL(file);
	// alert(url);

	$('#videoid').attr('src', url);
	$('#videoid').load();
	// document.getElementById("video").load();
	// alert("why?")
	// document.getElementById("video").src = url;
	// $('#video').load();


}
