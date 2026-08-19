function show_stim(sc,DataToSave)
{		
	$('#Dial').hide();
	if (AutomaticResponse ==1){setTimeout("$('#button_3').click()",2500)}	// to run alone

	var StartTime = +new Date();											//starts the trial
	$('#preg').html('¿A qué nube pertenece el puntito amarillo?');
	var maincanvas = document.getElementById('MainCanvas');
	var context_canvas = maincanvas.getContext('2d');

	if (sc.trial < 3){
		setTimeout("$('#preg').fadeIn(100);",0);}

	var myVar1 = setTimeout(function(){
		$('#MainCanvas').show();
		$('#button_1').show();
		$('#button_2').show();
		$('#button_3').show();
		if (sc.nstim[sc.trial-1] ==2){										// hides 1 unused button
			var clc = DataToSave.col_clouds[2];			
			$('#button_'+ (DataToSave.col_bots.indexOf(clc) + 1).toString()).hide();}
		}, T1);
	
	var myVar2 = setTimeout(function(){										// blinking target
		context_canvas.beginPath();
		context_canvas.arc(DataToSave.Xtarget, DataToSave.Ytarget, DataToSave.Target_exterior_size, 0, 2 * Math.PI);
		context_canvas.closePath();
		context_canvas.fillStyle = "#FFFF00";
		context_canvas.fill( );
	
		context_canvas.beginPath();
		context_canvas.arc(DataToSave.Xtarget, DataToSave.Ytarget, DataToSave.Target_interior_size, 0, 2 * Math.PI);
		context_canvas.closePath();
		context_canvas.fillStyle = "#000000";
		context_canvas.fill( );
		}, T1+150);

	var myVar3 = setTimeout(function(){
		context_canvas.beginPath();
		context_canvas.arc(DataToSave.Xtarget, DataToSave.Ytarget, DataToSave.Target_exterior_size, 0, 2 * Math.PI);
		context_canvas.closePath();
		context_canvas.fillStyle = "#000000";
		context_canvas.fill( );
	
		context_canvas.beginPath();
		context_canvas.arc(DataToSave.Xtarget, DataToSave.Ytarget, DataToSave.Target_interior_size, 0, 2 * Math.PI);
		context_canvas.closePath();
		context_canvas.fillStyle = "#FFFF00";
		context_canvas.fill( );
			}, T1+300);
							

	var myVar4 = setTimeout(function(){
		$("#MainCanvas").hide()
	}, T2);
	
	
$('.resp_button').click(function(event)									// waits for a click/tap
{
	DataToSave.RT	= +new Date() - StartTime - T1;
	clearTimeout(myVar2);												// cancels automatic ending of stimuli
	clearTimeout(myVar3);
	clearTimeout(myVar4);

	if ($(event.target).is('#button_1')){
		sc.response[sc.trial-1] = 0;  		// clicked button
		
		$("#button_1").unbind('click');
		$("#MainCanvas").fadeOut(150);
	}
	if ($(event.target).is('#button_2')){
		sc.response[sc.trial-1] = 1;	   
		$("#button_2").unbind('click');
		$("#MainCanvas").fadeOut(150);
	}

	if ($(event.target).is('#button_3')){
		sc.response[sc.trial-1] = 2;	   
		$("#button_3").unbind('click');
		$("#MainCanvas").fadeOut(150);
	}

	var nube = 0;
	if(sc.response[sc.trial-1] == DataToSave.col_clouds[0]){
		nube = 1;
		DataToSave.nube = nube;
	}	
	if(sc.response[sc.trial-1] == DataToSave.col_clouds[1]){
		nube = 2;
		DataToSave.nube = nube;
	}
	if(sc.response[sc.trial-1] == DataToSave.col_clouds[2]){
		nube = 3;
		DataToSave.nube = nube;
	}
	
	
	
	sc.correct[sc.trial-1] = sc.response[sc.trial-1] == DataToSave.CorrectColor 

	$('.resp_button').fadeOut(200);
	$('#preg').fadeOut(200)
	$(".resp_button").unbind('click');
	DataToSave.response    = sc.response[sc.trial-1];
	DataToSave.correct 	   = sc.correct[sc.trial-1];
	setTimeout("$('#MainCanvas').hide(0,introspective_response(DataToSave))",500);
});

}
