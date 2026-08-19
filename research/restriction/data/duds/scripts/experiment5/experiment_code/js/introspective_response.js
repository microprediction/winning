function introspective_response(DataToSave)
{
	var LIMMIN = Math.ceil(screen.width/2) - Math.ceil($('#line1').width()/2) - 10;
	var LIMMAX = Math.ceil(screen.width/2) + Math.ceil($('#line1').width()/2) - 10;
	var range = LIMMAX - LIMMIN;


	$('#scale').css('color','#004D00')
	if (sc.trial > 1){   													// reset the animation from the previous trial
		$("#scale").animate({ right: '-15px',top: '15px', fontSize: '1em',opacity: '1.0'},10);
	}

	$('#line1').show();
	if (sc.trial < 4){$('#preg_segu').show();}

	var x = 0;  
	var StartTime = +new Date();

	function updateMouseDial(e){
		x = e.pageX;
		if (e.pageX >= LIMMAX){x = LIMMAX};
		if (e.pageX <= LIMMIN){x = LIMMIN};

		var percen = (100 * (x - LIMMIN)/range).toFixed(0)					// get percentaje 
		$('#scale').html(percen.toString().concat(' %'))
		$('#scale').css({"left": x- Math.ceil(screen.width/2)})
		$('#Dial').css({"left": x});
	}

	/*------------------mouse ---------*/
	if (InputMethod == 'Mouse' || InputMethod == 'Pad')
	{	
	if (AutomaticResponse ==1){setTimeout("$('#Dial').click()",500)}	// to run alone

	
	//$('#scale').hide();
	$(document).mousemove(function(e){									// slides the cursor
		updateMouseDial(e);
		$('#scale').show();
		$('#Dial').show();
	})


	$(document).click(function(ee){										// gets the click
		$(document).unbind('mousemove');
		x = ee.pageX;
		
		if (ee.pageX >= LIMMAX){x = LIMMAX};
		if (ee.pageX <= LIMMIN){x = LIMMIN};
		$('#scale').show();
		updateMouseDial(ee);
		
		
		$('#Dial').show();
		$('#Dial').css({"left": x})
		DataToSave.confidence		= (x - LIMMIN)/range;
		DataToSave.reactiontimeconf	= +new Date() - StartTime;
																		// animates
		$("#scale").animate({ right: '15px',top: '-15px', fontSize: '2em',opacity: '0.0'},500);


		$('#Dial').fadeOut(500);
		$('#line1').fadeOut(550);
		$('#preg_segu').fadeOut(550);
		$('#scale').fadeOut(500);
		$(document).unbind('click');
	
		Trials_save.push(sc.trial)
		Nalternativas_save.push(DataToSave.nstim)
		Distance_ratio_1_2_save.push(DataToSave.distance_21)
		angle_save.push(DataToSave.angle);
		Xtarget_save.push(DataToSave.Xtarget)
		Ytarget_save.push(DataToSave.Ytarget)
		SD_save.push(DataToSave.SD)
		Nsamples_save.push(DataToSave.Nsamples)
		Color_clouds_save.push(DataToSave.col_clouds)
		Color_buttons_save.push(DataToSave.col_bots)
		CorrectColor_save.push(DataToSave.CorrectColor)
		CorrectCloud_save.push(DataToSave.CorrectCloud)
		RT_type1_save.push(DataToSave.RT)
		Response_save.push(DataToSave.response)
		nube_save.push(DataToSave.nube)
		Correct_save.push(DataToSave.correct)
		Confidence_save.push(DataToSave.confidence)
		RT_Confidence_save.push(DataToSave.reactiontimeconf)


		if (sc.trial == sc.maxtrials ){
			var Data = {"Code": CODE,
					"Trial_number": Trials_save,
					"Nalternativas": Nalternativas_save,
					"Distance_ratio_1_2": Distance_ratio_1_2_save,
					"angle": angle_save,
					"Xtarget": Xtarget_save,
					"Ytarget": Ytarget_save,
					"SD": SD_save,
					"Nsamples": Nsamples_save,
					"color_clouds": Color_clouds_save,
					"color_buttons": Color_buttons_save,
					"CorrectColor": CorrectColor_save,
					"CorrectCloud": CorrectCloud_save,
					"RT_type1": RT_type1_save,
					"Response": Response_save,
					"cloud_clicked": nube_save,
					"Correct": Correct_save,
					"Confidence": Confidence_save,
					"RT_Confidence": RT_Confidence_save};
						
			jatos.appendResultData(JSON.stringify(Data));}

			if (sc.trial == sc.maxtrials){
			
			$('#Question_2').css('display', 'none');
			$('#botones').css('display', 'none');
			$('#Stimuli').css('display', 'none');
			
			$('#preg').html("El estudio ha finalizado. Para dudas y/o consultas podés comunicarte con los investigadores responsables a nicocomay@gmail.com ¡Muchas gracias por tu participación!<br>Si querés podes dejar tus datos para futuros experimentos.<br><button id='finalizar_estudio' class='button_instr'>Finalizar sin dejar datos</button><br><br><button id='base_experimentos' class='button_instr'>Finalizar dejando datos</button><br>¡Sería de gran ayuda que compartas este experimento con otras personas!"); // <button id='finalizar_estudio'>Finalizar estudio</button>"
			$('#preg').css("fontSize", "90%" );
			$('#preg').show();
			$('#Letterhead1').css("top", "-40px");
			$('#Question2').css('display', 'none'); //borramos de la pantalla este div para que no impida tocar el boton de finalizar
			$('#bar1').css('display', 'none');
			$('#bar2').css('display', 'none');
			$('#marks').css('display', 'none');
			$('#finalizar_estudio').click(function(){
				jatos.endStudyAndRedirect("https://google.com.ar/"); //uso endstudyandredirect porque endstudyajax da unos errores de cookies (samesite)
			});	
			$('#base_experimentos').click(function(){
				jatos.endStudyAndRedirect("https://sites.google.com/view/experimentosonline");
			});	
		return}
		setTimeout("$('div.linesConf').hide(0,newtrial(sc))",500);
  });
}

/*------------------touchscreen ---------*/
if (InputMethod == 'Touchscreen')
{	
	return

	$('#scale').hide();
	$('#Dial').hide();

	if (AutomaticResponse ==1){setTimeout("$('#Dial').trigger('touchend')",1000)}	// to run alone

	var Xpos = 0
	$(document).on('touchstart',function (e){										// gets the touch
		e.preventDefault();

		Xpos = e.originalEvent.touches[0].pageX;
		if (e.originalEvent.touches[0].pageX >= LIMMAX){Xpos = LIMMAX};
		if (e.originalEvent.touches[0].pageX <= LIMMIN){Xpos = LIMMIN};
		percen = (100 * (Xpos - LIMMIN)/range).toFixed(0) 							// get percentaje 
		$('#scale').html(percen.toString().concat(' %'))
		$('#Dial').css({"left": Xpos});
		$('#scale').show()
		$('#Dial').show();
	});

	$(document).on('touchmove',function (ee){										// slides the cursor
		$('#Dial').show();
		$('#scale').show();
		ee.preventDefault();
		Xpos = ee.originalEvent.touches[0].pageX;
		if (ee.originalEvent.touches[0].pageX >= LIMMAX){Xpos = LIMMAX};
		if (ee.originalEvent.touches[0].pageX <= LIMMIN){Xpos = LIMMIN};

		percen = (100 * (Xpos - LIMMIN)/range).toFixed(0) 							// get percentaje 
		$('#scale').html(percen.toString().concat(' %'))
		$('#scale').css({"left": Xpos- Math.ceil(screen.width/2)})
		$('#Dial').css({"left": Xpos});
	})

	$(document).on('touchend',function (eee){										// end of touch, get response
		eee.preventDefault();
		$(document).unbind('mousemove');
		DataToSave.confidence		= (Xpos - LIMMIN)/range;
		DataToSave.reactiontimeconf	= +new Date() - StartTime;

		$(document).unbind('touchstart');
		$(document).unbind('touchmove');
		$(document).unbind('touchend');

		$("#scale").animate({ right: '15px',top: '-15px',
            fontSize: '2em',opacity: '0.0'},500);

		$('#Dial').fadeOut(500);
		$('#line1').fadeOut(550);
		$('#preg_segu').fadeOut(550);
		$('#scale').fadeOut(500);
		$(document).unbind('click');

		
		Trials_save.push(sc.trial)
		Nalternativas_save.push(DataToSave.nstim)
		Distance_ratio_1_2_save.push(DataToSave.distance_21)
		Mu1_save.push(DataToSave.Mu1)
		Mu2_save.push(DataToSave.Mu2)
		X1_save.push(DataToSave.X1)
		X2_save.push(DataToSave.X2)
		X3_save.push(DataToSave.X3)
		Xtarget_save.push(DataToSave.Xtarget)
		Ytarget_save.push(DataToSave.Ytarget)
		Invierte_save.push(DataToSave.invierte)
		SD_save.push(DataToSave.SD)
		Nsamples_save.push(DataToSave.Nsamples)
		Color_clouds_save.push(DataToSave.col_clouds)
		Color_buttons_save.push(DataToSave.col_bots)
		CorrectColor_save.push(DataToSave.CorrectColor)
		RT_type1_save.push(DataToSave.RT)
		Response_save.push(DataToSave.response)
		Correct_save.push(DataToSave.correct)
		Confidence_save.push(DataToSave.confidence)
		RT_Confidence_save.push(DataToSave.reactiontimeconf)

		if (sc.trial == sc.maxtrials ){
			var Data = {"Code": CODE,
					"Trial_number": Trials_save,
					"Nalternativas": Nalternativas_save,
					"Distance_ratio_1_2": Distance_ratio_1_2_save,
					"Mu1": Mu1_save,
					"Mu2": Mu2_save,
					"X1": X1_save,
					"X2": X2_save,
					"X3": X3_save,
					"Xtarget": Xtarget_save,
					"Ytarget": Ytarget_save,
					"invierte": Invierte_save,
					"SD": SD_save,
					"Nsamples": Nsamples_save,
					"color_clouds": Color_clouds_save,
					"color_buttons": Color_buttons_save,
					"CorrectColor": CorrectColor_save,
					"RT_type1": RT_type1_save,
					"Response": Response_save,
					"Correct": Correct_save,
					"Confidence": Confidence_save,
					"RT_Confidence": RT_Confidence_save};
						
			jatos.appendResultData(JSON.stringify(Data));}

			if (sc.trial == sc.maxtrials){

			$('#Question_2').css('display', 'none');
			$('#botones').css('display', 'none');
			$('#Stimuli').css('display', 'none');
			
			$('#preg').html("El estudio ha finalizado. Para dudas y/o consultas podés comunicarte con los investigadores responsables a nicocomay@gmail.com ¡Muchas gracias por tu participación!<br>Si querés podes dejar tus datos para futuros experimentos.<br><button id='finalizar_estudio' class='button_instr'>Finalizar sin dejar datos</button><br><br><button id='base_experimentos' class='button_instr'>Finalizar dejando datos</button><br>¡Sería de gran ayuda que compartas este experimento con otras personas!"); // <button id='finalizar_estudio'>Finalizar estudio</button>"
			$('#preg').css("fontSize", "90%" );
			$('#preg').show();
			$('#Letterhead1').css("top", "-40px");
			$('#Question2').css('display', 'none'); //borramos de la pantalla este div para que no impida tocar el boton de finalizar
			$('#bar1').css('display', 'none');
			$('#bar2').css('display', 'none');
			$('#marks').css('display', 'none');
			$('#finalizar_estudio').click(function(){
				jatos.endStudyAndRedirect("https://google.com.ar/"); //uso endstudyandredirect porque endstudyajax da unos errores de cookies (samesite)
			});	
			$('#base_experimentos').click(function(){
				jatos.endStudyAndRedirect("https://sites.google.com/view/experimentosonline");
			});	


		return}
		setTimeout("$('div.linesConf').hide(0,newtrial(sc))",500);
  })
};
}
