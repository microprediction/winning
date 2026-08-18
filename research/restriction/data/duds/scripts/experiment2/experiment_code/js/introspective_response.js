function introspective_response(DataToSave)
{

var LIMMIN = Math.ceil(screen.width/2) - Math.ceil($('#line1').width()/2) - 10;
var LIMMAX = Math.ceil(screen.width/2) + Math.ceil($('#line1').width()/2) - 10;
var range = LIMMAX - LIMMIN;

///////////////////////////////////////////////////////////
/*var start_scale = (range * .4 ) +  Math.random() * (range * .2 );

var percent_start = (100 * (start_scale/range))
percent_start = percent_start.toFixed(0)


$('#scale').html(percent_start.toString().concat(' %'))
$('#scale').css({"left": start_scale - LIMMIN})
$('#Dial').css({"left": start_scale + LIMMIN});*/
/////////////////////////////////////////////////////////////


$('#scale').css('color','#004D00')
if (sc.trial > 1)   													// reset the animation from the previous trial
{
	$("#scale").animate({ right: '-15px',top: '15px', fontSize: '1em',opacity: '1.0'},10);
}

$('#line1').show();
if (sc.trial < 4){$('#preg_segu').show();}

var x = 0;  
var StartTime = +new Date();
var percent = 0;

function updateMouseDial(e){
	x = e.pageX;
	if (e.pageX >= LIMMAX){x = LIMMAX};
	if (e.pageX <= LIMMIN){x = LIMMIN};

	percen = (100 * (x - LIMMIN)/range).toFixed(0)					// get percentaje 
	$('#scale').html(percen.toString().concat(' %'))
	$('#scale').css({"left": x- Math.ceil(screen.width/2)})
	$('#Dial').css({"left": x});
}

/*------------------mouse ---------*/
if (InputMethod == 'Mouse') (InputMethod == 'Pad')
{	
	if (AutomaticResponse ==1){setTimeout("$('#Dial').click()",1000)}	// to run alone

	
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

		setTimeout("$('#Dial').css('background','#ff99ce')",100);		// blinks
		setTimeout("$('#Dial').css('background','#99004f')",400);

		$('#Dial').fadeOut(500);
		$('#line1').fadeOut(550);
		$('#preg_segu').fadeOut(550);
		$('#scale').fadeOut(500);
		$(document).unbind('click');
		
		Trials_save.push(sc.trial)
		Angle_save.push(DataToSave.angle)
		StepAngle_save.push(DataToSave.stepangle)
		Area1_save.push(DataToSave.area1)
		Area2_save.push(DataToSave.area2)
		Area3_save.push(DataToSave.area3)
		Area4_save.push(DataToSave.area4)
		Area5_save.push(DataToSave.area5)
		ThirdSquareOrCircle_save.push(DataToSave.ThirdCircleOrSquare)
		FourthSquareOrCircle_save.push(DataToSave.FourthCircleOrSquare)
		FifthSquareOrCircle_save.push(DataToSave.FifthCircleOrSquare)
		BiggerCircleOrSquare_save.push(DataToSave.whichisbigger)
		PosBig_save.push(DataToSave.posbig)
		StimVal_save.push(DataToSave.stimval)
		StimVal3_save.push(DataToSave.stimval_3)
		Nalternativas_save.push(DataToSave.nstim)
		RT_type1_save.push(DataToSave.reactiontime)
		Response_save.push(DataToSave.response)
		Correct_save.push(DataToSave.correct)
		Confidence_save.push(DataToSave.confidence)
		RT_Confidence_save.push(DataToSave.reactiontimeconf)

		if (sc.trial == sc.maxtrials ){
			var Data = {"Code": CODE,
					"Trial_number": Trials_save,
					"Angle": Angle_save,
					"StepAngle": StepAngle_save,
					"Area1": Area1_save,
					"Area2": Area2_save,
					"Area3": Area3_save,
					"Area4": Area4_save,
					"Area5": Area5_save,
					"3rdSquareOrCircle": ThirdSquareOrCircle_save, 
					"4thSquareOrCircle": FourthSquareOrCircle_save,
					"5thSquareOrCircle": FifthSquareOrCircle_save, 
					"BiggerCircleOrSquare": BiggerCircleOrSquare_save,
					"PosBig": PosBig_save,
					"StimVal": StimVal_save,
					"StimVal3": StimVal3_save, 
					"Nalternativas": Nalternativas_save,
					"RT_type1": RT_type1_save,
					"Response": Response_save,
					"Correct": Correct_save,
					"Confidence": Confidence_save,
					"RT_Confidence": RT_Confidence_save};
						
			jatos.appendResultData(JSON.stringify(Data));}


		if (sc.trial == sc.maxtrials){
			$('#preg').html("El estudio ha finalizado. Para dudas y/o consultas podés comunicarte con los investigadores responsables a nicocomay@gmail.com ¡Muchas gracias por tu participación!<br>Si querés podes dejar tus datos para futuros experimentos.<br><button id='finalizar_estudio' class='button_instr'>Finalizar sin dejar datos</button><br><br><button id='base_experimentos' class='button_instr'>Finalizar dejando datos</button><br>¡Sería de gran ayuda que compartas este experimento con otras personas!"); // <button id='finalizar_estudio'>Finalizar estudio</button>"
			$('#preg').css("fontSize", "30px" );
			//$('#preg').html("<button class = 'button_instr' id='finalizar_estudio'>Finalizar estudio</button>"); // onclick = 'jatos.endStudyAjax();'
			$('#preg').show();
			//$('#finalizar_estudio').show();
			$('#Question2').css('display', 'none'); //borramos de la pantalla este div para que no impida tocar el boton de finalizar
			$('#bar1').css('display', 'none');
			$('#bar2').css('display', 'none');
			$('#marks').css('display', 'none');
			$('#finalizar_estudio').click(function(){
				jatos.endStudyAndRedirect("https://google.com.ar/"); //uso endstudyandredirect porque endstudyajax da unos errores de cookies (samesite)
			});	
			$('#base_experimentos').click(function(){
				jatos.endStudyAndRedirect("https://groups.google.com/forum/#!forum/c3online/join");
			});	

		return}

		/*var total=0;
		for(var i in sc.correct) { total += sc.correct[i];}				// checks how many correct responses so far
		if (total % CT ==0 && !(total ==0) && !(total == AcumTotal)){   // every CT correct responses, next level
			AcumTotal = total;											
			Level = Level + 1;
		//	$('#Letterhead2').html('Nivel: '+ Level.toString());
		//	var sentence = '¡Pasaste al nivel '.concat(Level.toString())
		//	$("#preg").html(sentence.concat('!'));
		//	$("#preg").fadeIn(300);
		//	setTimeout("$('#preg').fadeOut(200);$('div.linesConf').hide(0,newtrial(sc))",2500)}
		//else{*/
		setTimeout("$('div.linesConf').hide(0,newtrial(sc))",1500);

  });
}

/*------------------touchscreen ---------*/
if (InputMethod == 'Touchscreen')
{	
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

		setTimeout("$('#Dial').css('background','#ff99ce')",100);
		setTimeout("$('#Dial').css('background','#99004f')",400);

		$('#Dial').fadeOut(500);
		$('#line1').fadeOut(550);
		$('#preg_segu').fadeOut(550);
		$('#scale').fadeOut(500);
		$(document).unbind('click');

		Trials_save.push(sc.trial)
		Angle_save.push(DataToSave.angle)
		StepAngle_save.push(DataToSave.stepangle)
		Area1_save.push(DataToSave.area1)
		Area2_save.push(DataToSave.area2)
		Area3_save.push(DataToSave.area3)
		Area4_save.push(DataToSave.area4)
		Area5_save.push(DataToSave.area5)
		ThirdSquareOrCircle_save.push(DataToSave.ThirdCircleOrSquare)
		FourthCircleOrSquare_save.push(DataToSave.FourthCircleOrSquare)
		FifthCircleOrSquare_save.push(DataToSave.FifthCircleOrSquare)
		BiggerCircleOrSquare_save.push(DataToSave.whichisbigger)
		PosBig_save.push(DataToSave.posbig)
		StimVal_save.push(DataToSave.stimval)
		StimVal3_save.push(DataToSave.stimval_3)
		Nalternativas_save.push(DataToSave.nstim)
		RT_type1_save.push(DataToSave.reactiontime)
		Response_save.push(DataToSave.response)
		Correct_save.push(DataToSave.correct)
		Confidence_save.push(DataToSave.confidence)
		RT_Confidence_save.push(DataToSave.reactiontimeconf)

		

		if (sc.trial == sc.maxtrials ){
			var Data = {"Code": CODE,
					"Trial_number": Trials_save,
					"Angle": Angle_save,
					"StepAngle": StepAngle_save,
					"Area1": Area1_save,
					"Area2": Area2_save,
					"Area3": Area3_save,
					"Area4": Area4_save,
					"Area5": Area5_save,
					"3rdSquareOrCircle": ThirdSquareOrCircle_save, 
					"4thSquareOrCircle": FourthSquareOrCircle_save, 
					"5thSquareOrCircle": FifthSquareOrCircle_save, 
					"BiggerCircleOrSquare": BiggerCircleOrSquare_save,
					"PosBig": PosBig_save,
					"StimVal": StimVal_save,
					"StimVal3": StimVal3_save, 
					"Nalternativas": Nalternativas_save,
					"RT_type1": RT_type1_save,
					"Response": Response_save,
					"Correct": Correct_save,
					"Confidence": Confidence_save,
					"RT_Confidence": RT_Confidence_save};
						
			jatos.appendResultData(JSON.stringify(Data));}


		if (sc.trial == sc.maxtrials){
			
			$('#preg').html("El estudio ha finalizado. Para dudas y/o consultas podés comunicarte con los investigadores responsables a nicocomay@gmail.com ¡Muchas gracias por tu participación!<br>Si querés podes dejar tus datos para futuros experimentos.<br><button id='finalizar_estudio' class='button_instr'>Finalizar sin dejar datos</button><br><br><button id='base_experimentos' class='button_instr'>Finalizar dejando datos</button><br>¡Sería de gran ayuda que compartas este experimento con otras personas!"); // <button id='finalizar_estudio'>Finalizar estudio</button>"
			$('#preg').css("fontSize", "90%" );
			//$('#preg').html("<button class = 'button_instr' id='finalizar_estudio'>Finalizar estudio</button>"); // onclick = 'jatos.endStudyAjax();'
			$('#preg').show();
			$('#Letterhead1').css("top", "-40px");
			//$('#finalizar_estudio').show();
			$('#Question2').css('display', 'none'); //borramos de la pantalla este div para que no impida tocar el boton de finalizar
			$('#bar1').css('display', 'none');
			$('#bar2').css('display', 'none');
			$('#marks').css('display', 'none');
			$('#finalizar_estudio').click(function(){
				jatos.endStudyAndRedirect("https://google.com.ar/"); //uso endstudyandredirect porque endstudyajax da unos errores de cookies (samesite)
			});	
			$('#base_experimentos').click(function(){
				jatos.endStudyAndRedirect("https://groups.google.com/forum/#!forum/c3online/join");
			});	


		return}
		
		/*var total=0;
		for(var i in sc.correct) { total += sc.correct[i];}				// checks how many correct responses so far
		if (total % CT ==0 && !(total ==0) && !(total == AcumTotal)){   // every CT correct responses, next level
			AcumTotal = total;											
			Level = Level + 1;
		//	$('#Letterhead2').html('Nivel: '+ Level.toString());
		//	var sentence = '¡Pasaste al nivel '.concat(Level.toString())
		//	$("#preg").html(sentence.concat('!'));
		//	$("#preg").fadeIn(300);
			setTimeout("$('#preg').fadeOut(200);$('div.linesConf').hide(0,newtrial(sc))",2500)}
		else{*/
		setTimeout("$('div.linesConf').hide(0,newtrial(sc))",1500);

  })
};
}
