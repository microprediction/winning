function create_stim(sc,DataToSave)
{
	
	document.getElementById('MainCanvas').height = screen.height/1.2;		// canvas size 1.3
	document.getElementById('MainCanvas').width = screen.width;

	var CenterXX = screen.width/2 - document.getElementById('MainCanvas').width/2;	 // coordenadas para centrar el canvas	
	$('#MainCanvas').css({position: 'absolute',left: CenterXX+'px'});
	$('#MainCanvas').css({position: 'absolute',top: -50+'px'});

//	$('#button_1').css('left', screen.width/2 - 90 +'px');
//	$('#button_2').css('left', screen.width/2 - 10 +'px');
//	$('#button_3').css('left', screen.width/2 + 70 +'px');

	
	/* -----------'responsive' parameters--------------------*/
	if (screen.width  < 400){								 var SD = 15; var Nsamples = 280;var samplesize = 1.5; var Target_exterior_size = 4; var Target_interior_size = 2;	$("#preg").css("fontSize", "20px");$("#preg_segu").css("fontSize", "20px");	}
	if ((screen.width >= 400)  && (screen.width < 600)) {	 var SD = 25; var Nsamples = 280;var samplesize = 1.5; var Target_exterior_size = 4; var Target_interior_size = 2;  $("#preg").css("fontSize", "30px");$("#preg_segu").css("fontSize", "30px");		}
	if ((screen.width >= 600)  && (screen.width < 900)) {	 var SD = 35; var Nsamples = 375;var samplesize = 2; var Target_exterior_size = 6; var Target_interior_size = 3;  	$("#preg").css("fontSize", "30px");$("#preg_segu").css("fontSize", "30px");		}
	if ((screen.width >= 900)  && (screen.width < 1200)){	 var SD = 40; var Nsamples = 375;var samplesize = 2; var Target_exterior_size = 6; var Target_interior_size = 3;  	$("#preg").css("fontSize", "30px");$("#preg_segu").css("fontSize", "30px");		}
	if ((screen.width >= 1200) && (screen.width < 1700)){	 var SD = 60; var Nsamples = 375;var samplesize = 3; var Target_exterior_size = 6; var Target_interior_size = 3;  	$("#preg").css("fontSize", "30px");$("#preg_segu").css("fontSize", "30px");		}
	if ((screen.width >= 1700) && (screen.width < 2000)){	 var SD = 60; var Nsamples = 375;var samplesize = 3; var Target_exterior_size = 6; var Target_interior_size = 3;  	$("#preg").css("fontSize", "30px");$("#preg_segu").css("fontSize", "30px");		}
	if (screen.width >= 2000) {							 	 var SD = 60; var Nsamples = 375;var samplesize = 3; var Target_exterior_size = 6; var Target_interior_size = 3;  	$("#preg").css("fontSize", "30px");$("#preg_segu").css("fontSize", "30px");		}
	/* -------------------------------------------------*/

	if (screen.width  < 400){						     var R =  110;}	// responsive diameter
	if ((screen.width >= 400) && (screen.width < 600)) { var R =  135;}
	if ((screen.width >= 600) && (screen.width < 900)) { var R =  110;}
	if ((screen.width >= 900) && (screen.width < 1200)){ var R =  110;}
	if (screen.width  >= 1200){							 var R =  110;}
	
	DataToSave.SD 		 = SD;
	DataToSave.Nsamples  = Nsamples;
	DataToSave.Target_interior_size = Target_interior_size;
	DataToSave.Target_exterior_size = Target_exterior_size;

	var CenterXCanvas = document.getElementById('MainCanvas').width/2;   // coordenadas del centro del canvas, para plot de puntos
	var CenterYCanvas = document.getElementById('MainCanvas').height/2;

	var distance_21 = sc.dist_12_values[sc.trial-1];
	DataToSave.distance_21 = distance_21;
	DataToSave.nstim = sc.nstim[sc.trial-1];

	var variance = Math.pow(SD,2)

	var angle = Math.random() * Math.PI * 2;							// phase from 0 degrees, so canvas appear shifted every trial	
	var stepangle = -Math.PI * 2/3;										// 120 deg step
	
	DataToSave.angle = angle;           

	X1 = Math.round(Math.cos(0) * R + CenterXCanvas);								// posicion canvas 1 
	Y1 = Math.round(Math.sin(0) * R + CenterYCanvas);
	X2 = Math.round(Math.cos(0 + stepangle) * R + CenterXCanvas);					// posicion canvas 2
	Y2 = Math.round(Math.sin(0 + stepangle) * R + CenterYCanvas);
	X3 = Math.round(Math.cos(0 + stepangle + stepangle) * R * 2.5 + CenterXCanvas);		// posicion canvas 3
	Y3 = Math.round(Math.sin(0 + stepangle + stepangle) * R * 2.5 + CenterYCanvas);

	
	var CorrectCloud = (Math.round(Math.random()) + 1);  // close cloud will be 1 or 2.
	DataToSave.CorrectCloud = CorrectCloud;

	if (CorrectCloud ==1){
		Xtarget = Math.max(X1,X2) - (Math.abs(X1 - X2)) / distance_21
		Ytarget = Math.max(Y1,Y2) - (Math.abs(Y1 - Y2)) / distance_21}

	if (CorrectCloud ==2){
		Xtarget = Math.min(X1,X2) + (Math.abs(X1 - X2)) / distance_21
		Ytarget = Math.min(Y1,Y2) + (Math.abs(Y1 - Y2)) / distance_21}

	DataToSave.Xtarget = Xtarget;
	DataToSave.Ytarget = Ytarget;

	console.log('x target:', Xtarget);
	console.log('y target:', Ytarget);
	console.log('distance:' , distance_21);

	/*-----------*/
	var gaussian = GaussDist();

	var distributionX1 = gaussian(X1,variance);		// distributions for the first gaussian
	var distributionY1 = gaussian(Y1,variance);

	var distributionX2 = gaussian(X2,variance);		// distributions for the second gaussian
	var distributionY2 = gaussian(Y2,variance);

	var distributionX3 = gaussian(X3,variance);		// distributions for the third gaussian
	var distributionY3 = gaussian(Y3,variance);

	
	/*-----get the distributions and plot------*/
	var Samples1 =[];
	var Samples2 =[];
	var Samples3 =[];
	var Samples4 =[];
	var Samples5 =[];
	var Samples6 =[];

	for (var a =0; a < Nsamples; a++){

		Samples1.push( distributionX1.ppf(Math.random()) );		// sample all distibutions
		Samples2.push( distributionY1.ppf(Math.random()) );
		Samples3.push( distributionX2.ppf(Math.random()) );
		Samples4.push( distributionY2.ppf(Math.random()) );
		Samples5.push( distributionX3.ppf(Math.random()) );
		Samples6.push( distributionY3.ppf(Math.random()) );
	}

	var maincanvas = document.getElementById('MainCanvas');
	var context_canvas = maincanvas.getContext('2d');

	context_canvas.clearRect(0, 0, maincanvas.width,maincanvas.height);
	var ox = maincanvas.width / 2;
	var oy = maincanvas.height / 2;
	context_canvas.translate(ox, oy);
	context_canvas.rotate(angle);
	context_canvas.translate(-ox, -oy);

		 // canvas color, to check size and position
		// var c = document.getElementById("MainCanvas");var ctx = c.getContext("2d");ctx.globalAlpha = 0.2;ctx.fillStyle = "#88f2a4";ctx.fillRect(0, 0, maincanvas.width, maincanvas.height);	ctx.globalAlpha = 1;
		 						

	var colores = ['e62315','0cc90f','2112c7'];
	DataToSave.col_clouds = shuffle([0,1,2]); 
	DataToSave.col_bots = [0,1,2];
	
	DataToSave.CorrectColor = DataToSave.col_clouds[DataToSave.CorrectCloud - 1];		// color of the correct gaussian
	
	console.log('correct cloud:', CorrectCloud);
	

	document.getElementById("button_1").style.background='#' + colores[DataToSave.col_bots[0]];
	document.getElementById("button_2").style.background='#' + colores[DataToSave.col_bots[1]];
	document.getElementById("button_3").style.background='#' + colores[DataToSave.col_bots[2]];
	
	var ix1 = 0; var iy1 = 0;
	var ix2 = 0; var iy2 = 0;
	var ix3 = 0; var iy3 = 0;

	for (var iSample =0; iSample < Nsamples; iSample++){ 
		
		ix1 = Samples1[iSample];
		iy1 = Samples2[iSample];
		
		ix2 = Samples3[iSample];
		iy2 = Samples4[iSample];
		
		if (sc.nstim[sc.trial-1] ==3){
			ix3 = Samples5[iSample];
			iy3 = Samples6[iSample];}

		context_canvas.beginPath();
		context_canvas.arc(ix1, iy1, samplesize, 0, 2 * Math.PI);
		context_canvas.closePath();
		context_canvas.fillStyle = '#' + colores[DataToSave.col_clouds[0]];
		context_canvas.fill( );


		context_canvas.beginPath();
		context_canvas.arc(ix2, iy2, samplesize, 0, 2 * Math.PI);
		context_canvas.closePath();
		context_canvas.fillStyle = '#' + colores[DataToSave.col_clouds[1]];
		context_canvas.fill( );
		

		if (sc.nstim[sc.trial-1] ==3){
			context_canvas.beginPath();
			context_canvas.arc(ix3, iy3, samplesize, 0, 2 * Math.PI);
			context_canvas.closePath();
			context_canvas.fillStyle = '#' + colores[DataToSave.col_clouds[2]];
			context_canvas.fill( );}
	}

	// plot the target
	context_canvas.beginPath();
	context_canvas.arc(Xtarget, Ytarget, Target_exterior_size, 0, 2 * Math.PI);
	context_canvas.closePath();
	context_canvas.fillStyle = "#FFFF00";
	context_canvas.fill( );

	context_canvas.beginPath();
	context_canvas.arc(Xtarget, Ytarget, Target_interior_size, 0, 2 * Math.PI);
	context_canvas.closePath();
	context_canvas.fillStyle = "#000000";
	context_canvas.fill( );

	setTimeout("$('#MainCanvas').hide(0,show_stim(sc,DataToSave))",100);

}
