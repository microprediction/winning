function create_stim(sc,callback)
{
	if (sc.trial == 1)
	{
		stream1 = new Random();															// http://www.simjs.com/random.html#why-not-math-random

		canvas1 = document.getElementById('myCanvas1');
		canvas2 = document.getElementById('myCanvas2');
		canvas3 = document.getElementById('myCanvas3');
		canvas4 = document.getElementById('myCanvasDot');

		context1 = canvas1.getContext('2d');											// stimuli canvas are global variables
		context2 = canvas2.getContext('2d');
		context3 = canvas3.getContext('2d');
		context4 = canvas4.getContext('2d');

	 	context4.beginPath();															// fixation dot canvas
		context4.arc((canvas4.width/2), (canvas4.height/2), 5, 0, 2 * Math.PI);
		context4.closePath();
		context4.fillStyle = "#000000";
		context4.fill( );
	
		canvasR1 = document.getElementById('myCanvasR1');
		canvasR2 = document.getElementById('myCanvasR2');
		canvasR3 = document.getElementById('myCanvasR3');

		contextR1 = canvasR1.getContext('2d');											// response canvas, global variables
		contextR2 = canvasR2.getContext('2d');
		contextR3 = canvasR3.getContext('2d');

		contextR1.beginPath();				 											// response dots
		contextR1.arc((canvas1.width/2), (canvas1.height/2), 10, 0, 2 * Math.PI);
		contextR1.closePath();
		contextR1.fillStyle = "#ffaa33";
		contextR1.fill( );

		contextR2.beginPath();
		contextR2.arc((canvas1.width/2), (canvas1.height/2), 10, 0, 2 * Math.PI);
		contextR2.closePath();
		contextR2.fillStyle = "#ffaa33";
		contextR2.fill( );

		contextR3.beginPath();
		contextR3.arc((canvas1.width/2), (canvas1.height/2), 10, 0, 2 * Math.PI);
		contextR3.closePath();
		contextR3.fillStyle = "#ffaa33";
		contextR3.fill( );

		/*contextR2.beginPath();
		contextR2.arc((canvas1.width/2), (canvas1.height/2), 10, 0, 2 * Math.PI);
		contextR2.closePath();
		contextR2.fillStyle = "#ffaa33";
		contextR2.fill( ); */

		/*contextR3.beginPath();
		contextR3.arc((canvas1.width/2), (canvas1.height/2), 10, 0, 2 * Math.PI);
		contextR3.closePath();
		contextR3.fillStyle = "#ffaa33";
		contextR3.fill( ); */

		contextR1.strokeStyle="#ffff66";												// write on response canvas, mostly for debugging
		contextR1.font="20px Arial";
		contextR1.textAlign="center"
		contextR1.strokeText("",canvasR1.width/2,canvasR1.height/2 + 5);

		contextR2.strokeStyle="#ffff66";
		contextR2.font="20px Arial";
		contextR2.textAlign="center"
		contextR2.strokeText("",canvasR2.width/2,canvasR2.height/2 + 5);

		contextR3.strokeStyle="#ffff66";
		contextR3.font="20px Arial";
		contextR3.textAlign="center"
		contextR3.strokeText("",canvasR3.width/2,canvasR3.height/2 + 5);
		}


		if (screen.width  < 400){						  	 var AreaSum = 5500}		// responsive areas
		if ((screen.width >= 400) && (screen.width < 600)) { var AreaSum = 6500}
		if ((screen.width >= 600) && (screen.width < 900)) { var AreaSum = 7500}
		if ((screen.width >= 900) && (screen.width < 1200)){ var AreaSum = 7500}
		if (screen.width  >= 1200){							 var AreaSum = 7500}

		var trial_stimval	 = sc.stimvalues[sc.trial - 1]
		var stimval_stim3 	 = sc.tomultiply[sc.trial-1]
		var trial_nstim      = sc.nstim[sc.trial-1]
		
		DataToSave.stimval 	 = trial_stimval
		DataToSave.stimval_3 = stimval_stim3
		
		//----------------------set areas ----------
		var area1 = 0;
		while((area1 < 2000) || (area1 > 14000)) {area1 = AreaSum + stream1.normal(0,1) * AreaSum/4;}
		var area2 = area1 * trial_stimval;	
		var area3 = area2 * stimval_stim3;
		//----------------------set areas ----------
		
		DataToSave.area1 = area1;
		DataToSave.area2 = area2;
		DataToSave.area3 = area3;
		DataToSave.circleorsquare = (trial_nstim > 2) ? Math.round(Math.random()) + 1 : 0; 		// if there's a third stimulus, a circle or a square, otherwise put 0
		DataToSave.whichisbigger  = sc.whichisbigger[sc.trial-1];                     	// bigger the square or the circle

		if (sc.whichisbigger[sc.trial-1] ==1){  				// ==2 						// bigger the circle
			var Radius 	= Math.sqrt( area1 / Math.PI );
			var Lado2 	= Math.sqrt(area2)}
		else {  																		// bigger the square
			var Radius  = Math.sqrt( area2 / Math.PI);
			var Lado2 	= Math.sqrt(area1)}


	    context1.clearRect(0, 0, canvas1.width,canvas1.height);
		context2.clearRect(0, 0, canvas2.width,canvas2.height);
    	context3.clearRect(0, 0, canvas3.width,canvas3.height);

		context1.beginPath();															 // plots the circle, always in context1
		context1.arc((canvas1.width/2), (canvas1.height/2), Radius, 0, 2 * Math.PI);
		context1.closePath();
		context1.fillStyle = "#4CAF50";
		context1.fill( );

		context2.beginPath();															// plots the square, always in context2
		context2.rect((canvas2.width/2) - Lado2/2, (canvas2.height/2) - Lado2/2, Lado2, Lado2);
		context2.closePath();
		context2.fillStyle = "#4CAF50";
		context2.fill( );

    if (DataToSave.circleorsquare ==1){  												// plots the distractor, always in context3
    	var Lado3 	= Math.sqrt(area3); 												// (may be a circle or a square)
        context3.beginPath();
		context3.rect((canvas3.width/2) - Lado3/2, (canvas3.height/2) - Lado3/2, Lado3, Lado3);
		context3.closePath();
		context3.fillStyle = "#4CAF50";
		context3.fill( );}
    else
    {
        var Radius 	= Math.sqrt( area3 / Math.PI );
        context3.beginPath();
        context3.arc((canvas3.width/2), (canvas3.height/2), Radius, 0, 2 * Math.PI);
        context3.closePath();
        context3.fillStyle = "#4CAF50";
        context3.fill( );
		}

		if (sc.whichisbigger[sc.trial-1] ==1){ 											// bigger the circle, correct response in context1
			DataToSave.posbig = 1}
		else{
			DataToSave.posbig = 2}  				  			        				// bigger the square, correct response in context2

}
