function show_stim(sc)
{
$('#Dial').hide();
if (AutomaticResponse ==1){setTimeout("$('#myCanvas1').click()",2000)}	// to run alone



if (screen.width  < 400){						     var R =  110;}	// responsive diameter
if ((screen.width >= 400) && (screen.width < 600)) { var R =  135;}
if ((screen.width >= 600) && (screen.width < 900)) { var R =  150;}
if ((screen.width >= 900) && (screen.width < 1200)){ var R =  170;}
if (screen.width  >= 1200){							 var R =  170;}

var angle = Math.random() * Math.PI * 2;							// phase from 0 degrees, so canvas appear shifted every trial	
var stepangle = Math.PI * 2/3 * (Math.round(Math.random()) * 2 -1); // 120 deg step, clockwise or anticlockwise for random stimuli location

DataToSave.angle = angle;           
DataToSave.stepangle = stepangle;

var Center = screen.width/2 - document.getElementById('myCanvas1').width/2;

X1 = Math.round(Math.cos(angle) * R + Center);								// posicion canvas 1 
Y1 = Math.round(Math.sin(angle) * R);
X2 = Math.round(Math.cos(angle + stepangle) * R + Center);					// posicion canvas 2
Y2 = Math.round(Math.sin(angle + stepangle) * R);
X3 = Math.round(Math.cos(angle + stepangle + stepangle) * R + Center);		// posicion canvas 3
Y3 = Math.round(Math.sin(angle + stepangle + stepangle) * R);
X4 = Center;																// punto de fijacion				


// NICO: guardo el número de alternativas de este trial
DataToSave.nstim = sc.nstim[sc.trial-1];                       // guarda el n de items (2 o 3)
/*var Nitems = {"Nalternativas": DataToSave.nstim,
			"Trial_number": DataToSave.trial};	

jatos.appendResultData(JSON.stringify(Nitems));  */

$('#myCanvas1').css('left',X1+'px');
$('#myCanvas1').css('top', Y1+'px');
$('#myCanvas2').css('left',X2+'px');
$('#myCanvas2').css('top', Y2+'px');
$('#myCanvas3').css('left',X3+'px');
$('#myCanvas3').css('top', Y3+'px');

$('#myCanvasDot').css('left',X4+'px');


$('#myCanvasR1').css('left',X1+'px');
$('#myCanvasR1').css('top', Y1+'px');
$('#myCanvasR2').css('left',X2+'px');
$('#myCanvasR2').css('top', Y2+'px');
$('#myCanvasR3').css('left',X3+'px');
$('#myCanvasR3').css('top', Y3+'px');

 
var StartTime = +new Date();											//starts the trial
$('#SoundButton').show()
$('#preg').html('¿Cuál es el más grande?');


if (sc.trial < 3){
	setTimeout("$('#preg').fadeIn(100);",0);
	}

var myVar1 = setTimeout(function(){

 	$("#myCanvas1").hide()
	$("#myCanvas2").hide()
  	$("#myCanvas3").hide()
  	$("#myCanvasDot").hide()
 	$("#myCanvasR1").show()
	$("#myCanvasR2").show()
  	if (sc.nstim[sc.trial-1] ==3){$("#myCanvasR3").show()}

}, T2);


$('#myCanvasDot').fadeIn(100);											// show and hide stimuli
setTimeout("$('#myCanvas1').show();",T1);
setTimeout("$('#myCanvas2').show();",T1);
if (sc.nstim[sc.trial-1] ==3){ setTimeout("$('#myCanvas3').show();",T1);}


$('.clickresp').click(function(event)									// waits for a click/tap
{
	DataToSave.reactiontime	= +new Date() - StartTime - T1;
	clearTimeout(myVar1);												// cancels automatic ending of trial

	if ($(event.target).is('#myCanvas1') ) { 							// checks which canvas was clicked 
		sc.response[sc.trial-1] = 1;	   

		$("#myCanvas1").unbind('click');
		$("#myCanvas1").fadeOut(50);										// blinks					
		$("#myCanvas1").fadeIn(100)
		$("#myCanvas1").fadeOut(150);}
	
	if ($(event.target).is('#myCanvas2') ) { 
		sc.response[sc.trial-1] = 2;
		
		$("#myCanvas2").unbind('click');
		$("#myCanvas2").fadeOut(50);
		$("#myCanvas2").fadeIn(100)
		$("#myCanvas2").fadeOut(150);}
	
	if ($(event.target).is('#myCanvas3') ) { 
		sc.response[sc.trial-1] = 3;
		
		$("#myCanvas3").unbind('click');
		$("#myCanvas3").fadeOut(50);
		$("#myCanvas3").fadeIn(100)
		$("#myCanvas3").fadeOut(150);}

	if ($(event.target).is('#myCanvasR1') ) { 
		sc.response[sc.trial-1] = 1;	   
		
		$("#myCanvasR1").unbind('click');
		$("#myCanvasR1").fadeOut(50);							
		$("#myCanvasR1").fadeIn(100)
		$("#myCanvasR1").fadeOut(150);}
	
	if ($(event.target).is('#myCanvasR2') ) { 
		sc.response[sc.trial-1] = 2;
		
		$("#myCanvasR2").unbind('click');
		$("#myCanvasR2").fadeOut(50);
		$("#myCanvasR2").fadeIn(100)
		$("#myCanvasR2").fadeOut(150);}
	
	if ($(event.target).is('#myCanvasR3') ) { 
		sc.response[sc.trial-1] = 3;
		
		$("#myCanvasR3").unbind('click');
		$("#myCanvasR3").fadeOut(50);
		$("#myCanvasR3").fadeIn(100)
		$("#myCanvasR3").fadeOut(150);}

	sc.correct[sc.trial-1] = sc.response[sc.trial-1] == DataToSave.posbig;    // checks whether the response is correct

		// NICO: acá guardo la respuesta y si fue o no correcta
	DataToSave.response    = sc.response[sc.trial-1];
	DataToSave.correct 	   = sc.correct[sc.trial-1];
	/*var correcta = {"Trial_number": sc.trial,
					"Respuesta": DataToSave.response,
					"Correcta": DataToSave.correct};

	jatos.appendResultData(JSON.stringify(correcta));*/

 	$("#myCanvas1").fadeOut(500)
	$("#myCanvas2").fadeOut(500)
  	$("#myCanvas3").fadeOut(500)
 	$("#myCanvasR1").fadeOut(500)
	$("#myCanvasR2").fadeOut(500)
  	$("#myCanvasR3").fadeOut(500)
	$("#myCanvasDot").fadeOut(500)
	$('#preg').fadeOut(500)

	$(".clickresp").unbind('click');
	setTimeout("$('#myCanvas1').hide(0,introspective_response(DataToSave))",500);
});

}
