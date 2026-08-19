function show_stim(sc)
{
$('#Dial').hide();
if (AutomaticResponse ==1){setTimeout("$('#myCanvas1').click()",2000)}	// to run alone



if (screen.width  < 400){						     var R =  110;}	// responsive diameter
if ((screen.width >= 400) && (screen.width < 600)) { var R =  135;}
if ((screen.width >= 600) && (screen.width < 900)) { var R =  150;}
if ((screen.width >= 900) && (screen.width < 1200)){ var R =  170;}
if (screen.width  >= 1200){							 var R =  170;}

var angle = Math.random() * Math.PI * 2;	// phase from 0 degrees, so canvas appear shifted every trial	
var stepangle = Math.PI * 2/5; 				// 72 deg step

DataToSave.angle = angle;           
DataToSave.stepangle = stepangle;

var Center = screen.width/2 - document.getElementById('myCanvas1').width/2;

var xx = new Array();
var yy = new Array();

xx[0] = Math.round(Math.cos(angle) * R + Center);								// posicion 1 
yy[0] = Math.round(Math.sin(angle) * R);

xx[1] = Math.round(Math.cos(angle + stepangle) * R + Center);					// posicion 2
yy[1] = Math.round(Math.sin(angle + stepangle) * R);

xx[2] = Math.round(Math.cos(angle + stepangle * 2) * R + Center);				// posicion 3
yy[2] = Math.round(Math.sin(angle + stepangle * 2) * R);

xx[3] = Math.round(Math.cos(angle + stepangle * 3) * R + Center);				// posicion 4
yy[3] = Math.round(Math.sin(angle + stepangle * 3) * R);

xx[4] = Math.round(Math.cos(angle + stepangle * 4) * R + Center);				// posicion 5
yy[4] = Math.round(Math.sin(angle + stepangle * 4) * R);

X6 = Center;																// punto de fijacion				


// randomly assigns canvas to position
var assign = new Array(0,1,2,3,4);
shuffle(assign)

var X1 = xx[assign[0]]
var X2 = xx[assign[1]]
var X3 = xx[assign[2]]
var X4 = xx[assign[3]]
var X5 = xx[assign[4]]

var Y1 = yy[assign[0]]
var Y2 = yy[assign[1]]
var Y3 = yy[assign[2]]
var Y4 = yy[assign[3]]
var Y5 = yy[assign[4]]



$('#myCanvas1').css('left',X1+'px');
$('#myCanvas1').css('top', Y1+'px');
$('#myCanvas2').css('left',X2+'px');
$('#myCanvas2').css('top', Y2+'px');
$('#myCanvas3').css('left',X3+'px');
$('#myCanvas3').css('top', Y3+'px');
$('#myCanvas4').css('left',X4+'px');
$('#myCanvas4').css('top', Y4+'px');
$('#myCanvas5').css('left',X5+'px');
$('#myCanvas5').css('top', Y5+'px');

$('#myCanvasDot').css('left',X6+'px');


$('#myCanvasR1').css('left',X1+'px');
$('#myCanvasR1').css('top', Y1+'px');
$('#myCanvasR2').css('left',X2+'px');
$('#myCanvasR2').css('top', Y2+'px');
$('#myCanvasR3').css('left',X3+'px');
$('#myCanvasR3').css('top', Y3+'px');
$('#myCanvasR4').css('left',X4+'px');
$('#myCanvasR4').css('top', Y4+'px');
$('#myCanvasR5').css('left',X5+'px');
$('#myCanvasR5').css('top', Y5+'px');

 
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
	$("#myCanvas4").hide()
	$("#myCanvas5").hide()
	$("#myCanvasDot").hide()
 	$("#myCanvasR1").show()
	$("#myCanvasR2").show()
	if (sc.nstim[sc.trial-1] >2){$("#myCanvasR3").show()}
	if (sc.nstim[sc.trial-1] >3){$("#myCanvasR4").show()}
	if (sc.nstim[sc.trial-1] >4){$("#myCanvasR5").show()}

}, T2);


$('#myCanvasDot').fadeIn(100);											// show and hide stimuli
setTimeout("$('#myCanvas1').show();",T1);
setTimeout("$('#myCanvas2').show();",T1);
if (sc.nstim[sc.trial-1] >2){ setTimeout("$('#myCanvas3').show();",T1);}
if (sc.nstim[sc.trial-1] >3){ setTimeout("$('#myCanvas4').show();",T1);}
if (sc.nstim[sc.trial-1] >4){ setTimeout("$('#myCanvas5').show();",T1);}


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

	if ($(event.target).is('#myCanvas4') ) { 
		sc.response[sc.trial-1] = 4;
		
		$("#myCanvas4").unbind('click');
		$("#myCanvas4").fadeOut(50);
		$("#myCanvas4").fadeIn(100)
		$("#myCanvas4").fadeOut(150);}
		
	
	if ($(event.target).is('#myCanvas5') ) { 
		sc.response[sc.trial-1] = 5;
		
		$("#myCanvas5").unbind('click');
		$("#myCanvas5").fadeOut(50);
		$("#myCanvas5").fadeIn(100)
		$("#myCanvas5").fadeOut(150);}


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

	if ($(event.target).is('#myCanvasR4') ) { 
		sc.response[sc.trial-1] = 4;
		
		$("#myCanvasR4").unbind('click');
		$("#myCanvasR4").fadeOut(50);
		$("#myCanvasR4").fadeIn(100)
		$("#myCanvasR4").fadeOut(150);}

	if ($(event.target).is('#myCanvasR5') ) { 
		sc.response[sc.trial-1] = 5;
				
		$("#myCanvasR5").unbind('click');
		$("#myCanvasR5").fadeOut(50);
		$("#myCanvasR5").fadeIn(100)
		$("#myCanvasR5").fadeOut(150);}

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
	$("#myCanvas4").fadeOut(500)
	$("#myCanvas5").fadeOut(500)
	$("#myCanvasR1").fadeOut(500)
	$("#myCanvasR2").fadeOut(500)
	$("#myCanvasR3").fadeOut(500)
	$("#myCanvasR4").fadeOut(500)
	$("#myCanvasR5").fadeOut(500)
	$("#myCanvasDot").fadeOut(500)
	$('#preg').fadeOut(500)

	$(".clickresp").unbind('click');
	setTimeout("$('#myCanvas1').hide(0,introspective_response(DataToSave))",500);
});

}
