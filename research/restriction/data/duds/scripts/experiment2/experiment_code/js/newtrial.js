function  newtrial(sc){
window.scrollTo(0, 1);
//$('#Letterhead2').show();
$('#Dial').css("background","rgb(255,150,0)");
$('#Dial').hide();
$(document).unbind("click");

var Name 		= DataToSave.name;
DataToSave 		= [];						// clear variables to save from previous trials
DataToSave.code = CODE; 
DataToSave.name = Name;

sc.trial 		= sc.trial + 1;             // increment the total trial count
DataToSave.trial= sc.trial;


/*---------------- porcentaje de avance---------*/
var Rate = 100* sc.trial / sc.maxtrials;

console.log(Rate)
if (Rate < 1){Rate = 1;}

$('#Letterhead2').html('grado de avance: '+ Math.floor(Rate).toString().concat(' %'));
$('#Letterhead2').show();

   
/*----------------- nuevo trial o pausa------*/
if ((sc.trial % 40)==0){  
    $('#preg').html('Descansa. Click para continuar');
    $('#preg').show()
    $(document).click(function(pplp){
        $(document).unbind('click');
        setTimeout( function(){create_stim(sc,show_stim(sc));},800); //callback + settimeout to be sure the stimuli are ready
        $('#preg').fadeOut(500)})   
}else{
    setTimeout( function(){create_stim(sc,show_stim(sc));},800);}




}