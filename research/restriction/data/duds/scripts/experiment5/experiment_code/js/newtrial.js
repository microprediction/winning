function  newtrial(sc){
window.scrollTo(0, 1);
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

if (Rate < 1){Rate = 1;}

$('#Letterhead2').html('grado de avance: '+ Math.floor(Rate).toString().concat(' %'));
$('#Letterhead2').show();


/*----------------- nuevo trial o pausa------*/
if ((sc.trial % 60)==0 && (sc.trial < 100)){  
    $('#preg').html('Si querés podés descansar un ratito. Cuando estés listo/a, hacé click para continuar');
    $('#preg').show()
    $(document).click(function(pplp){
        $(document).unbind('click');
        setTimeout("$('#MainCanvas').hide(0,create_stim(sc,DataToSave))",800);
        $('#preg').fadeOut(500)})   
}else{
    setTimeout("$('#MainCanvas').hide(0,create_stim(sc,DataToSave))",800);}







}






   

