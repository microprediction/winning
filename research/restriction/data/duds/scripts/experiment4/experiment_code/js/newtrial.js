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

setTimeout( function(){create_stim(sc,show_stim(sc));},800); //callback + settimeout to be sure the stimuli are ready
}
